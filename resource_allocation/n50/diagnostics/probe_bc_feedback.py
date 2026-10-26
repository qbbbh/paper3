"""Two-arm learning probe: behaviour cloning, then gated continuous feedback.

Arm 1 (``bc_only``) is supervised on the reference action at one frozen block
and is evaluated by generating its *own* complete actions.

Arm 2 (``bc_masac``) starts from the identical cloned actor, fills replay with
a balanced set (reference, improved legal neighbours, random-policy coverage
actions and cloned-policy proposals), fits the critic in a critic-only warm-up,
and only then allows actor steps.  Every new proposal is scored by the same
frozen forecast bank, labelled with its own predicted *mean per-link EE*
(``mean_k rate_k / power_k``, the paper's definition) and budget gap on the
same quantity, and appended to replay.  Per-link minimum EE and network EE are
still reported, but only as diagnostics.  If the critic cannot be validated on
held-out actions, the actor gate stays closed and the run reports that instead
of degrading.

This is a single-state capacity/plumbing experiment.  It is not a training
result and it cannot rank algorithms.  The output directory must be fresh.
"""
import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from p3learn.allocation import Action, copy_action, protect_budget
from p3learn.credit import legal_replacements
from p3learn.features import NODE_DIM, EDGE_DIM
from p3learn.masac import (HybridMASAC, SACConfig, latent_from_power,
                           power_from_latent)
from p3learn.models import ModelConfig
from probe_common import (action_metrics, build_frozen_block, jsonable,
                          power_summary, truth_metrics, truth_window)


def ensemble_q(learner, features, action):
    with torch.no_grad():
        return float(np.mean([float(q(features, copy_action(action)))
                              for q in learner.critics]))


def critic_report(learner, features, actions, labels):
    qs = [ensemble_q(learner, features, a) for a in actions]
    ys = [l["mean_ee"] / learner.config.reward_scale - l["cost"] for l in labels]
    mae = float(np.mean(np.abs(np.asarray(qs) - np.asarray(ys))))
    agree = total = 0
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            if abs(ys[i] - ys[j]) < 1e-12:
                continue
            total += 1
            agree += (qs[i] - qs[j]) * (ys[i] - ys[j]) > 0
    feasible = [q for q, l in zip(qs, labels) if l["cost"] == 0.0]
    infeasible = [q for q, l in zip(qs, labels) if l["cost"] > 0.0]
    return dict(n=len(qs), mae=mae,
                concordance=float(agree / total) if total else None,
                pairs=total, q_mean=float(np.mean(qs)), q_min=min(qs), q_max=max(qs),
                q_spread=float(max(qs) - min(qs)),
                mean_q_feasible=float(np.mean(feasible)) if feasible else None,
                mean_q_infeasible=float(np.mean(infeasible)) if infeasible else None)


def with_reference_powers(action, reference):
    """Same matching, reference (floor) powers: isolates structure from power."""
    return Action(list(action.links),
                  np.array(reference.powers, dtype=float).copy(),
                  {k: np.array(v, copy=True) for k, v in action.ports.items()})


def pin_floor_power(learner, latent=-12.0, log_std=-5.0):
    """Hard-project the actor's power channel onto the 1 mW floor.

    ``SharedHybridActor.head`` ends in ``Linear(hidden, 3)``: channel 0 is the
    power latent mean, 1 the discrete scores, 2 the log_std.  Overwriting rows
    0 and 2 (and their biases) after every optimiser step makes power constant
    and state-independent while leaving the discrete head untouched.

    This is a *diagnostic* branch only.  It removes power drift from the
    comparison; it does not make discrete exploration easier, because the
    discrete policy and its sampling probabilities are unchanged.
    """
    linear = learner.actor.head[-1]
    with torch.no_grad():
        linear.weight[0].zero_()
        linear.bias[0].fill_(float(latent))
        linear.weight[2].zero_()
        linear.bias[2].fill_(float(log_std))
    return learner


def evaluate_policy(learner, frozen, delta, samples, heldout_actions=None,
                    window=None):
    """Generate actions with the policy itself and score them by forecast.

    ``window`` optionally adds executed (truth) metrics for the same actions.
    It is validation-only and must be read after the block's decisions are
    fixed; it never affects selection.
    """
    deterministic = learner.select_action(frozen.features, deterministic=True)
    det_metrics = action_metrics(frozen.bank, deterministic.action, frozen.reference.eta_ref, delta)
    # Control: the same matching at the reference's 1 mW floor.  A gap here is
    # structural; a gap in det_metrics that disappears here is power drift.
    det_floor = action_metrics(frozen.bank,
                               with_reference_powers(deterministic.action,
                                                     frozen.reference.action),
                               frozen.reference.eta_ref, delta)
    det_protected = protect_budget(deterministic.action, frozen.reference, delta=delta,
                                   n=frozen.scene.n, cfg=frozen.cfg, candidates=frozen.candidates,
                                   business=frozen.business, evaluator=frozen.bank.evaluate)
    det_executed = action_metrics(frozen.bank, det_protected.action,
                                  frozen.reference.eta_ref, delta)
    reference_links = {(int(tx), int(rx), mode) for tx, rx, mode in frozen.reference.action.links}
    proposals, labels, executed, fallbacks, floor_labels = [], [], [], [], []
    for _ in range(samples):
        sample = learner.select_action(frozen.features)
        label = action_metrics(frozen.bank, sample.action, frozen.reference.eta_ref, delta)
        protected = protect_budget(sample.action, frozen.reference, delta=delta,
                                   n=frozen.scene.n, cfg=frozen.cfg,
                                   candidates=frozen.candidates, business=frozen.business,
                                   evaluator=frozen.bank.evaluate)
        executed.append(action_metrics(frozen.bank, protected.action,
                                       frozen.reference.eta_ref, delta))
        floor_labels.append(action_metrics(
            frozen.bank, with_reference_powers(sample.action, frozen.reference.action),
            frozen.reference.eta_ref, delta))
        fallbacks.append(bool(protected.fallback))
        proposals.append(sample.action)
        labels.append(label)
    powers = np.concatenate([np.asarray(l["powers"]) for l in labels])
    summary = dict(
        deterministic=dict(metrics={k: v for k, v in det_metrics.items() if k != "powers"},
                           structural_match_with_reference=(
                               {(int(tx), int(rx), mode) for tx, rx, mode in
                                deterministic.action.links} == reference_links),
                           floor_power_control={k: v for k, v in det_floor.items()
                                                if k != "powers"},
                           fallback=bool(det_protected.fallback),
                           executed={k: v for k, v in det_executed.items() if k != "powers"}),
        stochastic=dict(
            n=samples,
            mean_ee=float(np.mean([l["mean_ee"] for l in labels])),
            median_ee=float(np.median([l["mean_ee"] for l in labels])),
            worst_candidate_mean_ee=float(np.min([l["mean_ee"] for l in labels])),
            mean_min_pair_ee=float(np.mean([l["min_ee"] for l in labels])),
            mean_network_ee=float(np.mean([l["network_ee"] for l in labels])),
            mean_budget_cost=float(np.mean([l["cost"] for l in labels])),
            floor_power_control_mean_ee=float(np.mean([l["mean_ee"] for l in floor_labels])),
            floor_power_control_mean_budget_cost=float(np.mean([l["cost"] for l in floor_labels])),
            acceptance_rate=float(1.0 - np.mean(fallbacks)),
            executed_mean_ee=float(np.mean([e["mean_ee"] for e in executed])),
            executed_mean_min_pair_ee=float(np.mean([e["min_ee"] for e in executed])),
            executed_mean_network_ee=float(np.mean([e["network_ee"] for e in executed])),
            executed_mean_budget_cost=float(np.mean([e["cost"] for e in executed])),
            power=power_summary(powers)))
    if heldout_actions:
        summary["heldout_critic"] = critic_report(learner, frozen.features,
                                                  heldout_actions[0], heldout_actions[1])
    if window is not None:
        # Truth is read only after every decision above is fixed.  Forecast and
        # executed columns stay separate so a prediction gap cannot be hidden
        # inside an "actual performance" number.
        def truth(action):
            return {k: v for k, v in truth_metrics(
                frozen, window, action, frozen.reference.eta_ref, delta).items()
                if k != "powers"}
        sample_truth = [truth(a) for a in proposals]
        summary["deterministic"]["truth"] = truth(deterministic.action)
        summary["deterministic"]["truth_floor_power_control"] = truth(
            with_reference_powers(deterministic.action, frozen.reference.action))
        summary["deterministic"]["truth_executed"] = truth(det_protected.action)
        summary["stochastic"].update(
            truth_mean_ee=float(np.mean([t["mean_ee"] for t in sample_truth])),
            truth_median_ee=float(np.median([t["mean_ee"] for t in sample_truth])),
            truth_worst_candidate_mean_ee=float(np.min([t["mean_ee"] for t in sample_truth])),
            truth_mean_min_pair_ee=float(np.mean([t["min_ee"] for t in sample_truth])),
            truth_mean_network_ee=float(np.mean([t["network_ee"] for t in sample_truth])),
            truth_mean_budget_cost=float(np.mean([t["cost"] for t in sample_truth])))
    return summary


def neighbour_pool(frozen, delta, limit):
    """Legal one-step neighbours of the reference that meet budget and improve."""
    replacements = legal_replacements(
        frozen.reference.action, n=frozen.scene.n, cfg=frozen.cfg,
        candidates=frozen.candidates, business=frozen.business,
        ports_master=frozen.ports)
    seen, scored = set(), []
    for item in replacements.items:
        action = item.action
        key = (tuple(action.links), tuple(np.round(np.asarray(action.powers), 9)))
        if key in seen:
            continue
        seen.add(key)
        label = action_metrics(frozen.bank, action, frozen.reference.eta_ref, delta)
        scored.append((label, action))
    improved = [(l, a) for l, a in scored
                if l["cost"] == 0.0 and l["mean_ee"] > frozen.reference_mean_ee * 1.001]
    improved.sort(key=lambda pair: pair[0]["mean_ee"], reverse=True)
    # Headroom above the reference matters: if no legal neighbour improves, the
    # reference is already the best action in the class and "RL beats cloning"
    # is unanswerable at this state.
    return dict(generated=replacements.generated, legal=len(scored), improved=len(improved),
                kept=len(improved[:limit]),
                reference_mean_ee=frozen.reference_mean_ee,
                kept_mean_ee=[float(l["mean_ee"]) for l, _ in improved[:limit]],
                kept_min_pair_ee=[float(l["min_ee"]) for l, _ in improved[:limit]],
                best_improvement_ratio=(float(improved[0][0]["mean_ee"]
                                              / frozen.reference_mean_ee)
                                        if improved else None)), improved[:limit]


def bc_pretrain(learner, features, reference, steps, lr, log_std_target, log_every):
    optimizer = torch.optim.Adam(learner.actor.parameters(), lr=lr)
    curve, started = [], time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        report = learner.imitation_loss(features, reference, log_std_target=log_std_target)
        report["loss"].backward()
        torch.nn.utils.clip_grad_norm_(learner.actor.parameters(), 10.0)
        optimizer.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append(dict(step=step, loss=float(report["loss"].detach()),
                              discrete_loss=float(report["discrete_loss"].detach()),
                              power_loss=float(report["power_loss"].detach()),
                              std_loss=float(report["std_loss"].detach())))
    final = learner.imitation_loss(features, reference, log_std_target=log_std_target)
    return dict(steps=steps, lr=lr, log_std_target=log_std_target,
                seconds=time.perf_counter() - started, curve=curve,
                final={k: float(v) for k, v in final.items() if k != "steps"},
                final_steps=int(final["steps"]))


def gradient_split(learner, features, samples):
    """Gradient norms of the actual discrete/continuous objective pieces.

    The discrete term of ``update`` is ``coefficient * log pi``; this splits
    that coefficient into its negative-Q part and its entropy/baseline part so
    "entropy dominates" can be measured instead of asserted.
    """
    params = [p for p in learner.actor.parameters() if p.requires_grad]
    rows = []

    def norm(term):
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((g.pow(2).sum() if g is not None else 0.) for g in grads)))

    alpha_d, alpha_c = learner.config.alpha_discrete, learner.config.alpha_continuous
    for _ in range(samples):
        sample = learner._sample(features)
        q = torch.minimum(*(critic(features, sample.action, sample.powers)
                            for critic in learner.critics))
        continuous = alpha_c * sample.log_prob_continuous - q
        q_discrete = (-q.detach()) * sample.log_prob_discrete
        entropy_discrete = (alpha_c * sample.log_prob_continuous.detach()
                            + alpha_d * (sample.log_prob_discrete.detach() + 1)
                            - learner.discrete_baseline) * sample.log_prob_discrete
        rows.append(dict(q=float(q.detach()),
                         log_prob_discrete=float(sample.log_prob_discrete.detach()),
                         log_prob_continuous=float(sample.log_prob_continuous.detach()),
                         coefficient_continuous_entropy=float(
                             alpha_c * sample.log_prob_continuous.detach()),
                         coefficient_discrete_entropy=float(
                             alpha_d * (sample.log_prob_discrete.detach() + 1)),
                         coefficient_value=float(-q.detach()),
                         coefficient_baseline=float(learner.discrete_baseline),
                         q_gradient=norm(q_discrete),
                         entropy_gradient=norm(entropy_discrete),
                         continuous_gradient=norm(continuous),
                         continuous_q_gradient=norm(-q),
                         continuous_entropy_gradient=norm(alpha_c * sample.log_prob_continuous)))
    return dict(samples=samples,
                mean_q=float(np.mean([r["q"] for r in rows])),
                q_spread=float(max(r["q"] for r in rows) - min(r["q"] for r in rows)),
                mean_log_prob_discrete=float(np.mean([r["log_prob_discrete"] for r in rows])),
                mean_log_prob_continuous=float(np.mean([r["log_prob_continuous"] for r in rows])),
                # The score-function coefficient mixes three very different
                # magnitudes; reporting them separately stops a large power
                # Jacobian term from being read as "discrete entropy".
                mean_coefficient=dict(
                    continuous_entropy=float(np.mean([r["coefficient_continuous_entropy"]
                                                      for r in rows])),
                    discrete_entropy=float(np.mean([r["coefficient_discrete_entropy"]
                                                    for r in rows])),
                    value=float(np.mean([r["coefficient_value"] for r in rows])),
                    baseline=float(np.mean([r["coefficient_baseline"] for r in rows]))),
                mean_q_gradient=float(np.mean([r["q_gradient"] for r in rows])),
                mean_entropy_gradient=float(np.mean([r["entropy_gradient"] for r in rows])),
                mean_continuous_gradient=float(np.mean([r["continuous_gradient"] for r in rows])),
                mean_continuous_q_gradient=float(np.mean([r["continuous_q_gradient"] for r in rows])),
                mean_continuous_entropy_gradient=float(np.mean([r["continuous_entropy_gradient"]
                                                                for r in rows])),
                alpha_discrete=alpha_d, alpha_continuous=alpha_c)


def power_response(learner, frozen, delta, factors):
    """Critic vs true target along a pure power sweep at the reference action.

    The actor's power drift is only diagnosable if the critic can order powers
    near the 1 mW floor; this prints whether its preference is monotone.
    """
    rows = []
    for factor in factors:
        powers = np.clip(np.asarray(frozen.reference.action.powers, dtype=float) * factor,
                         1e-3, 2.0)
        action = Action(list(frozen.reference.action.links), powers,
                        {k: np.array(v, copy=True)
                         for k, v in frozen.reference.action.ports.items()})
        label = action_metrics(frozen.bank, action, frozen.reference.eta_ref, delta)
        rows.append(dict(factor=factor, mean_power_w=float(np.mean(action.powers)),
                         mean_ee=label["mean_ee"], min_pair_ee=label["min_ee"],
                         network_ee=label["network_ee"],
                         cost=label["cost"], q=ensemble_q(learner, frozen.features, action),
                         target=label["mean_ee"] / learner.config.reward_scale - label["cost"]))
    return rows


def sample_set(learner, frozen, delta, count):
    actions, labels = [], []
    for _ in range(count):
        sample = learner.select_action(frozen.features)
        actions.append(sample.action)
        labels.append(action_metrics(frozen.bank, sample.action,
                                     frozen.reference.eta_ref, delta))
    return actions, labels


SELECTORS = ("critic", "training_label", "feasible_forecast",
             "forecast_ignoring_budget", "first_sample")


def select_candidates(qs, ees, costs, reward_scale):
    """Rank one candidate set under five like-for-like references.

    ``ees`` is each candidate's predicted mean per-link EE.  ``training_label``
    maximises exactly what the critic was fitted on, ``ee/reward_scale - cost``.
    ``feasible_forecast`` first keeps only candidates whose predicted budget
    cost is zero and then maximises predicted mean per-link EE -- that is the
    question the study actually asks.  ``forecast_ignoring_budget`` drops the
    feasibility filter and is kept only as an explicit contrast; its regret may
    be negative because an infeasible pick can score higher on raw EE.
    """
    target = [float(m) / reward_scale - float(c) for m, c in zip(ees, costs)]
    feasible = [i for i, c in enumerate(costs) if float(c) <= 0.0]
    best_feasible = max((float(ees[i]) for i in feasible), default=None)
    picks = dict(
        critic=int(np.argmax(qs)),
        training_label=int(np.argmax(target)),
        forecast_ignoring_budget=int(np.argmax([float(m) for m in ees])),
        feasible_forecast=(max(feasible, key=lambda i: float(ees[i]))
                           if feasible else None),
        first_sample=0)
    picks["feasible"] = feasible
    picks["best_feasible_ee"] = best_feasible
    picks["regret_predicted"] = {
        name: (None if picks[name] is None or best_feasible is None
               else best_feasible - float(ees[picks[name]]))
        for name in SELECTORS}
    return picks


def pair_concordance(a, b):
    """Share of pairs ordered the same way; ties in ``b`` are excluded."""
    agree = total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if abs(float(b[i]) - float(b[j])) < 1e-12:
                continue
            total += 1
            agree += (float(a[i]) - float(a[j])) * (float(b[i]) - float(b[j])) > 0
    return agree / total if total else None


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(count=0)
    return dict(count=len(values), mean=float(np.mean(values)),
                median=float(np.median(values)), min=float(np.min(values)),
                max=float(np.max(values)))


def selection_policy(learner, frozen, delta, k, repeats, window=None):
    """Draw k proposals and compare five selection rules on the same set.

    This is an inference-time rule (sample k, rank, shield), not a training
    change.  It separates "the sampler never produces a good action" from "the
    actor cannot update toward it": if a selector finds the outlier out of k
    samples, the value is already in the proposal distribution and the
    score-function update is what fails to extract it.

    Every rule sees the same k candidates, the same forecast bank and the same
    shield, so the only difference is the ranking.  ``window`` (truth) is
    optional and validation-only; it never feeds a decision.
    """
    scale = learner.config.reward_scale
    eta = frozen.reference.eta_ref
    rows = {name: [] for name in SELECTORS}
    agreements = dict(critic_equals_training_label=0,
                      critic_equals_feasible_forecast=0,
                      training_label_equals_feasible_forecast=0)
    fallbacks = {name: 0 for name in SELECTORS}
    no_feasible = 0
    per_repeat = []
    before = frozen.bank.evaluator_calls
    started = time.perf_counter()
    for _ in range(repeats):
        actions, labels = sample_set(learner, frozen, delta, k)
        qs = [ensemble_q(learner, frozen.features, action) for action in actions]
        picks = select_candidates(qs, [l["mean_ee"] for l in labels],
                                  [l["cost"] for l in labels], scale)
        no_feasible += int(not picks["feasible"])
        agreements["critic_equals_training_label"] += int(
            picks["critic"] == picks["training_label"])
        agreements["critic_equals_feasible_forecast"] += int(
            picks["feasible_forecast"] is not None
            and picks["critic"] == picks["feasible_forecast"])
        agreements["training_label_equals_feasible_forecast"] += int(
            picks["feasible_forecast"] is not None
            and picks["training_label"] == picks["feasible_forecast"])
        truths, best_truth = None, None
        if window is not None:
            truths = [truth_metrics(frozen, window, a, eta, delta) for a in actions]
            feasible_truth = [i for i, t in enumerate(truths) if t["cost"] <= 0.0]
            if feasible_truth:
                best_truth = max(truths[i]["mean_ee"] for i in feasible_truth)
        target = [float(l["mean_ee"]) / scale - float(l["cost"]) for l in labels]
        ees = [float(l["mean_ee"]) for l in labels]
        feasible = picks["feasible"]
        repeat_row = dict(
            candidates=k, no_feasible=not feasible,
            best_feasible_ee=picks["best_feasible_ee"],
            best_truth_ee=best_truth,
            picks={name: picks[name] for name in SELECTORS},
            regret_predicted=picks["regret_predicted"],
            concordance_q_vs_target=pair_concordance(qs, target),
            concordance_q_vs_ee=pair_concordance(qs, ees),
            concordance_q_vs_target_feasible=(
                pair_concordance([qs[i] for i in feasible], [target[i] for i in feasible])
                if len(feasible) > 1 else None))
        for name in SELECTORS:
            index = picks[name]
            if index is None:
                repeat_row[name] = None
                continue
            protected = protect_budget(actions[index], frozen.reference, delta=delta,
                                       n=frozen.scene.n, cfg=frozen.cfg,
                                       candidates=frozen.candidates,
                                       business=frozen.business,
                                       evaluator=frozen.bank.evaluate)
            executed = action_metrics(frozen.bank, protected.action, eta, delta)
            fallbacks[name] += int(protected.fallback)
            entry = dict(predicted=labels[index], executed=executed, q=qs[index],
                         regret_predicted=picks["regret_predicted"][name],
                         fallback=bool(protected.fallback))
            if truths is not None:
                entry["truth"] = truths[index]
                entry["truth_executed"] = truth_metrics(frozen, window,
                                                        protected.action, eta, delta)
                entry["regret_truth"] = (None if best_truth is None else
                                         best_truth - truths[index]["mean_ee"])
                entry["regret_truth_executed"] = (None if best_truth is None else
                                                  best_truth
                                                  - entry["truth_executed"]["mean_ee"])
            rows[name].append(entry)
            repeat_row[name] = dict(fallback=bool(protected.fallback),
                                    regret_predicted=entry["regret_predicted"],
                                    regret_truth=entry.get("regret_truth"))
        per_repeat.append(repeat_row)
    seconds = time.perf_counter() - started
    evaluations = frozen.bank.evaluator_calls - before

    summary, power = {}, {}
    for name in SELECTORS:
        items = rows[name]
        summary[name] = dict(
            count=len(items),
            predicted_mean_ee=_stats([i["predicted"]["mean_ee"] for i in items]),
            predicted_min_pair_ee=_stats([i["predicted"]["min_ee"] for i in items]),
            predicted_cost=_stats([i["predicted"]["cost"] for i in items]),
            executed_mean_ee=_stats([i["executed"]["mean_ee"] for i in items]),
            executed_min_pair_ee=_stats([i["executed"]["min_ee"] for i in items]),
            executed_cost=_stats([i["executed"]["cost"] for i in items]),
            truth_mean_ee=_stats([i["truth"]["mean_ee"] for i in items if "truth" in i]),
            truth_cost=_stats([i["truth"]["cost"] for i in items if "truth" in i]),
            truth_executed_mean_ee=_stats([i["truth_executed"]["mean_ee"]
                                           for i in items if "truth_executed" in i]),
            regret_predicted=_stats([i["regret_predicted"] for i in items]),
            regret_truth=_stats([i["regret_truth"] for i in items if "regret_truth" in i]),
            fallback_rate=float(fallbacks[name] / len(items)) if items else None)
        powers = [np.asarray(i["predicted"]["powers"], dtype=float) for i in items]
        power[name] = power_summary(np.concatenate(powers)) if powers else dict(count=0)
    return dict(
        k=k, repeats=repeats,
        note="all rules rank the same k candidates with the same forecast bank and "
             "the same budget shield; 'feasible_forecast' is the rule that matches the "
             "research question, 'training_label' is what the critic was fitted on, "
             "'forecast_ignoring_budget' keeps the old (inconsistent) oracle column "
             "only as a contrast",
        agreements={key: value / repeats for key, value in agreements.items()},
        concordance=dict(
            q_vs_target=_stats([r["concordance_q_vs_target"] for r in per_repeat]),
            q_vs_ee=_stats([r["concordance_q_vs_ee"] for r in per_repeat]),
            q_vs_target_feasible=_stats([r["concordance_q_vs_target_feasible"]
                                         for r in per_repeat])),
        no_feasible_candidate_rate=no_feasible / repeats,
        by_selector=summary, power=power, per_repeat=per_repeat,
        accounting=dict(bank_evaluations=int(evaluations),
                        bank_evaluations_per_repeat=evaluations / repeats,
                        seconds=seconds, seconds_per_repeat=seconds / repeats,
                        includes_shield=True))


def action_scores(learner, features, action):
    """Teacher-forced log pi of one legal action, plus per-step detail.

    Replays the action in the sampler's own anchor order against the same
    completion mask, so ``mean_log_prob_discrete`` is comparable across
    actions (every action has exactly n/2 steps).
    """
    node, edge, mask = features.tensors(learner.device)
    scores, means, log_stds = learner.actor(node, edge, mask)
    adjacency = features.mode_mask.any(-1)
    targets = {}
    for (tx, rx, mode), power in zip(action.links, action.powers):
        key = (int(tx), int(rx))
        if key in targets:
            raise ValueError('action contains a duplicate link')
        targets[key] = (int(tx), int(rx), 0 if mode == 'RF' else 1, float(power))
    if len(targets) != features.n // 2:
        raise ValueError('action must be one complete matching')
    remaining, steps, log_d, log_c = list(range(features.n)), [], 0., 0.
    while remaining:
        anchor = remaining[0]
        keys = [key for key in targets if anchor in key]
        if len(keys) != 1:
            raise ValueError('action does not cover every vehicle exactly once')
        tx, rx, mode, power = targets.pop(keys[0])
        choices = learner._legal_choices(adjacency, features.mode_mask, remaining, anchor)
        if (tx, rx, mode) not in choices:
            raise ValueError('action link is not legal under the block-start mask')
        index = choices.index((tx, rx, mode))
        indices = tuple(torch.tensor([c[d] for c in choices], device=learner.device)
                        for d in range(3))
        log_prob = float(torch.log_softmax(scores[indices], 0)[index])
        mean, log_std = means[tx, rx, mode], log_stds[tx, rx, mode]
        latent = float(latent_from_power(power, learner.config.min_power_w,
                                         learner.config.max_power_w))
        normal = -0.5 * (((latent - mean) / log_std.exp()).square()
                          + 2 * log_std + math.log(2 * math.pi))
        _, log_jacobian = power_from_latent(torch.tensor(latent), learner.config.min_power_w,
                                            learner.config.max_power_w)
        log_c += float((normal - log_jacobian).detach())
        log_d += log_prob
        steps.append(dict(anchor=int(anchor), link=[int(tx), int(rx)],
                          mode='RF' if mode == 0 else 'VLC', log_probability=log_prob))
        remaining = [v for v in remaining if v not in (tx, rx)]
    return dict(log_prob_discrete=log_d, log_prob_continuous=log_c,
                mean_log_prob_discrete=log_d / len(steps), steps=steps)


def switch_report(learner, frozen, reference, candidates):
    """How far the policy is from switching to each candidate action.

    For the *first* sequential decision where the two paths diverge, report the
    policy's probability of taking the candidate's link instead of the
    reference's.  That is the sample budget the actor would need, not a score.
    Later steps are not comparable: the two paths leave different vehicles
    unmatched, so their anchors differ from that point on.
    """
    ref = action_scores(learner, frozen.features, reference)
    rows = []
    for name, action in candidates:
        alt = action_scores(learner, frozen.features, action)
        divergence = None
        for a, b in zip(ref['steps'], alt['steps']):
            if a['link'] != b['link'] or a['mode'] != b['mode']:
                divergence = dict(anchor=a['anchor'], reference_link=a['link'],
                                  candidate_link=b['link'],
                                  p_reference_link=math.exp(a['log_probability']),
                                  p_candidate_link=math.exp(b['log_probability']),
                                  odds_reference_over_candidate=math.exp(
                                      a['log_probability'] - b['log_probability']))
                break
        rows.append(dict(
            name=name,
            delta_log_prob_discrete=alt['log_prob_discrete'] - ref['log_prob_discrete'],
            mean_log_prob_discrete=alt['mean_log_prob_discrete'],
            first_divergence=divergence))
    return dict(reference=dict(mean_log_prob_discrete=ref['mean_log_prob_discrete'],
                               log_prob_continuous=ref['log_prob_continuous']),
                candidates=rows)


def main(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("probe output must be fresh")
    torch.set_num_threads(args.threads)
    frozen = build_frozen_block(args.n, args.block, args.draws,
                                args.channel_seed, args.forecast_seed)
    reference_label = action_metrics(frozen.bank, frozen.reference.action,
                                     frozen.reference.eta_ref, args.delta)
    frozen.reference_mean_ee = reference_label["mean_ee"]
    frozen.reference_min_ee = reference_label["min_ee"]  # diagnostic only
    # Read the block's true channel once.  It is a separate oracle built from
    # the same seed and is only ever consulted after decisions are fixed, so it
    # cannot leak into forecast, replay, selection or training.
    window = truth_window(frozen) if args.truth else None
    reference_truth = (truth_metrics(frozen, window, frozen.reference.action,
                                     frozen.reference.eta_ref, args.delta)
                       if window is not None else None)
    neighbours, neighbour_actions = neighbour_pool(frozen, args.delta, args.max_neighbors)

    model = ModelConfig(node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                        hidden_dim=args.hidden_dim, encoder=args.encoder)
    sac = SACConfig(seed=args.learning_seed, batch_size=args.batch_size,
                    replay_capacity=args.replay_capacity)
    # Both branches keep the same reward, replay, batch and seed.  The only
    # change is whether the continuous (power) entropy term is in the actor
    # objective.  Continuous-entropy tuning is a learning-method change, not a
    # redefinition of the communication service.
    if args.branch in ("no_continuous_entropy", "fixed_power"):
        sac.alpha_continuous = 0.0
    explorer = HybridMASAC(model, SACConfig(seed=args.learning_seed - 1,
                                           batch_size=args.batch_size,
                                           replay_capacity=args.replay_capacity))
    coverage_actions, coverage_labels = sample_set(explorer, frozen, args.delta, args.coverage)

    bc_only = HybridMASAC(model, sac)
    pretrain = bc_pretrain(bc_only, frozen.features, frozen.reference.action,
                          args.bc_steps, args.bc_lr, math.log(args.bc_log_std), args.log_every)
    bc_arm = evaluate_policy(bc_only, frozen, args.delta, args.samples,
                             window=window)
    pretrain_split = gradient_split(bc_only, frozen.features, args.grad_samples)

    masac = HybridMASAC(model, sac)
    masac.actor.load_state_dict(bc_only.actor.state_dict())

    def pin():
        # Applied to ``masac`` only, and only after the identical BC stage, so
        # the cloned arm stays comparable across branches.
        if args.branch == "fixed_power":
            pin_floor_power(masac)

    def step(optimize_actor):
        masac.update(actor=optimize_actor)
        pin()

    pin()
    candidates = [(f'neighbour#{index}', action)
                  for index, (_, action) in enumerate(neighbour_actions)]
    switch_at_clone = switch_report(masac, frozen, frozen.reference.action, candidates)

    def add(action, label, source):
        masac.observe(frozen.features, action, label["mean_ee"], cost=label["cost"],
                      label_source=source)

    add(frozen.reference.action, reference_label, "reference_forecast")
    for label, action in neighbour_actions:
        add(action, label, "neighbour_forecast")
    for action, label in zip(coverage_actions, coverage_labels):
        add(action, label, "proposal_forecast")
    cloned_actions, cloned_labels = sample_set(bc_only, frozen, args.delta, args.coverage_clone)
    for action, label in zip(cloned_actions, cloned_labels):
        add(action, label, "proposal_forecast")
    initial_replay = len(masac.replay)

    heldout_actions, heldout_labels = sample_set(explorer, frozen, args.delta, args.heldout // 2)
    good_actions, good_labels = sample_set(bc_only, frozen, args.delta,
                                          args.heldout - args.heldout // 2)
    heldout_actions += good_actions
    heldout_labels += good_labels
    warmup_history, warmup_steps, trusted = [], 0, False
    while warmup_steps < args.warmup_max:
        for _ in range(args.warmup):
            step(False)
            warmup_steps += 1
        check = critic_report(masac, frozen.features, heldout_actions, heldout_labels)
        warmup_history.append(dict(steps=warmup_steps, **check))
        trusted = (check["mae"] <= args.gate_mae
                   and (check["concordance"] or 0.) >= args.gate_concordance)
        if trusted or warmup_steps >= args.warmup_max:
            break

    checks, observed, actor_steps = [], 0, 0
    observed_actions, observed_labels, round_trace = [], [], []
    started = time.perf_counter()
    for round_index in range(args.rounds if trusted else 0):
        round_best = None
        for _ in range(args.proposals_per_round):
            sample = masac.select_action(frozen.features)
            label = action_metrics(frozen.bank, sample.action, frozen.reference.eta_ref, args.delta)
            masac.observe(frozen.features, sample.action, label["mean_ee"], cost=label["cost"],
                          proposal=sample, label_source="proposal_forecast")
            observed += 1
            observed_actions.append(sample.action)
            observed_labels.append(label)
            if round_best is None or label["mean_ee"] > round_best["mean_ee"]:
                round_best = label
        round_trace.append(dict(round=round_index, best_mean_ee=round_best["mean_ee"],
                                mean_mean_ee=float(np.mean(
                                    [l["mean_ee"] for l in observed_labels[-args.proposals_per_round:]]))))
        # Keep the reference and improved neighbours in replay so a flood of
        # equally bad proposals cannot dominate the critic.
        add(frozen.reference.action, reference_label, "reference_forecast")
        for label, action in neighbour_actions:
            add(action, label, "neighbour_forecast")
        for _ in range(args.critic_steps):
            step(False)
        step(True)
        actor_steps += 1
        if round_index % args.check_every == 0 or round_index == args.rounds - 1:
            fresh_bad = sample_set(explorer, frozen, args.delta, args.heldout // 2)
            fresh_good = sample_set(masac, frozen, args.delta,
                                    args.heldout - args.heldout // 2)
            fresh = (fresh_bad[0] + fresh_good[0], fresh_bad[1] + fresh_good[1])
            checks.append(dict(round=round_index, updates=masac.updates,
                               actor_steps=actor_steps, replay=len(masac.replay),
                               gradient_split=gradient_split(masac, frozen.features,
                                                             args.grad_samples),
                               **evaluate_policy(masac, frozen, args.delta,
                                                 args.samples, heldout_actions=fresh)))
    elapsed = time.perf_counter() - started

    masac_arm = evaluate_policy(masac, frozen, args.delta, args.samples,
                                window=window)
    selection = dict(
        bc_only=selection_policy(bc_only, frozen, args.delta,
                                 args.selection_k, args.selection_repeats,
                                 window=window),
        bc_masac=selection_policy(masac, frozen, args.delta,
                                  args.selection_k, args.selection_repeats,
                                  window=window))

    def links_of(action):
        return {(int(tx), int(rx), mode) for tx, rx, mode in action.links}

    def histogram(values, upper):
        counts = np.zeros(upper + 1, dtype=int)
        for value in values:
            counts[min(int(value), upper)] += 1
        return {str(i): int(c) for i, c in enumerate(counts) if c}

    proposal_links = [links_of(a) for a in observed_actions]
    reference_links = links_of(frozen.reference.action)
    distances = [len(reference_links ^ links) for links in proposal_links]
    to_neighbours = [[len(links ^ links_of(action)) for links in proposal_links]
                     for _, action in candidates]
    # A proposal only counts as "closer" when it is nearer to the neighbour than
    # the reference is; a single swap already puts the reference two links away,
    # so a plain "<= 2" threshold would count every reference-like proposal.
    reference_distance = [len(reference_links ^ links_of(action)) for _, action in candidates]
    best_index = (int(np.argmax([l["mean_ee"] for l in observed_labels]))
                  if observed_labels else None)
    exploration = dict(
        note="teacher-forced log pi of each improved neighbour; the first divergent "
             "decision gives the per-sample probability the sampler needs to produce it",
        switch_at_clone=switch_at_clone,
        switch_at_end=switch_report(masac, frozen, frozen.reference.action, candidates),
        proposals=dict(
            count=len(observed_actions),
            distance_from_reference=histogram(distances, frozen.scene.n // 2),
            distance_to_each_neighbour=[histogram(d, frozen.scene.n // 2)
                                        for d in to_neighbours],
            min_distance_to_each_neighbour=[int(min(d)) if d else None for d in to_neighbours],
            reference_distance_to_each_neighbour=reference_distance,
            proposals_closer_than_reference_to_each_neighbour=[
                int(np.sum(np.asarray(d) < limit)) for d, limit in
                zip(to_neighbours, reference_distance)],
            best_mean_ee=(float(observed_labels[best_index]["mean_ee"])
                          if best_index is not None else None),
            best_mean_ee_round=int(best_index // args.proposals_per_round)
            if best_index is not None else None,
            distance_of_best_to_each_neighbour=[int(d[best_index]) if best_index is not None else None
                                                for d in to_neighbours],
            critic_on_training_proposals=(
                critic_report(masac, frozen.features, observed_actions, observed_labels)
                if observed_actions else None)),
        per_round=round_trace)
    result = dict(
        scope="single-state behaviour-cloning + gated continuous-feedback probe; not "
              "convergence, not an algorithm comparison",
        n=args.n, block=args.block, delta=args.delta, draws=args.draws,
        encoder=args.encoder, hidden_dim=args.hidden_dim,
        branch=args.branch,
        seeds=dict(learning=args.learning_seed, channel=args.channel_seed,
                   forecast=args.forecast_seed),
        reference=dict(metrics={k: v for k, v in reference_label.items() if k != "powers"},
                       eta_ref=frozen.reference.eta_ref,
                       truth=({k: v for k, v in reference_truth.items() if k != "powers"}
                              if reference_truth is not None else None)),
        neighbours=neighbours,
        exploration=exploration,
        selection=selection,
        truth_scope=("executed truth on this block's 1 ms channel; read after all "
                     "decisions, never used for training or selection"
                     if window is not None else "not requested (--truth)"),
        bc_pretrain=pretrain,
        bc_gradient_split=pretrain_split,
        replay_composition=dict(initial=initial_replay, coverage=args.coverage,
                                cloned=args.coverage_clone),
        critic_warmup=dict(steps=warmup_steps, trusted=trusted,
                           gate=dict(mae=args.gate_mae, concordance=args.gate_concordance),
                           history=warmup_history),
        power_response=dict(factors=[.5, .8, 1., 1.05, 1.2, 2., 5., 10., 100.],
                            masac=power_response(masac, frozen, args.delta,
                                                 [.5, .8, 1., 1.05, 1.2, 2., 5., 10., 100.]),
                            bc_only=power_response(bc_only, frozen, args.delta,
                                                   [.5, .8, 1., 1.05, 1.2, 2., 5., 10., 100.])),
        arms=dict(
            bc_only=dict(evaluation=bc_arm, replay=0, updates=0),
            bc_masac=dict(evaluation=masac_arm, replay=len(masac.replay),
                          critic_steps=masac.updates - actor_steps, actor_steps=actor_steps,
                          proposals_observed=observed, rounds=args.rounds if trusted else 0,
                          proposals_per_round=args.proposals_per_round,
                          critic_steps_per_round=args.critic_steps,
                          batch_size=args.batch_size, elapsed_s=elapsed,
                          checkpoints=checks)),
        model=asdict(model), sac=asdict(sac))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        bc_only.save(args.out / "actor_bc.pt")
        masac.save(args.out / "actor_masac.pt")
    (args.out / "probe.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(50, 100), default=50)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--bc-steps", type=int, default=600)
    parser.add_argument("--bc-lr", type=float, default=3e-3)
    parser.add_argument("--bc-log-std", type=float, default=.25)
    parser.add_argument("--coverage", type=int, default=200)
    parser.add_argument("--coverage-clone", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--warmup-max", type=int, default=1600)
    parser.add_argument("--gate-mae", type=float, default=.2)
    parser.add_argument("--gate-concordance", type=float, default=.75)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--proposals-per-round", type=int, default=8)
    parser.add_argument("--critic-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--replay-capacity", type=int, default=8000)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--check-every", type=int, default=15)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--heldout", type=int, default=24)
    parser.add_argument("--grad-samples", type=int, default=8)
    parser.add_argument("--selection-k", type=int, default=16)
    parser.add_argument("--selection-repeats", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--encoder", choices=("mlp", "hetero_graph", "hetero_graph_v2"),
                        default="mlp")
    parser.add_argument("--branch", choices=("original", "no_continuous_entropy",
                                             "fixed_power"), default="original",
                        help="original: unchanged objective; no_continuous_entropy: "
                             "alpha_c=0; fixed_power: alpha_c=0 and the actor's power "
                             "channel is pinned to the 1 mW floor (diagnostic only)")
    parser.add_argument("--truth", action="store_true",
                        help="also score the same actions on this block's true 1 ms "
                             "channel; validation only, never used for decisions")
    parser.add_argument("--delta", type=float, default=.05)
    parser.add_argument("--channel-seed", type=int, default=2026102401)
    parser.add_argument("--forecast-seed", type=int, default=2026102500)
    parser.add_argument("--learning-seed", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="write actor_bc.pt / actor_masac.pt beside probe.json")
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
