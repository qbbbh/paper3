"""Offline diagnosis of a saved BC / BC+MASAC probe checkpoint.

Answers two questions that the probe summary cannot:

1. Does the actor objective pull the power latent up or down at the 1 mW
   floor, and which term dominates -- the action-value term ``-q`` or the
   continuous entropy term ``alpha_c * log_prob_continuous``?  With the
   log-uniform map, ``log |dp/dz| = log p + log ln(ratio) + log s(1-s)``.
   Near the floor ``s = sigmoid(z) ~ 1.3e-4``, so the ``log s(1-s)`` term
   dominates and its z-slope is ``1 - 2s ~= 1`` (the ``log p`` term adds only
   ``ln(ratio)*s(1-s) ~= 1e-3``).  Since ``log pi_c`` carries ``-log|dp/dz|``
   and ``alpha_c = 1e-3``, the entropy slope is about ``-1e-3`` per link,
   while the Q slope is ``dQ/dp * dp/dz`` and ``dp/dz ~= 1e-6`` W per unit z,
   so it is four orders smaller.
2. Are the two arms' deterministic actions different in matching or only in
   power?  The matching is compared link by link against the reference.

Diagnostic only: one frozen block, no training, no truth sampling.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from p3learn.allocation import Action
from p3learn.masac import HybridMASAC, power_from_latent
from probe_common import action_metrics, build_frozen_block, jsonable


def deterministic_trace(learner, features):
    """Chosen links plus their power latent statistics, in sampler order."""
    node, edge, mask = features.tensors(learner.device)
    scores, means, log_stds = learner.actor(node, edge, mask)
    adjacency = features.mode_mask.any(-1)
    remaining, rows = list(range(features.n)), []
    while remaining:
        anchor = remaining[0]
        choices = learner._legal_choices(adjacency, features.mode_mask, remaining, anchor)
        indices = tuple(torch.tensor([c[d] for c in choices], device=learner.device)
                        for d in range(3))
        chosen = int(scores[indices].argmax())
        tx, rx, mode = choices[chosen]
        rows.append((tx, rx, mode, means[tx, rx, mode], log_stds[tx, rx, mode]))
        remaining = [v for v in remaining if v not in (tx, rx)]
    return rows


def objective_slopes(learner, features):
    """dQ/dz, d(alpha_c log pi_c)/dz and their sum, per power latent.

    The actor minimises ``L = alpha_c * log pi_c - Q``, so ``dL/dz`` is
    ``d_entropy_dz - dq_dz``: a negative value there means the update raises
    the power latent.
    """
    rows = deterministic_trace(learner, features)
    links = [((int(tx), int(rx), 'RF' if mode == 0 else 'VLC'))
             for tx, rx, mode, _, _ in rows]
    ports = dict(tx=np.zeros((features.n, features.n), dtype=int),
                 rx=np.zeros((features.n, features.n), dtype=int))
    for (tx, rx, mode) in links:
        if mode == 'VLC':
            ports['tx'][tx, rx], ports['rx'][tx, rx] = features.port_choices[tx, rx, 1]
    latents = torch.stack([mean for _, _, _, mean, _ in rows])
    log_stds = torch.stack([std for _, _, _, _, std in rows])

    def slopes_for(latent):
        powers, log_jacobian = power_from_latent(latent, learner.config.min_power_w,
                                                 learner.config.max_power_w)
        action = Action(list(links), powers.detach().numpy().astype(float).copy(),
                        {k: np.array(v, copy=True) for k, v in ports.items()})
        q = torch.minimum(*(critic(features, action, powers) for critic in learner.critics))
        normal = -0.5 * (2 * log_stds + np.log(2 * np.pi))
        logp_c = (normal - log_jacobian).sum()
        entropy = learner.config.alpha_continuous * logp_c
        g_q = torch.autograd.grad(q, latent, retain_graph=True)[0]
        g_e = torch.autograd.grad(entropy, latent)[0]
        return dict(q=float(q.detach()), power=float(powers.detach().mean()),
                    log_prob_continuous=float(logp_c.detach()),
                    mean_dq_dz=float(g_q.mean()), mean_d_entropy_dz=float(g_e.mean()),
                    mean_dl_dz=float((g_e - g_q).mean()),
                    per_link=[dict(link=tuple(int(v) for v in links[i][:2]) + (links[i][2],),
                                  power_w=float(powers[i]), dq_dz=float(g_q[i]),
                                  d_entropy_dz=float(g_e[i]),
                                  dl_dz=float(g_e[i] - g_q[i]))
                              for i in range(len(links))])

    at_policy = slopes_for(latents)
    at_floor = slopes_for(torch.tensor([-9.0] * len(rows), requires_grad=True))
    return dict(at_policy=at_policy, at_floor_latent_minus_9=at_floor,
                alpha_continuous=learner.config.alpha_continuous,
                note="L = alpha_c*log pi_c - Q; dL/dz = d_entropy_dz - dq_dz. A negative "
                     "mean_dl_dz means the actor update raises the power latent.")


def coefficient_decomposition(learner, features, samples):
    """Which part of the score-function coefficient drives the discrete step.

    ``update`` multiplies ``log pi_discrete`` by
    ``detach(alpha_c*log pi_c - Q + alpha_d*(log pi_d + 1) - baseline)``.
    At the 1 mW floor ``alpha_c*log pi_c`` is large and positive because
    ``log pi_c`` carries ``-log|dp/dz|``; if that piece dominates the
    coefficient, the discrete update is being driven by the power
    parameterisation rather than by the advantage.
    """
    params = [p for p in learner.actor.parameters() if p.requires_grad]

    def norm(term):
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((g.pow(2).sum() if g is not None else 0.) for g in grads)))

    alpha_d, alpha_c = learner.config.alpha_discrete, learner.config.alpha_continuous
    rows = []
    for _ in range(samples):
        sample = learner._sample(features)
        q = torch.minimum(*(critic(features, sample.action, sample.powers)
                            for critic in learner.critics))
        parts = dict(continuous_entropy=alpha_c * sample.log_prob_continuous.detach(),
                     value=-q.detach(),
                     discrete_entropy=alpha_d * (sample.log_prob_discrete.detach() + 1),
                     baseline=torch.tensor(float(learner.discrete_baseline),
                                           device=sample.log_prob_discrete.device))
        parts['total'] = (parts['continuous_entropy'] + parts['value']
                          + parts['discrete_entropy'] - parts['baseline'])
        rows.append(dict(log_prob_discrete=float(sample.log_prob_discrete.detach()),
                         log_prob_continuous=float(sample.log_prob_continuous.detach()),
                         **{f'{k}_value': float(v) for k, v in parts.items()},
                         **{f'{k}_gradient': norm(v * sample.log_prob_discrete)
                            for k, v in parts.items()}))
    return dict(samples=samples,
                **{f'mean_{key}': float(np.mean([r[key] for r in rows]))
                   for key in rows[0] if key.startswith(('log_prob', 'continuous_entropy',
                                                         'value', 'discrete_entropy',
                                                         'baseline', 'total'))},
                alpha_discrete=alpha_d, alpha_continuous=alpha_c)


def arm_report(learner, frozen, delta, samples=8):
    deterministic = learner.select_action(frozen.features, deterministic=True)
    own = action_metrics(frozen.bank, deterministic.action, frozen.reference.eta_ref, delta)
    floor_action = Action(list(deterministic.action.links),
                          np.asarray(frozen.reference.action.powers, dtype=float).copy(),
                          {k: np.array(v, copy=True)
                           for k, v in deterministic.action.ports.items()})
    floor = action_metrics(frozen.bank, floor_action, frozen.reference.eta_ref, delta)
    reference_links = {(int(a), int(b), m) for a, b, m in frozen.reference.action.links}
    own_links = {(int(a), int(b), m) for a, b, m in deterministic.action.links}
    return dict(
        own={k: v for k, v in own.items() if k != 'powers'},
        floor_power_control={k: v for k, v in floor.items() if k != 'powers'},
        matching_differs=[list(sorted(reference_links - own_links)),
                          list(sorted(own_links - reference_links))],
        changed_links=len(reference_links ^ own_links) // 2,
        objective=objective_slopes(learner, frozen.features),
        coefficient=coefficient_decomposition(learner, frozen.features, samples))


def main(args):
    torch.set_num_threads(args.threads)
    frozen = build_frozen_block(args.n, args.block, args.draws,
                                args.channel_seed, args.forecast_seed)
    reference_label = action_metrics(frozen.bank, frozen.reference.action,
                                     frozen.reference.eta_ref, args.delta)
    out = dict(n=args.n, block=args.block, delta=args.delta,
               reference={k: v for k, v in reference_label.items() if k != 'powers'},
               checkpoints={})
    for name, path in args.checkpoint:
        learner = HybridMASAC.load(path)
        out['checkpoints'][name] = dict(
            path=str(path), updates=learner.updates,
            **arm_report(learner, frozen, args.delta))
    text = json.dumps(jsonable(out), indent=2, ensure_ascii=False, allow_nan=False)
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', nargs=2,
                        metavar=('NAME', 'PATH'), required=True)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--block', type=int, default=2)
    parser.add_argument('--draws', type=int, default=4)
    parser.add_argument('--delta', type=float, default=.05)
    parser.add_argument('--channel-seed', type=int, default=2026102401)
    parser.add_argument('--forecast-seed', type=int, default=2026102500)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', type=Path)
    main(parser.parse_args())
