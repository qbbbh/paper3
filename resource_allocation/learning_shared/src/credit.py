"""Counterfactual credit for complete legal actions -- forecast-only diagnostic.

Two rules drive this module:

1. Only *legal replacements* are compared.  A replacement keeps every vehicle
   served exactly once; deleting a link or dropping a vehicle is never a
   counterfactual, because it changes the block the budget is defined on.
2. VLC ports come from the block-start master port map, never from the
   executed action.  An actor-sampled Action only fills ``ports`` for VLC
   links; every other entry is zero, so reusing it would silently evaluate a
   VLC replacement at port pair (0, 0).

A single replacement changes every link's SINR through same-mode interference,
so there is no per-link decomposition of the action's EE.  Reports carry
action-level deltas plus a per-slot rate delta vector, and are labelled as
conditional-trajectory forecasts, not measured truth.

Nothing here writes results.csv, summary.json or the NPZ: the output audit
matches method sets, denominators and array names exactly.
"""
from dataclasses import dataclass
import numpy as np

from p3base.block import validate_action

from .allocation import Action, copy_action

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Replacement:
    """One legal counterfactual action and the link it was built around."""

    kind: str
    primary: int
    secondary: int
    action: Action


@dataclass(frozen=True)
class ReplacementSet:
    items: tuple
    generated: int
    skipped_illegal: int
    noop: int = 0


def _cap(cfg, mode):
    return float(getattr(cfg, mode.lower()).power_max_w)


def _clip(power, cfg, mode):
    return float(min(max(float(power), .001), _cap(cfg, mode)))


def _make(links, powers, ports_master):
    return Action(list(links), np.asarray(powers, float).copy(),
                  {key: np.array(ports_master[key], copy=True) for key in ("tx", "rx")})


def _checked(action, n, cfg, candidates, business, ports_master):
    """Full block-start domain check plus the VLC master-port identity."""
    try:
        validate_action(action.links, action.powers, action.ports, n, cfg,
                        candidates=candidates, business=business)
    except ValueError:
        return False
    for tx, rx, mode in action.links:
        if mode == 'VLC' and (
                int(action.ports['tx'][tx, rx]) != int(ports_master['tx'][tx, rx])
                or int(action.ports['rx'][tx, rx]) != int(ports_master['rx'][tx, rx])):
            return False
    return True


def _cheap_legal(links, powers, n, cfg, candidates, business, ports_master):
    """Contract check without building an Action or copying port matrices."""
    try:
        validate_action(links, np.asarray(powers, float), ports_master, n, cfg,
                        candidates=candidates, business=business)
    except ValueError:
        return False
    return True


def legal_replacements(action, *, n, cfg, candidates, business, ports_master):
    """Enumerate legal one-step replacements in a stable, deterministic order.

    Four kinds: mode swap (RF<->VLC), direction swap (tx<->rx), partner swap
    (two links re-paired as t_k->r_l, t_l->r_k, transmitter/receiver roles
    preserved) and power scaling (x0.25, x4, clipped to the mode cap).
    Duplicates and no-ops are dropped.  ``generated`` counts every attempt and
    ``skipped_illegal`` every attempt rejected by the action contract or dedup,
    so the legal fraction is visible instead of being silently filtered.
    """
    links = list(action.links)
    powers = np.asarray(action.powers, float)
    items, seen = [], set()
    generated = skipped = noop = 0

    def add(kind, primary, secondary, new_links, new_powers):
        nonlocal generated, skipped
        generated += 1
        key = (tuple(new_links), tuple(np.round(new_powers, 12).tolist()))
        if key in seen:
            skipped += 1
            return
        if not _cheap_legal(new_links, new_powers, n, cfg, candidates, business, ports_master):
            skipped += 1
            return
        candidate = _make(new_links, new_powers, ports_master)
        if not _checked(candidate, n, cfg, candidates, business, ports_master):
            skipped += 1
            return
        seen.add(key)
        items.append(Replacement(kind, primary, secondary, candidate))

    for index, (tx, rx, mode) in enumerate(links):
        other = 'VLC' if mode == 'RF' else 'RF'
        new_links, new_powers = links.copy(), powers.copy()
        new_links[index] = (tx, rx, other)
        new_powers[index] = _clip(powers[index], cfg, other)
        add('mode', index, -1, new_links, new_powers)

    for index, (tx, rx, mode) in enumerate(links):
        new_links = links.copy()
        new_links[index] = (rx, tx, mode)
        add('direction', index, -1, new_links, powers.copy())

    for first in range(len(links)):
        tx_a, rx_a, mode_a = links[first]
        for second in range(first + 1, len(links)):
            tx_b, rx_b, mode_b = links[second]
            new_links = links.copy()
            new_links[first] = (tx_a, rx_b, mode_a)
            new_links[second] = (tx_b, rx_a, mode_b)
            add('partner', first, second, new_links, powers.copy())

    for index, (_, _, mode) in enumerate(links):
        for factor in (.25, 4.):
            scaled = _clip(powers[index] * factor, cfg, mode)
            if abs(scaled - powers[index]) <= 1e-12:
                # Clipped to the same power: a no-op, not an illegal action.
                generated += 1
                noop += 1
                continue
            new_powers = powers.copy()
            new_powers[index] = scaled
            add('power', index, -1, links.copy(), new_powers)

    return ReplacementSet(tuple(items), generated, skipped, noop)


def _rates(evaluator, action):
    value = evaluator(copy_action(action))
    if not isinstance(value, dict):
        raise ValueError("evaluator must return a prediction dictionary")
    rates = np.asarray(value.get("mean_rate_bps"), float)
    if (rates.shape != (len(action.links),) or not np.isfinite(rates).all()
            or np.any(rates < 0)):
        raise ValueError("evaluator returned invalid mean_rate_bps")
    return rates


def _summary(rates, powers, eta_ref, delta):
    """Per-link EE averaged is the objective; network EE is a diagnostic."""
    network = float(np.sum(rates) / np.sum(powers))
    pair_ee = np.asarray(rates, float) / np.asarray(powers, float)
    mean_pair = float(pair_ee.mean())
    return dict(network_ee=network, mean_pair_ee=mean_pair,
                min_pair_ee=float(pair_ee.min()),
                budget_margin=(None if eta_ref is None
                               else mean_pair - (1 - delta) * eta_ref),
                argmin_link=int(np.argmin(pair_ee)))


def _same_action(first, second):
    return (list(first.links) == list(second.links)
            and np.array_equal(np.asarray(first.powers), np.asarray(second.powers)))


def _proposal_record(proposal, *, n, cfg, candidates, business, ports_master,
                     evaluator, eta_ref, delta):
    record = dict(legal=False, network_ee=None, mean_pair_ee=None,
                  min_pair_ee=None, budget_violation=None, reason="")
    if not _checked(copy_action(proposal), n, cfg, candidates, business, ports_master):
        record["reason"] = "proposal_illegal"
        return record
    rates = _rates(evaluator, proposal)
    powers = np.asarray(proposal.powers, float)
    pair_ee = rates / powers
    record.update(legal=True, network_ee=float(rates.sum() / powers.sum()),
                  mean_pair_ee=float(pair_ee.mean()),
                  min_pair_ee=float(np.min(pair_ee)),
                  budget_violation=(None if eta_ref is None
                                    else max(0., (1 - delta) * eta_ref
                                             - float(pair_ee.mean()))),
                  reason="proposal_evaluated")
    return record


def credit_report(action, forecast, *, n, cfg, candidates, business, ports_master,
                  evaluator, block, forecast_seed, delta, eta_ref,
                  proposal=None, fallback=False, max_evals=48, focus=3):
    """Evaluate legal one-step replacements of ``action`` and rank them.

    ``action``/``forecast`` are the executed action and its cached prediction;
    they are never mutated.  ``max_evals=0`` lifts the cap (n=100 full
    enumeration costs several seconds per block).  Returns a JSON-ready record
    plus the number of evaluator calls made here.
    """
    if max_evals < 0 or focus < 0:
        raise ValueError("max_evals and focus must be nonnegative")
    baseline_rates = np.asarray(forecast["mean_rate_bps"], float)
    if baseline_rates.shape != (len(action.links),):
        raise ValueError("forecast does not match the executed action")
    base = _summary(baseline_rates, action.powers, eta_ref, delta)

    calls = 0
    proposal_record = None
    if proposal is not None and not _same_action(proposal, action):
        proposal_record = _proposal_record(
            proposal, n=n, cfg=cfg, candidates=candidates, business=business,
            ports_master=ports_master, evaluator=evaluator, eta_ref=eta_ref, delta=delta)
        calls += 1

    replacements = legal_replacements(action, n=n, cfg=cfg, candidates=candidates,
                                      business=business, ports_master=ports_master)
    pair_ee = baseline_rates / np.asarray(action.powers, float)
    weakest = set(int(index) for index in np.argsort(pair_ee, kind="stable")[:focus])
    items = replacements.items
    focused = [index for index, item in enumerate(items)
               if item.primary in weakest or item.secondary in weakest]
    outside = [index for index in range(len(items)) if index not in set(focused)]
    ordered = [items[index] for index in focused + outside]
    chosen = ordered if max_evals == 0 else ordered[:max_evals]

    evaluated = []
    for item in chosen:
        rates = _rates(evaluator, item.action)
        calls += 1
        metrics = _summary(rates, item.action.powers, eta_ref, delta)
        evaluated.append(dict(
            kind=item.kind, primary=item.primary, secondary=item.secondary,
            links=[list(link) for link in item.action.links],
            powers=[float(p) for p in item.action.powers],
            mean_pair_ee=metrics["mean_pair_ee"],
            network_ee=metrics["network_ee"], min_pair_ee=metrics["min_pair_ee"],
            delta_mean_pair_ee=metrics["mean_pair_ee"] - base["mean_pair_ee"],
            delta_network_ee=metrics["network_ee"] - base["network_ee"],
            delta_min_pair_ee=metrics["min_pair_ee"] - base["min_pair_ee"],
            budget_margin=metrics["budget_margin"], argmin_link=metrics["argmin_link"],
            # Slot-aligned only: partner swaps re-pair the slots, so this is a
            # rate delta per link slot, never a per-link EE attribution.
            rate_delta_bps=[float(x) for x in (rates - baseline_rates)],
            legal=True))

    best = max(evaluated, key=lambda row: row["delta_mean_pair_ee"]) if evaluated else None
    worst = min(evaluated, key=lambda row: row["delta_mean_pair_ee"]) if evaluated else None
    record = dict(
        schema_version=SCHEMA_VERSION,
        information="conditional_trajectory_forecast_only",
        calibration_status="approximate_not_certified",
        block=int(block), forecast_seed=int(forecast_seed), delta=float(delta),
        eta_ref=None if eta_ref is None else float(eta_ref),
        fallback=bool(fallback), baseline=base, proposal=proposal_record,
        counts=dict(generated=replacements.generated, legal=len(replacements.items),
                    skipped_illegal=replacements.skipped_illegal, noop=replacements.noop,
                    focus=max(focus, 0), evaluated=len(evaluated),
                    max_evals=int(max_evals), truncated=len(chosen) < len(ordered)),
        replacements=evaluated,
        best=None if best is None else {key: best[key] for key in
                                        ("kind", "primary", "secondary",
                                         "delta_mean_pair_ee", "delta_min_pair_ee",
                                         "delta_network_ee", "budget_margin")},
        worst=None if worst is None else {key: worst[key] for key in
                                          ("kind", "primary", "secondary",
                                           "delta_mean_pair_ee", "delta_min_pair_ee",
                                           "delta_network_ee", "budget_margin")})
    return record, calls
