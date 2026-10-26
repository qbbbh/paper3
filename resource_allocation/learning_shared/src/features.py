"""Causal features with documented units, no Scene or oracle argument.

Node columns 8-11 summarise the weakest incident candidate edge per vehicle.
An "incident" directed edge is one where the vehicle is the transmitter *or*
the receiver; each directed edge is counted once per endpoint.  All four
columns use block-start beliefs only.  Empty sets take explicit sentinels:
-2.0 for the log-gain minima (= log10(1e-30)/15, i.e. no usable channel),
0.0 for the mean variance and for the stale-CSI fraction.

"Stale CSI" reuses the edge channels rather than introducing a new constant:
age clipped to 1 s already saturates at 1.0 and the missing flag is explicit,
so ``age >= 1.0 or missing`` marks every missing and every >=1 s observation.
"""
import numpy as np
from .models import FeatureBatch

NODE_DIM, EDGE_DIM = 12, 12
FEATURE_VERSION = "causal_normalized_12x12_v2"
NO_GAIN_SENTINEL = -2.0


def _incident_sum(values, mask):
    """Sum over directed edges incident to each node, counted once per endpoint.

    ``values`` and ``mask`` are N x N for one mode; a directed edge (i, j)
    contributes to node i as transmitter and to node j as receiver.
    """
    outgoing = np.where(mask, values, 0.).sum(1)
    incoming = np.where(mask.T, values.T, 0.).sum(1)
    return outgoing + incoming


def _incident_min(values, mask):
    """Minimum over incident directed edges; NO_GAIN_SENTINEL when none."""
    big = np.finfo(np.float64).max
    outgoing = np.where(mask, values, big).min(1)
    incoming = np.where(mask.T, values.T, big).min(1)
    combined = np.minimum(outgoing, incoming)
    return np.where(combined >= big, NO_GAIN_SENTINEL, combined)


def build_features(state, belief, geometry_rf, geometry_vlc, ports, business):
    n = len(state["speed_mps"])
    mask = np.stack([belief["rf_candidate"], belief["vlc_candidate"]], axis=-1)
    mask &= np.asarray(business, bool)[..., None]
    mask[np.arange(n), np.arange(n)] = False
    xy = np.asarray(state["center_xy_m"], float)
    speed, heading = state["speed_mps"], state["heading_rad"]
    age = np.minimum(np.nan_to_num(belief["rf_age_s"], posinf=1.), 1.)
    edge = np.zeros((n, n, 2, EDGE_DIM))
    choices = np.zeros((n, n, 2, 2), dtype=int)
    choices[..., 0, :] = -1
    choices[..., 1, 0], choices[..., 1, 1] = ports["tx"], ports["rx"]
    ii, jj = np.indices((n, n))
    tp, rp = ports["tx"], ports["rx"]
    velocity = np.asarray(speed)[:, None] * np.stack([np.cos(heading), np.sin(heading)], -1)
    relspeed = np.linalg.norm(velocity[:, None] - velocity[None], axis=-1)
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    for mode, prefix in enumerate(("rf", "vlc")):
        def selected(key):
            values = np.asarray(belief[prefix + "_" + key])
            return values if mode == 0 else values[ii, jj, tp, rp]
        gain = selected("gain_mean")
        mu, var = selected("state_mean"), selected("state_var")
        geom = geometry_rf["gain"] if mode == 0 else geometry_vlc["gain"][ii, jj, tp, rp]
        loggain = np.log10(np.maximum(gain, 1e-30)) / 15.
        relative_var = var / np.maximum(np.abs(mu) ** 2 + var, 1e-20) if mode == 0 else var
        incoming = np.maximum(gain.sum(0)[None] - gain, 0)
        outgoing = np.maximum(gain.sum(1)[:, None] - gain, 0)
        age_s = selected("age_s")
        edge[..., mode, :] = np.stack([
            loggain, np.clip(relative_var, 0, 10), np.minimum(np.nan_to_num(age_s, posinf=1), 1),
            ~np.isfinite(age_s), distance / 100.,
            geometry_rf["los"] if mode == 0 else geom > 0,
            relspeed / 20., np.log10(np.maximum(geom, 1e-30)) / 15.,
            np.clip(selected("gain_low") / np.maximum(gain, 1e-30), 0, 10),
            np.clip(selected("gain_high") / np.maximum(gain, 1e-30), 0, 10),
            np.log1p(incoming / np.maximum(gain, 1e-30)) / 10.,
            np.log1p(outgoing / np.maximum(gain, 1e-30)) / 10.], -1)
    # Node summaries are aggregated from the same edge tensor so there is one
    # source of truth for what a candidate channel looks like.
    legal = mask.astype(bool)
    counts = legal.sum(1) + legal.sum(0)
    rf_min = _incident_min(edge[..., 0, 0], legal[..., 0])
    vlc_min = _incident_min(edge[..., 1, 0], legal[..., 1])
    rf_var_sum = _incident_sum(edge[..., 0, 1], legal[..., 0])
    rf_var = np.where(counts[:, 0] > 0, rf_var_sum / np.maximum(counts[:, 0], 1), 0.)
    stale = np.zeros(n)
    for mode in (0, 1):
        flagged = ((edge[..., mode, 2] >= 1.) | (edge[..., mode, 3] > 0)).astype(float)
        stale += _incident_sum(flagged, legal[..., mode])
    stale_fraction = np.where(counts.sum(1) > 0, stale / np.maximum(counts.sum(1), 1), 0.)
    nodes = np.column_stack([(xy - xy.mean(0)) / 100., np.cos(heading), np.sin(heading),
                             speed / 10., mask[..., 0].mean(1), mask[..., 1].mean(1), age.mean(1),
                             rf_min, vlc_min, rf_var, stale_fraction])
    return FeatureBatch(nodes.astype(np.float32), edge.astype(np.float32), mask, choices)
