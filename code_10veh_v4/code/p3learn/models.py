"""Shared vehicle policies and centralized critics using causal feature arrays.

The RF/VLC graph has distinct relation weights.  It needs only PyTorch, not
torch_geometric.  An MLP encoder is the 50-car baseline; the graph encoders are
the 100-car representation.  Neither encoder receives future scene frames.

``hetero_graph_v2`` splits each mode into two disjoint relation types:
data-eligible candidate edges, and non-candidate edges that are still strong
enough to interfere.  The split is derived from block-start masks and belief
gains only, never from the realized action, so the decoder cannot be fed its
own output.  This is "data-eligible vs non-data-eligible", not "signal vs
interference": a selected candidate also interferes, and its interference role
is already carried by edge features 10-11.

Directed asymmetry is intentional.  For VLC the port pair of (u, v) is only a
causal proxy for the interference channel, which physically depends on u's
transmit port and v's receive port.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


@dataclass
class FeatureBatch:
    node_features: np.ndarray
    edge_features: np.ndarray
    mode_mask: np.ndarray
    port_choices: np.ndarray

    def __post_init__(self):
        self.node_features = np.asarray(self.node_features, dtype=np.float32)
        self.edge_features = np.asarray(self.edge_features, dtype=np.float32)
        self.mode_mask = np.asarray(self.mode_mask, dtype=bool)
        self.port_choices = np.asarray(self.port_choices, dtype=np.int64)
        n = len(self.node_features)
        if self.node_features.ndim != 2 or n == 0 or n % 2:
            raise ValueError('Features require a positive even number of vehicles')
        if self.edge_features.ndim != 4 or self.edge_features.shape[:3] != (n, n, 2):
            raise ValueError('edge_features must have shape N x N x 2 x E')
        if self.mode_mask.shape != (n, n, 2) or self.port_choices.shape != (n, n, 2, 2):
            raise ValueError('Invalid mode mask or port choices shape')
        if self.mode_mask[np.arange(n), np.arange(n)].any():
            raise ValueError('Self links must be masked')
        if not np.isfinite(self.node_features).all() or not np.isfinite(self.edge_features).all():
            raise ValueError('Model features must be finite; encode missing CSI ages explicitly')
        vlc_ports = self.port_choices[:, :, 1, :][self.mode_mask[:, :, 1]]
        if vlc_ports.size and not np.isin(vlc_ports, (0, 1)).all():
            raise ValueError('Eligible VLC links need fixed valid tx/rx ports')

    @property
    def n(self):
        return len(self.node_features)

    def copy(self):
        return FeatureBatch(*(a.copy() for a in (
            self.node_features, self.edge_features, self.mode_mask, self.port_choices)))

    def tensors(self, device='cpu'):
        return (torch.as_tensor(self.node_features, device=device),
                torch.as_tensor(self.edge_features, device=device),
                torch.as_tensor(self.mode_mask, device=device))


ENCODERS = ('mlp', 'hetero_graph', 'hetero_graph_v2')
INTERFERENCE_RANGE_DB = 30.
# edge[..., 0] is log10(gain)/15, so one decade is 1/15 and 30 dB is 3 decades.
INTERFERENCE_SCORE_WINDOW = INTERFERENCE_RANGE_DB / 150.


@dataclass
class ModelConfig:
    node_dim: int = 12
    edge_dim: int = 12
    hidden_dim: int = 64
    encoder: str = 'mlp'
    message_layers: int = 2

    def __post_init__(self):
        if self.encoder not in ENCODERS:
            raise ValueError('encoder must be one of ' + ', '.join(ENCODERS))
        if min(self.node_dim, self.edge_dim, self.hidden_dim) < 1 or self.message_layers < 1:
            raise ValueError('Model dimensions and layer count must be positive')


def mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, out_dim))


class RelationLayer(nn.Module):
    """Separate RF/VLC messages, aggregated in both directed orientations."""
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.relations = nn.ModuleList([
            mlp(2 * hidden_dim + edge_dim, hidden_dim, hidden_dim) for _ in range(2)])
        self.update = mlp(5 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge, mask):
        n = h.shape[0]
        hi = h[:, None].expand(n, n, -1)
        hj = h[None, :].expand(n, n, -1)
        summaries = [h]
        for mode, relation in enumerate(self.relations):
            active = mask[:, :, mode].to(h.dtype).unsqueeze(-1)
            message = relation(torch.cat((hi, hj, edge[:, :, mode]), -1)) * active
            outgoing = message.sum(1) / active.sum(1).clamp_min(1)
            incoming = message.sum(0) / active.sum(0).clamp_min(1)
            summaries.extend((outgoing, incoming))
        return self.norm(h + self.update(torch.cat(summaries, -1)))


def relation_masks(mask, edge):
    """Four disjoint directed relation masks from block-start beliefs.

    Returns candidate and non-candidate-interference masks for RF then VLC.
    ``candidate`` is the legal candidate/business mask; ``interference`` is a
    non-candidate off-diagonal pair with a usable gain and within
    INTERFERENCE_RANGE_DB of the strongest edge arriving at the same victim.
    Empty relations are expected: on a complete candidate graph the
    interference masks are empty by construction, and VLC interference covers
    only a minority of nodes.
    """
    n = mask.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=mask.device)
    output = []
    for mode in (0, 1):
        candidate = mask[:, :, mode] & off
        score = edge[:, :, mode, 0]
        # A VLC pair with zero optical gain is absence of a channel, not an
        # interference edge; log10(1e-30)/15 is the floor used by the features.
        has_gain = score > -2. + 1e-6
        vmax = score.masked_fill(~off, torch.finfo(score.dtype).min).amax(dim=0)
        interference = (off & ~candidate & has_gain
                        & (score >= vmax[None, :] - INTERFERENCE_SCORE_WINDOW))
        output.extend((candidate, interference))
    return output


def _masked_mean(message, active, dim):
    weight = active.to(message.dtype)
    return (message * weight.unsqueeze(-1)).sum(dim) / weight.sum(dim).clamp_min(1).unsqueeze(-1)


def _masked_min(message, active, dim):
    """Masked minimum over ``dim``; an empty set yields exact zeros, not NaN."""
    sentinel = torch.finfo(message.dtype).max
    values = message.masked_fill(~active.unsqueeze(-1), sentinel).amin(dim)
    return torch.where(active.sum(dim).unsqueeze(-1) > 0, values, torch.zeros_like(values))


class HeteroRelationLayer(nn.Module):
    """Candidate vs non-candidate interference relations, mean and min pooled.

    Minimum pooling keeps the weakest incoming/outgoing channel from being
    averaged away, which is what a max-min objective needs.  Each relation has
    its own parameters, and a per-relation fuse reconciles the different scales
    of mean- and min-pooled vectors before the residual update.
    """
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.relations = nn.ModuleList([
            mlp(2 * hidden_dim + edge_dim, hidden_dim, hidden_dim) for _ in range(4)])
        self.fuse = nn.ModuleList([
            mlp(2 * hidden_dim, hidden_dim, hidden_dim) for _ in range(4)])
        self.update = mlp(9 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge, mask):
        n = h.shape[0]
        hi = h[:, None].expand(n, n, -1)
        hj = h[None, :].expand(n, n, -1)
        summaries = [h]
        for index, active in enumerate(relation_masks(mask, edge)):
            mode = index // 2
            message = self.relations[index](torch.cat((hi, hj, edge[:, :, mode]), -1))
            for dim in (1, 0):
                summaries.append(self.fuse[index](torch.cat(
                    (_masked_mean(message, active, dim), _masked_min(message, active, dim)), -1)))
        return self.norm(h + self.update(torch.cat(summaries, -1)))


class StateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = mlp(config.node_dim, config.hidden_dim, config.hidden_dim)
        if config.encoder == 'mlp':
            layers = []
        elif config.encoder == 'hetero_graph':
            layers = [RelationLayer(config.hidden_dim, config.edge_dim)
                      for _ in range(config.message_layers)]
        elif config.encoder == 'hetero_graph_v2':
            layers = [HeteroRelationLayer(config.hidden_dim, config.edge_dim)
                      for _ in range(config.message_layers)]
        else:
            # Never fall back to the MLP silently: a typo must not train a
            # different architecture than the one requested.
            raise ValueError('encoder must be one of ' + ', '.join(ENCODERS))
        self.layers = nn.ModuleList(layers)

    def forward(self, node, edge, mask):
        h = self.input(node)
        for layer in self.layers:
            h = layer(h, edge, mask)
        return h


class SharedHybridActor(nn.Module):
    """A shared vehicle head scores receiver, direction, mode and power.

    Matching constraints are imposed by the probability distribution in masac,
    before sampling.  The continuous head is conditional on the selected edge.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = StateEncoder(config)
        self.head = mlp(2 * config.hidden_dim + config.edge_dim + 2,
                        config.hidden_dim, 3)

    def forward(self, node, edge, mask):
        h = self.encoder(node, edge, mask)
        n = h.shape[0]
        hi = h[:, None, None].expand(n, n, 2, -1)
        hj = h[None, :, None].expand(n, n, 2, -1)
        mode = torch.eye(2, dtype=h.dtype, device=h.device)[None, None].expand(n, n, 2, 2)
        output = self.head(torch.cat((hi, hj, edge, mode), -1))
        # The power latent stays unbounded: masac maps it log-uniformly onto
        # [1 mW, 2 W], so the deterministic floor is approached as z -> -inf.
        # Clamping here at -5 would pin the minimum at 14.38 mW and make the
        # reference 1 mW action unreachable by construction.
        return output[..., 0], output[..., 1], output[..., 2].clamp(-5, 1)


def critic_power_feature(powers):
    """Log10 power in watts, shifted so 1 mW is 0 (2 W is 3.301).

    The critic previously saw ``power/2``: 1 mW entered as 0.0005 and 1.05 mW
    as 0.0005, so the action-value head was effectively blind to power in the
    operating range.  A measured sweep showed Q flat from 1 mW to 100 mW while
    the true target fell from +1.0 to −0.9.
    """
    return torch.log10(torch.clamp(powers, min=1e-6)) + 3.0


class CentralCritic(nn.Module):
    """Permutation-pooled centralized Q(s, executed matching, powers, ports).

    The v2 graph encoder adds a weakest-candidate-edge summary per mode, so the
    critic can see a weak link that the executed matching did not select.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = StateEncoder(config)
        self.graph = config.encoder == 'hetero_graph_v2'
        self.link = mlp(2 * config.hidden_dim + config.edge_dim + 5,
                        config.hidden_dim, config.hidden_dim)
        self.output = mlp((7 if self.graph else 5) * config.hidden_dim, config.hidden_dim, 1)
        if self.graph:
            self.edge_proj = mlp(config.edge_dim, config.hidden_dim, config.hidden_dim)

    def forward(self, features, action, powers=None):
        device = next(self.parameters()).device
        node, edge, mask = features.tensors(device)
        h = self.encoder(node, edge, mask)
        tx = torch.tensor([a[0] for a in action.links], device=device, dtype=torch.long)
        rx = torch.tensor([a[1] for a in action.links], device=device, dtype=torch.long)
        modes = torch.tensor([0 if a[2] == 'RF' else 1 for a in action.links],
                             device=device, dtype=torch.long)
        if powers is None:
            powers = torch.as_tensor(np.asarray(action.powers, dtype=np.float32), device=device)
        onehot = torch.nn.functional.one_hot(modes, 2).to(h.dtype)
        ports = np.array([[action.ports['tx'][i, j], action.ports['rx'][i, j]]
                          if mode == 'VLC' else [0, 0] for i, j, mode in action.links], dtype=np.float32)
        port_tensor = torch.as_tensor(ports, device=device)
        selected = self.link(torch.cat((h[tx], h[rx], edge[tx, rx, modes], onehot,
                                       critic_power_feature(powers)[:, None],
                                       port_tensor), -1))
        pooled = [h.mean(0), h.amax(0), selected.mean(0),
                  selected.amax(0), selected.amin(0)]
        if self.graph:
            projected = self.edge_proj(edge)
            for mode in (0, 1):
                active = mask[:, :, mode].reshape(-1)
                message = projected[:, :, mode].reshape(-1, projected.shape[-1])
                weakest = _masked_min(message, active, 0)
                pooled.append(torch.where(active.any(), weakest,
                                          torch.zeros_like(weakest)))
        return self.output(torch.cat(pooled)).squeeze(-1)
