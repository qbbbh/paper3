"""Trainable hybrid multi-agent SAC baseline for a contextual bandit (gamma=0).

All vehicles share an actor.  A centralized, sequential sampler takes only
edges whose removal admits a complete matching.  The categorical joint log
probability is therefore that of the *executed legal matching*, rather than
that of independent conflicting actor proposals.  Its gradient uses a score
function estimator; powers use a reparameterized log-uniform Normal, so the
deterministic head can reach the 1 mW floor instead of stopping at
sigmoid(-5) = 14.38 mW.

Critics learn the *labelled* action.  By default the caller labels the state
with the proposal's own forecast (predicted minimum pair EE and predicted
budget gap), even when the deployment shield executed the reference instead.
That keeps a rejected proposal in the training set with its own poor label;
attaching the reference reward to it would teach the critic the wrong thing.
The executed action is still logged for evaluation.  `label_source` records
which convention produced each record.

No queue, discount bootstrapping, target critic, QoS gate or convergence claim
is introduced.  A learned expectation penalty cannot certify a per-block EE
budget; the caller must evaluate and record any reference fallback separately.
"""
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .allocation import Action, has_perfect_matching
from .models import FeatureBatch, ModelConfig, SharedHybridActor, CentralCritic


POWER_PARAMETERIZATION = 'log_uniform_v1'


POWER_ENDPOINT_TOLERANCE = 1e-3


def latent_from_power(power_w, min_power_w, max_power_w,
                      tolerance=POWER_ENDPOINT_TOLERANCE):
    """Inverse of :func:`power_from_latent` with an explicit endpoint margin.

    The forward map reaches ``min_power_w`` only as ``z -> -inf``, so an exact
    inverse at the endpoint is not finite.  Targets are clipped to
    ``[min*(1+tolerance), max*(1-tolerance)]``; the reconstructed power then
    differs from the request by less than ``tolerance`` (relative), which is
    the tolerance supervised power fitting is accepted under.
    """
    if not 0 < min_power_w < max_power_w:
        raise ValueError('Power limits must satisfy 0 < min < max')
    if not 0 < tolerance < 1:
        raise ValueError('tolerance must be in (0, 1)')
    ratio = math.log(max_power_w / min_power_w)
    low, high = min_power_w * (1 + tolerance), max_power_w * (1 - tolerance)
    clipped = np.clip(np.asarray(power_w, dtype=float), low, high)
    s = np.clip(np.log(clipped / min_power_w) / ratio, 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def power_from_latent(latent, min_power_w, max_power_w):
    """Log-uniform map from an unbounded latent onto [min, max] watts.

    ``p = p_min * (p_max / p_min) ** sigmoid(z)``.  Unlike a linear sigmoid it
    reaches the 1 mW floor asymptotically, so a deterministic mean of -12
    already gives 1.00005 mW.  Returns the power and ``log|dp/dz|`` for the
    change-of-variables term.
    """
    if not 0 < min_power_w < max_power_w:
        raise ValueError('Power limits must satisfy 0 < min < max')
    ratio = max_power_w / min_power_w
    latent = torch.as_tensor(latent)
    s = torch.sigmoid(latent)
    power = min_power_w * torch.exp(math.log(ratio) * s)
    log_dp_dz = (torch.log(power) + math.log(math.log(ratio))
                 + F.logsigmoid(latent) + F.logsigmoid(-latent))
    return power, log_dp_dz


@dataclass
class SACConfig:
    gamma: float = 0.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_discrete: float = 0.01
    alpha_continuous: float = 0.001
    reward_scale: float = 1e9
    cost_weight: float = 1.0
    batch_size: int = 16
    replay_capacity: int = 10000
    seed: int = 0
    device: str = 'cpu'
    # 学习侧只有**一对**功率界，与 v4 的 RF/VLC 分模式区间不兼容（RF 0.01–20 W，
    # VLC 0.001–2 W）。10 车实验不走学习路径，这里保持 v3 原值不动；
    # 若 50/100 车学习线恢复，需要把这里改成按模式取界，不能沿用单区间。
    min_power_w: float = 0.001
    max_power_w: float = 2.0
    gradient_clip: float = 10.0
    power_parameterization: str = POWER_PARAMETERIZATION

    def __post_init__(self):
        if self.gamma != 0:
            raise ValueError('This action-independent mobility baseline requires gamma=0')
        if self.power_parameterization != POWER_PARAMETERIZATION:
            raise ValueError(f'Unsupported power parameterization: {self.power_parameterization}')
        if self.reward_scale <= 0 or min(self.actor_lr, self.critic_lr) <= 0:
            raise ValueError('Reward scale and learning rates must be positive')
        if min(self.alpha_discrete, self.alpha_continuous, self.cost_weight) < 0:
            raise ValueError('Entropy coefficients and cost weight must be nonnegative')
        if self.batch_size < 1 or self.replay_capacity < 1 or self.gradient_clip <= 0:
            raise ValueError('Replay, batch and gradient clip values must be positive')
        if not 0.001 <= self.min_power_w < self.max_power_w <= 2:
            raise ValueError('Active communication powers must stay inside [0.001, 2] W')


@dataclass
class PolicySample:
    action: Action
    log_prob_discrete: float
    log_prob_continuous: float
    trace: list


@dataclass
class ReplayRecord:
    """One labelled state.  ``action`` is what the critic learns, not what ran.

    With proposal labels the stored action is the actor's proposal even when
    the shield executed the reference; ``fallback`` says the two differed.
    """
    features: FeatureBatch
    action: Action
    reward: float
    cost: float
    fallback: bool
    matches_proposal: bool | None
    label_source: str = 'executed_forecast'


@dataclass
class _TensorSample:
    action: Action
    powers: torch.Tensor
    log_prob_discrete: torch.Tensor
    log_prob_continuous: torch.Tensor
    trace: list


def _complete(mask):
    """Use sufficient certificates first; all uncertain cases use blossom."""
    n = len(mask)
    if n == 0:
        return True
    if n % 2:
        return False
    graph = np.asarray(mask, bool) | np.asarray(mask, bool).T
    # Dirac's degree condition guarantees a Hamiltonian cycle, hence a
    # perfect matching for even n.  This is only a sufficient acceptance
    # certificate, never an infeasibility rule, and is cheap on dense RF graphs.
    np.fill_diagonal(graph, False)
    if graph.sum(1).min() >= n // 2:
        return True
    remaining = set(range(n))
    while remaining:
        i = min(remaining)
        remaining.remove(i)
        neighbors = [j for j in sorted(remaining) if graph[i, j]]
        if not neighbors:
            return bool(has_perfect_matching(graph))
        remaining.remove(neighbors[0])
    return True


def _same_action(a, b):
    if list(a.links) != list(b.links) or not np.array_equal(a.powers, b.powers):
        return False
    return all(np.array_equal(a.ports[k], b.ports[k]) for k in ('tx', 'rx'))


class HybridMASAC:
    """Shared hybrid policy, twin central critics and actual-action replay.

    ``observe`` receives an unscaled min-pair EE reward in bit/J.  The replay
    keeps that number and a separately recorded normalized EE budget cost.
    The critic target is reward/reward_scale - cost_weight*cost.  There is no
    next-state target, because actions do not affect future traffic or CSI.
    """
    # v3: log-uniform power head, proposal-labelled replay and log-power critic
    # input.  Older checkpoints mean something different and must be retrained.
    checkpoint_version = 3

    def __init__(self, model_config=None, sac_config=None):
        self.model_config = model_config or ModelConfig()
        self.config = sac_config or SACConfig()
        self.device = torch.device(self.config.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.actor = SharedHybridActor(self.model_config).to(self.device)
            self.critics = nn.ModuleList([
                CentralCritic(self.model_config), CentralCritic(self.model_config)]).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.config.critic_lr)
        self.rng = np.random.default_rng(self.config.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self.replay = deque(maxlen=self.config.replay_capacity)
        self.updates = 0
        self.discrete_baseline = 0.0
        self.provenance = {}

    def _check_features(self, features):
        if features.node_features.shape[1] != self.model_config.node_dim:
            raise ValueError('node_features width does not match ModelConfig')
        if features.edge_features.shape[-1] != self.model_config.edge_dim:
            raise ValueError('edge_features width does not match ModelConfig')

    @staticmethod
    def _legal_choices(adjacency, mode_mask, remaining, anchor):
        """Enumerate (tx, rx, mode) in the fixed order the sampler uses.

        Both the sampler and imitation replay must call this, otherwise a
        supervised target index could silently mean a different action.
        """
        choices = []
        for other in remaining[1:]:
            if not (adjacency[anchor, other] or adjacency[other, anchor]):
                continue
            rest = [v for v in remaining if v not in (anchor, other)]
            if not _complete(adjacency[np.ix_(rest, rest)]):
                continue
            for tx, rx in ((anchor, other), (other, anchor)):
                choices.extend((tx, rx, mode) for mode in range(2)
                               if mode_mask[tx, rx, mode])
        return choices

    def _sample(self, features, deterministic=False):
        self._check_features(features)
        node, edge, mask = features.tensors(self.device)
        scores, means, log_stds = self.actor(node, edge, mask)
        adjacency = features.mode_mask.any(-1)
        if not _complete(adjacency):
            raise ValueError('coverage_failure: no complete matching in causal legal candidates')
        remaining = list(range(features.n))
        selected, sampled_powers, log_discrete, log_continuous, trace = [], [], [], [], []
        ports = dict(tx=np.zeros((features.n, features.n), dtype=int),
                     rx=np.zeros((features.n, features.n), dtype=int))
        while remaining:
            i = remaining[0]
            choices = self._legal_choices(adjacency, features.mode_mask, remaining, i)
            if not choices:
                raise ValueError('coverage_failure: completion mask has no feasible action')
            indices = tuple(torch.tensor([c[d] for c in choices], device=self.device)
                            for d in range(3))
            logits = scores[indices]
            log_probs = torch.log_softmax(logits, dim=0)
            chosen = (int(logits.argmax().item()) if deterministic else
                      int(torch.multinomial(log_probs.exp(), 1, generator=self.generator).item()))
            tx, rx, mode = choices[chosen]
            log_discrete.append(log_probs[chosen])
            mean, log_std = means[tx, rx, mode], log_stds[tx, rx, mode]
            noise = torch.zeros((), device=self.device) if deterministic else torch.randn(
                (), device=self.device, generator=self.generator)
            latent = mean + log_std.exp() * noise
            power, log_jacobian = power_from_latent(
                latent, self.config.min_power_w, self.config.max_power_w)
            normal_logp = -0.5 * (((latent - mean) / log_std.exp()).square()
                                   + 2 * log_std + math.log(2 * math.pi))
            log_continuous.append(normal_logp - log_jacobian)
            sampled_powers.append(power)
            selected.append((tx, rx, 'RF' if mode == 0 else 'VLC'))
            if mode == 1:
                ports['tx'][tx, rx], ports['rx'][tx, rx] = features.port_choices[tx, rx, mode]
            trace.append(dict(anchor=int(i), legal_choices=[tuple(c) for c in choices],
                              chosen_index=chosen, log_probability=float(log_probs[chosen].detach())))
            remaining = [v for v in remaining if v not in (tx, rx)]
        powers = torch.stack(sampled_powers)
        action = Action(selected, powers.detach().cpu().numpy().astype(float).copy(), ports)
        return _TensorSample(action, powers, torch.stack(log_discrete).sum(),
                             torch.stack(log_continuous).sum(), trace)

    def imitation_loss(self, features, action, power_weight=1.0,
                       log_std_target=None, std_weight=0.1):
        """Supervised loss that replays one demonstrated action step by step.

        The demonstration is replayed in the sampler's own anchor order and
        against the same completion mask, so every target index refers to the
        identical categorical distribution the policy would sample from.
        Pairing/direction/mode use cross-entropy; power is fitted in the
        unbounded log-uniform latent, with endpoint tolerance.
        """
        self._check_features(features)
        node, edge, mask = features.tensors(self.device)
        scores, means, log_stds = self.actor(node, edge, mask)
        adjacency = features.mode_mask.any(-1)
        if not _complete(adjacency):
            raise ValueError('coverage_failure: no complete matching in causal legal candidates')
        targets = {}
        for (tx, rx, mode), power in zip(action.links, action.powers):
            key = (int(tx), int(rx))
            if key in targets:
                raise ValueError('supervised action contains a duplicate link')
            targets[key] = (int(tx), int(rx), 0 if mode == 'RF' else 1, float(power))
        if len(targets) != features.n // 2:
            raise ValueError('supervised action must be one complete matching')
        remaining, discrete, power_losses = list(range(features.n)), [], []
        while remaining:
            anchor = remaining[0]
            matches = [key for key in targets if anchor in key]
            if len(matches) != 1:
                raise ValueError('supervised action does not cover every vehicle exactly once')
            tx, rx, mode, target_power = targets.pop(matches[0])
            choices = self._legal_choices(adjacency, features.mode_mask, remaining, anchor)
            if (tx, rx, mode) not in choices:
                raise ValueError('supervised link is not legal under the block-start mask')
            index = choices.index((tx, rx, mode))
            indices = tuple(torch.tensor([c[d] for c in choices], device=self.device)
                            for d in range(3))
            discrete.append(F.cross_entropy(
                scores[indices][None, :], torch.tensor([index], device=self.device)))
            latent_target = torch.tensor(
                float(latent_from_power(target_power, self.config.min_power_w,
                                        self.config.max_power_w)), device=self.device)
            power_losses.append((means[tx, rx, mode] - latent_target).pow(2))
            remaining = [v for v in remaining if v not in (tx, rx)]
        discrete_loss = torch.stack(discrete).mean()
        power_loss = torch.stack(power_losses).mean()
        total = discrete_loss + power_weight * power_loss
        std_loss = torch.zeros((), device=self.device)
        if log_std_target is not None:
            # A supervised warm start should also leave the exploration scale
            # sane; otherwise the deterministic mean is right while stochastic
            # proposals are dominated by an untrained log_std.
            std_loss = (log_stds - float(log_std_target)).pow(2).mean()
            total = total + std_weight * std_loss
        return dict(loss=total, discrete_loss=discrete_loss, power_loss=power_loss,
                    std_loss=std_loss, steps=len(discrete))

    @torch.no_grad()
    def select_action(self, features, deterministic=False):
        sample = self._sample(features, deterministic)
        return PolicySample(sample.action, float(sample.log_prob_discrete),
                            float(sample.log_prob_continuous), sample.trace)

    def _validate_action(self, features, action):
        if len(action.links) != features.n // 2 or np.asarray(action.powers).shape != (features.n // 2,):
            raise ValueError('Replay action must be a complete matching')
        endpoints = [v for tx, rx, _ in action.links for v in (tx, rx)]
        if sorted(endpoints) != list(range(features.n)):
            raise ValueError('Replay action has incomplete/conflicting endpoints')
        powers = np.asarray(action.powers)
        if not np.isfinite(powers).all() or np.any(powers < self.config.min_power_w) or np.any(powers > self.config.max_power_w):
            raise ValueError('Replay power lies outside active power limits')
        for tx, rx, mode in action.links:
            if mode not in ('RF', 'VLC'):
                raise ValueError('Invalid replay mode')
            m = int(mode == 'VLC')
            if not features.mode_mask[tx, rx, m]:
                raise ValueError('Replay action violates candidate/business legality')
            if m and tuple(action.ports[k][tx, rx] for k in ('tx', 'rx')) != tuple(features.port_choices[tx, rx, m]):
                raise ValueError('Replay VLC action differs from fixed causal port choices')

    def observe(self, features, training_action, reward, cost=0.0, fallback=False,
                proposal=None, label_source='executed_forecast'):
        """Store one labelled action, normally the proposal's own forecast.

        ``fallback`` records that the deployment shield executed a different
        action.  It must not relabel the proposal with the reference reward:
        the caller is responsible for passing the proposal's own predicted
        reward and budget gap.  ``proposal`` only marks whether the stored
        action is the actor's own sample.
        """
        if label_source not in ('proposal_forecast', 'reference_forecast',
                                'neighbour_forecast', 'executed_forecast', 'measured'):
            raise ValueError(f'Unknown label source: {label_source}')
        self._check_features(features)
        self._validate_action(features, training_action)
        if not np.isfinite(reward) or not np.isfinite(cost) or cost < 0:
            raise ValueError('Reward and nonnegative budget cost must be finite')
        matches = None if proposal is None else _same_action(proposal.action, training_action)
        if matches is False and not fallback:
            raise ValueError('A labelled action that differs from the proposal requires fallback=True')
        if label_source == 'proposal_forecast' and matches is False:
            raise ValueError('proposal_forecast labels require the proposal action itself')
        self.replay.append(ReplayRecord(features.copy(), copy.deepcopy(training_action),
                                        float(reward), float(cost), bool(fallback), matches,
                                        label_source))

    def update(self, batch_size=None, actor=True):
        """One critic step, optionally followed by one actor step.

        ``actor=False`` is for critic warm-up/gating: the critic can be fitted
        on collected proposals before its ranking is trusted to move the policy.
        """
        size = self.config.batch_size if batch_size is None else int(batch_size)
        if size < 1:
            raise ValueError('batch_size must be positive')
        if len(self.replay) < size:
            return dict(updated=False, replay_size=len(self.replay), required_batch_size=size)
        indices = self.rng.choice(len(self.replay), size=size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_losses = []
        for record in batch:
            target = torch.tensor(record.reward / self.config.reward_scale
                                  - self.config.cost_weight * record.cost, device=self.device)
            critic_losses.extend(F.mse_loss(q(record.features, record.action), target) for q in self.critics)
        critic_loss = torch.stack(critic_losses).mean()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critics.parameters(), self.config.gradient_clip)
        self.critic_optimizer.step()

        # States are exogenous in this contextual bandit.  Any replay state,
        # including one at which a reference was executed, can support a fresh
        # off-policy actor update.  We never reuse a proposal's old log-prob or
        # assign the reference action a likelihood under that proposal.
        eligible = batch
        actor_loss_value, actor_grad_value = None, None
        logpd_values, logpc_values, objective_values = [], [], []
        if eligible and actor:
            # Critic weights stay fixed, but gradients through selected powers
            # are needed for the continuous actor reparameterization.
            for parameter in self.critics.parameters():
                parameter.requires_grad_(False)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_losses = []
            try:
                for record in eligible:
                    sample = self._sample(record.features)
                    q = torch.minimum(*(critic(record.features, sample.action, sample.powers)
                                        for critic in self.critics))
                    continuous_objective = self.config.alpha_continuous * sample.log_prob_continuous - q
                    # Exact score-function derivative of E[alpha_d log pi_d + f].
                    # The baseline comes from previous updates, not this sample.
                    coefficient = (continuous_objective.detach()
                                   + self.config.alpha_discrete * (sample.log_prob_discrete.detach() + 1)
                                   - self.discrete_baseline)
                    actor_losses.append(continuous_objective + coefficient * sample.log_prob_discrete)
                    objective_values.append(float((continuous_objective.detach()
                                                  + self.config.alpha_discrete * sample.log_prob_discrete.detach())))
                    logpd_values.append(float(sample.log_prob_discrete.detach()))
                    logpc_values.append(float(sample.log_prob_continuous.detach()))
                actor_loss = torch.stack(actor_losses).mean()
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
                self.actor_optimizer.step()
                actor_loss_value, actor_grad_value = float(actor_loss.detach()), float(actor_grad)
                self.discrete_baseline = .95 * self.discrete_baseline + .05 * float(np.mean(objective_values))
            finally:
                for parameter in self.critics.parameters():
                    parameter.requires_grad_(True)
        self.updates += 1
        return dict(updated=True, updates=self.updates, replay_size=len(self.replay),
                    critic_loss=float(critic_loss.detach()), critic_gradient_norm=float(critic_grad),
                    actor_loss=actor_loss_value, actor_gradient_norm=actor_grad_value,
                    actor_samples=len(eligible) if actor else 0,
                    fresh_actor_samples=len(eligible) if actor else 0,
                    fallback_samples=sum(record.fallback for record in batch),
                    actor_on_fallback_states=(sum(record.fallback for record in eligible)
                                              if actor else 0),
                    mean_log_prob_discrete=float(np.mean(logpd_values)) if logpd_values else None,
                    mean_log_prob_continuous=float(np.mean(logpc_values)) if logpc_values else None,
                    gamma=0.0, reward_scale=self.config.reward_scale,
                    mean_budget_cost=float(np.mean([r.cost for r in batch])))

    def save(self, path, include_replay=False, provenance=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if provenance is not None:
            if not isinstance(provenance, dict):
                raise ValueError('Checkpoint provenance must be a dictionary')
            self.provenance = copy.deepcopy(provenance)
        state = dict(version=self.checkpoint_version, model_config=asdict(self.model_config),
                     sac_config=asdict(self.config), actor=self.actor.state_dict(),
                     critics=self.critics.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
                     critic_optimizer=self.critic_optimizer.state_dict(), updates=self.updates,
                     discrete_baseline=self.discrete_baseline, numpy_rng=self.rng.bit_generator.state,
                     torch_rng=self.generator.get_state(), provenance=copy.deepcopy(self.provenance),
                     metadata=dict(stage='development', gamma=0.0, trained_to_convergence=False,
                                   actor_information='causal_features_only',
                                   critic_actions='labelled_proposal_or_executed_forecast',
                                   power_parameterization=self.config.power_parameterization))
        if include_replay:
            state['replay'] = list(self.replay)
        torch.save(state, path)

    @classmethod
    def load(cls, path, device='cpu'):
        # These are locally created training checkpoints, including optimizer
        # and optional replay objects, not arbitrary downloaded model files.
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get('version') != cls.checkpoint_version:
            raise ValueError('Unsupported MASAC checkpoint version')
        config = dict(state['sac_config'], device=device)
        trainer = cls(ModelConfig(**state['model_config']), SACConfig(**config))
        trainer.actor.load_state_dict(state['actor'])
        trainer.critics.load_state_dict(state['critics'])
        trainer.actor_optimizer.load_state_dict(state['actor_optimizer'])
        trainer.critic_optimizer.load_state_dict(state['critic_optimizer'])
        trainer.updates = state['updates']
        trainer.discrete_baseline = state['discrete_baseline']
        trainer.provenance = copy.deepcopy(state.get('provenance', {}))
        trainer.rng.bit_generator.state = state['numpy_rng']
        trainer.generator.set_state(state['torch_rng'].cpu())
        trainer.replay.extend(state.get('replay', []))
        return trainer
