from collections import OrderedDict, namedtuple
from typing import Tuple
from itertools import chain

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
from rlkit.torch.networks import FlattenMlp, ParallelizedEnsembleFlattenMLP
from rlkit.torch.sac.policies import TanhGaussianPolicy
import gtimer as gt

SACNLosses = namedtuple(
    'SACNLosses',
    'policy_loss qf_loss alpha_loss',
)

class SACNTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            policy,
            qfs,
            target_qfs,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            eta=1.,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            # Custom.
            max_value=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.num_qs = self.qfs.ensemble_size
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.max_value = max_value
        self.eta = eta

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss(reduction='none')
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr)

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Gradient logging helper function
        """
        def log_grad(output, stats, root_name, depth = 1):

            for grad_fn, input in output.next_functions:

                stats.update(create_stats_ordered_dict(
                    f"{root_name} / {input[1].name}",
                    ptu.get_numpy(input[1].grad)
                ))

                if depth > 1:
                    log_grad(input, stats, input[1].name, depth - 1)
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        log_grad(losses.policy_loss, stats, 'policy_loss', depth=1)
        self.policy_optimizer.step()

        self.qf_optimizer.zero_grad()
        losses.qf_loss.backward()
        log_grad(losses.qf_loss, stats, 'qf_loss', depth=2)
        self.qf_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qfs, self.target_qfs, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACNLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = self.qfs.sample(obs, new_obs_actions)

        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        qs_pred = self.qfs(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = self.target_qfs.sample(next_obs, new_next_actions)

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = torch.tile(q_target.detach().unsqueeze(0), (self.num_qs, 1, 1))
        if self.max_value is not None:
            q_target = torch.clamp(q_target, max=self.max_value)
        qf_losses = self.qf_criterion(qs_pred, q_target).mean(dim=(1, 2))
        mean_qf_loss = qf_losses.mean()
        total_qf_loss = qf_losses.sum()

        """
        Diversity Loss
        """
        if self.eta > 0:
            obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
            actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
            qs_preds_tile = self.qfs(obs_tile, actions_tile)

            qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
            qs_pred_grads = qs_pred_grads.transpose(0, 1)

            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            masks = torch.eye(self.num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            qs_pred_grads = (1 - masks) * qs_pred_grads
            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self.num_qs - 1)

            total_qf_loss += self.eta * grad_loss
        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF Mean Loss'] = ptu.get_numpy(mean_qf_loss)
            eval_statistics['QF Std Loss'] = np.std(ptu.get_numpy(qf_losses))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()
            if self.eta > 0.:
                eval_statistics['Grad Loss'] = ptu.get_numpy(grad_loss)

        loss = SACNLosses(
            policy_loss=policy_loss,
            qf_loss=total_qf_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.target_qfs,
        )

    @staticmethod
    def get_networks(num_critics,
                     obs_dim,
                     action_dim,
                     layer_size,
                     num_layers=2):
        M = layer_size
        qfs = ParallelizedEnsembleFlattenMLP(
                ensemble_size=num_critics,
                input_size=obs_dim + action_dim,
                hidden_sizes=[M] * num_layers,
                output_size=1,
                layer_norm=None)
        target_qfs = ParallelizedEnsembleFlattenMLP(
                ensemble_size=num_critics,
                input_size=obs_dim + action_dim,
                hidden_sizes=[M] * num_layers,
                output_size=1,
                layer_norm=None)
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )
        networks = {'qfs': qfs,
                    'target_qfs': target_qfs,
                    'policy': policy}
        return networks
