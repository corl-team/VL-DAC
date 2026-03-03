import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import copy
import json
import os


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, action_space, max_new_tokens, algorithm="ppo", log_path=None):
        # Start of Selection
        self.obs = [{} for _ in range(num_steps + 1)]
        # hard-code to cases of max_new_tokens being smaller than 32
        self.output_ids = []
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = []
        self.reference_log_probs = []
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0
        self.algorithm = algorithm
        if log_path is not None:
            os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path

    def insert(
        self,
        obs,
        output_ids,
        actions,
        action_log_probs,
        reference_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks,
    ):  
        reference_log_probs_copy = copy.deepcopy(reference_log_probs)
        output_ids_copy = copy.deepcopy(output_ids)
        action_log_probs_copy = copy.deepcopy(action_log_probs)
        self.obs[self.step + 1] = copy.deepcopy(obs)
        self.output_ids.append(output_ids_copy)
        self.actions[self.step].copy_(actions)
        self.action_log_probs.append(action_log_probs_copy)
        self.reference_log_probs.append(reference_log_probs_copy)
        if self.algorithm == "ppo":
            self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self, iteration):
        self.log_info(iteration)
        self.obs[0] = copy.deepcopy(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.output_ids = []
        self.action_log_probs = []
        self.reference_log_probs = []
    
    def log_info(self, iteration):
        pass
        # base_path = os.path.join(self.log_path, f"{iteration}")
        # os.makedirs(base_path, exist_ok=True)
        # base_filename = os.path.join(base_path, "storage_log_0.json")
        # if not os.path.exists(base_filename):
        #     filename = base_filename
        # else:
        #     index = 1
        #     while os.path.exists(os.path.join(base_path, f"storage_log_{index}.json")):
        #         index += 1
        #     filename = os.path.join(base_path, f"storage_log_{index}.json")
        # data = {
        #     "returns": self.returns.tolist(),
        #     "rewards": self.rewards.tolist(),
        #     "masks": self.masks.tolist(),
        #     "actions": self.actions.tolist(),
        #     "values": self.value_preds.tolist(),
        #     "log_probs": self.action_log_probs.tolist(),
        # }
        # with open(filename, "w") as f:
        #     json.dump(data, f)

    def compute_returns(
        self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True, algorithm="ppo"
    ):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                    ) * self.bad_masks[step + 1] + (
                        1 - self.bad_masks[step + 1]
                    ) * self.value_preds[
                        step
                    ]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )
        #self.log_info()

    def feed_forward_generator(self, advantages, mini_batch_size=1):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )
        
        for indices in sampler:
            # indices = [indices]
            obs_batch = self.obs[:-1][indices[0]]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            output_ids_batch = self.output_ids[indices[0]]
            value_preds_batch = self.value_preds[:-1][indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs[indices[0]]
            reference_log_probs_batch = self.reference_log_probs[indices[0]]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, output_ids_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, reference_log_probs_batch, adv_targ
