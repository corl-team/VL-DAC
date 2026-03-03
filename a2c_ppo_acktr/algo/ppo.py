import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate
from tqdm import tqdm


class PPO:
    def __init__(
        self,    
        actor_critic,
        optimizer,
        accelerator,
        clip_param,
        ppo_epoch,
        mini_batch_size,
        value_loss_coef,
        entropy_coef,
        kl_beta,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        grad_accum_steps=128
    ):

        self.actor_critic = actor_critic

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.kl_beta = kl_beta
        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator
        self.grad_accum_steps = grad_accum_steps

    def update(self, rollouts, only_value_loss=False, kl=False):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages_raw = advantages.clone()  # Store raw advantages before normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_losses_to_save = []
        action_losses_to_save = []
        kls_to_save = []
        grad_step = 0
        self.actor_critic.train()
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.mini_batch_size
            )
            for i, sample in tqdm(enumerate(data_generator), desc="PPO training, epoch {}".format(e)):
                with self.accelerator.accumulate(self.actor_critic):
                    grad_step += 1
                    (
                        obs_batch,
                        output_ids_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        reference_log_probs_batch,
                        adv_targ,
                    ) = sample
                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        obs_batch, output_ids_batch
                    )
                    # values and action_log_probs on two different devices!! because they come from two llava
                    if torch.isnan(action_log_probs).any():
                        continue
                    old_action_log_probs_batch = old_action_log_probs_batch.to(
                        action_log_probs.device
                    ).view(-1)

                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)
                    reference_log_probs_batch = reference_log_probs_batch.to(values.device)

                    per_token_log_ratio= action_log_probs - old_action_log_probs_batch
                    per_token_ratio = torch.exp(per_token_log_ratio)
                    assert not reference_log_probs_batch.requires_grad, "reference_log_probs_batch should not have gradient"
                    per_token_kl = (
                        torch.exp(reference_log_probs_batch - action_log_probs) - (reference_log_probs_batch - action_log_probs) - 1
                    )
                    # # raise ValueError(f"Ratio is {ratio} on first PPO epoch, which is not 1.0")
                    # if e == 0 and i < self.grad_accum_steps and not torch.allclose(
                    #     per_token_ratio, torch.tensor(1.0, dtype=torch.bfloat16), atol=1e-5
                    # ):  
                    #     raise ValueError(f"Ratio is {per_token_ratio} on first PPO epoch, which is not 1.0")
                    
                    surr1 = per_token_ratio * adv_targ
                    surr2 = (
                        torch.clamp(per_token_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        * adv_targ
                    )
                    ## ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    if torch.any(per_token_ratio > 10):
                        per_token_action_loss = -surr2#.mean()
                        print("Ratio > 10")
                    else:
                        per_token_action_loss = -torch.min(surr1, surr2)#.mean()

                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + (
                            values - value_preds_batch
                        ).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (value_pred_clipped - return_batch).pow(
                            2
                        )
                        value_loss = (
                            self.value_loss_coef * torch.max(value_losses, value_losses_clipped).mean()
                        )
                    else:
                        value_loss =  self.value_loss_coef * (return_batch - values).pow(2).mean()

                    if only_value_loss:
                        loss = value_loss
                    else:
                        if kl:
                            per_token_loss = value_loss + per_token_action_loss + self.kl_beta * per_token_kl
                            loss = (per_token_loss).sum() / action_log_probs.shape[-1]
                        else:
                            per_token_loss = value_loss + per_token_action_loss
                            assert action_log_probs.shape[-1] > 1, "action_log_probs.shape[-1] is 1"
                            loss = per_token_loss
                    
                    # if only_value_loss:
                    #     loss = value_loss
                    # else:
                    #     per_token_loss = value_loss + per_token_action_loss + self.kl_beta * per_token_kl
                    #     assert action_log_probs.shape[-1] > 1, "action_log_probs.shape[-1] is 1"
                        
                
                    try:
                        assert not torch.isnan(value_loss), "value_loss is nan"
                        assert not torch.isnan(loss), "action_loss is nan"
                    except:
                        print("value/action loss is nan")
                        exit(1)
                    
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:

                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    value_loss_epoch += value_loss.item()
                    value_losses_to_save.append(value_loss_epoch)
                    action_loss_epoch += per_token_action_loss.mean().item()
                    action_losses_to_save.append(action_loss_epoch)
                    dist_entropy_epoch += per_token_kl.mean().item()
                    kls_to_save.append(dist_entropy_epoch)

        value_loss_epoch /= grad_step
        action_loss_epoch /= grad_step
        dist_entropy_epoch /= grad_step

        # Compute advantage statistics for logging
        # advantage_stats = {
        #     'raw_mean': advantages_raw.mean().item(),
        #     'raw_std': advantages_raw.std().item(),
        #     'raw_min': advantages_raw.min().item(),
        #     'raw_max': advantages_raw.max().item(),
        #     'raw_median': advantages_raw.median().item(),
        #     'normalized_mean': advantages.mean().item(),
        #     'normalized_std': advantages.std().item(),
        #     'normalized_min': advantages.min().item(),
        #     'normalized_max': advantages.max().item(),
        #     'normalized_median': advantages.median().item(),
        #     'zero_advantage_ratio': (advantages_raw.abs() < 1e-6).float().mean().item(),
        #     'small_advantage_ratio': (advantages_raw.abs() < 0.01).float().mean().item(),
        # }

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, value_losses_to_save, action_losses_to_save, kls_to_save
