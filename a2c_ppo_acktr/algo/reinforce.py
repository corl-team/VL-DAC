import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate
from tqdm import tqdm


class PPO:
    def __init__(
        self,    
        model,
        optimizer,
        accelerator,
        clip_param,
        ppo_epoch,
        mini_batch_size,
        value_loss_coef,
        entropy_coef,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        grad_accum_steps=128
    ):

        self.model = model

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator
        self.grad_accum_steps = grad_accum_steps

    def update(self, rollouts):
        advantages = rollouts.returns[:-1]

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
            for i, sample in tqdm(enumerate(data_generator), desc="REINFORCE training, epoch {}".format(e)):
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
                        adv_targ,
                    ) = sample
                    # Reshape to do in a single forward pass for all steps
                    action_log_probs = self.model.evaluate_actions(
                        obs_batch, output_ids_batch
                    )
                    # values and action_log_probs on two different devices!! because they come from two llava
                    policy_loss = -action_log_probs * adv_targ
                    loss = policy_loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:

                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    action_losses_to_save.append(action_loss_epoch)

        action_loss_epoch /= grad_step
        dist_entropy_epoch /= grad_step

        return 0, action_loss_epoch, 0, 0, 0, 0
