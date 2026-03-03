"""
Modular Trainer class for VLM RL training.
Handles the main training loop with configurable components.
"""

import os
import time
import copy
from collections import deque
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import accelerate
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
from tqdm import tqdm

from .environments import EnvironmentRegistry, EnvironmentWrapper
from .models import ModelRegistry
from .models.base import VLMValueModel
from .storage import RolloutStorage
from . import algo


class VLMTrainer:
    """
    Modular trainer for VLM-based reinforcement learning.
    
    Supports different:
    - Environments (MiniWorld, ALFWorld, WebShop, GymCards)
    - Models (Qwen2VL, Gemma3, LLaVA)
    - Algorithms (PPO with token-level rewards)
    """
    
    def __init__(
        self,
        config: "TrainerConfig",
        accelerator: Optional[accelerate.Accelerator] = None,
    ):
        self.config = config
        
        # Initialize accelerator
        if accelerator is None:
            self.accelerator = accelerate.Accelerator(
                gradient_accumulation_steps=config.grad_accum_steps
            )
        else:
            self.accelerator = accelerator
        
        self.device = self.accelerator.device
        set_seed(config.seed, device_specific=True)
        
        # Initialize directories
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Components (initialized lazily)
        self.env: Optional[EnvironmentWrapper] = None
        self.model_adapter = None
        self.value_model: Optional[VLMValueModel] = None
        self.actor_critic = None
        self.agent = None
        self.optimizer = None
        self.lr_scheduler = None
        self.rollouts = None
        
        # Tracking
        self.episode_rewards = deque(maxlen=config.eval_num_per_episode)
        self.episode_success_rate = deque(maxlen=config.eval_num_per_episode)
        self.running_episode_rewards = torch.zeros(config.num_processes).flatten()
        self.image_observations = deque(maxlen=config.max_image_obs_len)
        
        # Logging
        self.wandb_run = None
        
    def setup(self):
        """Set up all training components."""
        self._setup_model()
        self._setup_environment()
        self._setup_policy()
        self._setup_optimizer()
        self._setup_agent()
        self._setup_storage()
        self._setup_logging()
        
    def _setup_model(self):
        """Load and configure the VLM model."""
        print(f"Loading model: {self.config.model_path}")
        
        self.model_adapter = ModelRegistry.create(
            model_path=self.config.model_path,
            cache_dir=self.config.cache_dir,
            device=self.device,
            use_peft=self.config.use_peft,
            peft_config={
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            }
        )
        
        # Create value model
        self.value_model = VLMValueModel(
            self.model_adapter.model,
            self.model_adapter.hidden_size
        )
        self.value_model = self.value_model.to(self.device)
        
        print(f"Model loaded. Hidden size: {self.model_adapter.hidden_size}")
        
    def _setup_environment(self):
        """Set up the environment."""
        print(f"Creating environment: {self.config.env_name}")
        
        self.env = EnvironmentRegistry.create(
            env_name=self.config.env_name,
            seed=self.config.seed,
            rank=self.accelerator.process_index,
            max_episode_steps=self.config.max_episode_steps,
            max_image_obs_len=self.config.max_image_obs_len,
            log_dir=os.path.join(self.config.log_dir, self.config.env_name),
            prompt_version=self.config.prompt_version,
        )
        
        print(f"Environment created: {self.config.env_name}")
        
    def _setup_policy(self):
        """Set up the actor-critic policy."""
        from .model import VLMPolicy
        
        self.actor_critic = VLMPolicy(
            accelerator=self.accelerator,
            processor=self.model_adapter.processor,
            value_model=self.value_model,
            reference_model=None,
            projection_f=lambda x: self.env.env.text_to_action(x[0]) if x else None,
            args=self.config,
        )
        
    def _setup_optimizer(self):
        """Set up optimizer and learning rate scheduler."""
        self.optimizer = optim.Adam([
            {'params': self.value_model.parameters(), 'lr': self.config.init_lr},
        ], eps=self.config.eps, weight_decay=self.config.weight_decay)
        
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.lr_max_steps,
            eta_min=self.config.end_lr
        )
        
        # Prepare with accelerator
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
        
        self.actor_critic, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.actor_critic, self.optimizer, self.lr_scheduler
        )
        
    def _setup_agent(self):
        """Set up the RL agent."""
        self.agent = algo.PPO(
            self.actor_critic,
            self.optimizer,
            self.accelerator,
            self.config.clip_param,
            self.config.ppo_epoch,
            self.config.mini_batch_size,
            self.config.value_loss_coef,
            self.config.entropy_coef,
            self.config.kl_beta,
            max_grad_norm=self.config.max_grad_norm,
            grad_accum_steps=self.config.grad_accum_steps
        )
        
    def _setup_storage(self):
        """Set up rollout storage."""
        self.rollouts = RolloutStorage(
            self.config.num_steps,
            1,
            self.env.action_space,
            self.config.max_new_tokens,
            log_path=os.path.join(self.config.log_dir, self.config.env_name, "rollouts")
        )
        
    def _setup_logging(self):
        """Set up logging (wandb)."""
        if self.config.use_wandb:
            import wandb
            run_name_prefix = "debug-" if self.config.debug else ""
            run_name = run_name_prefix + self.config.wandb_run
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                group=run_name,
                config=vars(self.config)
            )
            
    def train(self):
        """Main training loop."""
        self.setup()
        
        num_updates = int(self.config.num_env_steps) // self.config.num_steps // self.config.num_processes
        start_time = time.time()
        
        observation = None
        infos = []
        
        for j in tqdm(range(num_updates), desc="Training"):
            for step in tqdm(range(self.config.num_steps), 
                           desc=f"Step on epoch {j}, rank {self.accelerator.process_index}"):
                
                # Initialize observation on first step

                # print(f"[DEBUG] Begin step: j={j}, step={step}")

                if j == 0 and step == 0:
                    # print("[DEBUG] Resetting environment")
                    observation = self.env.reset()
                    # print(f"[DEBUG] Initial observation: {observation}")
                    self.rollouts.obs[0] = observation

                # print("[DEBUG] Getting action from policy")
                value, output_id, action, tokens_log_probs, text_action = self.actor_critic.act(observation)
                # print(f"[DEBUG] Policy output - value: {value}, output_id: {output_id}, action: {action}, tokens_log_probs: {tokens_log_probs}, text_action: {text_action}")

                # print("[DEBUG] Getting reference model logits")
                reference_log_probs = self.actor_critic.get_reference_model_logits(observation, output_id)
                # print(f"[DEBUG] Reference log probs: {reference_log_probs}")

                if step % 10 == 0:
                    # print(f"[DEBUG] Action text at step {step}: {text_action}")
                    pass

                # Take step in environment
                if action is None:
                    # print(f"[WARNING] Invalid action detected at step {step} (text_action: {text_action}), setting action to 0")
                    action = 0  # Default action

                env_text = text_action[0] if isinstance(text_action, list) else text_action
                # print(f"[DEBUG] Stepping environment with action: {env_text}")

                observation, reward, done, info, env_action = self.env.step(env_text)
                # print(f"[DEBUG] Step output - observation: {observation}, reward: {reward}, done: {done}, info: {info}, env_action: {env_action}")

                # Process reward
                reward = self._process_reward(reward, done, info)
                # print(f"[DEBUG] Processed reward: {reward}")

                # Handle episode end
                if done:
                    # print(f"[DEBUG] Episode finished at step {step}")
                    pass

                # Create masks
                masks = torch.FloatTensor([[0.0] if done else [1.0]])
                bad_masks = torch.FloatTensor([[0.0] if info.get('TimeLimit.truncated', False) else [1.0]])
                # print(f"[DEBUG] Masks - masks: {masks}, bad_masks: {bad_masks}")

                # Track rewards
                self.running_episode_rewards += reward.flatten()
                # print(f"[DEBUG] Updated running_episode_rewards: {self.running_episode_rewards}")
                if done:
                    final_reward = self.running_episode_rewards[0].item()
                    self.episode_rewards.append(final_reward)
                    success = 1 if self.running_episode_rewards[0] > 0 else 0
                    self.episode_success_rate.append(success)
                    # print(f"[DEBUG] Episode rewards appended: {final_reward}, success: {success}")
                    self.running_episode_rewards[0] = 0

                # Store transition
                try:
                    env_action_used = env_action if env_action is not None else action
                    # print(f"[DEBUG] Storing transition with env_action_used: {env_action_used}")
                    self.rollouts.insert(
                        observation, output_id, torch.tensor([env_action_used]),
                        tokens_log_probs, reference_log_probs, value, reward, masks, bad_masks
                    )
                    # print("[DEBUG] Transition stored in rollouts")
                except Exception as e:
                    print(f"[ERROR] Exception during rollouts.insert: {e}")

            # Update policy
            print(f"****** Iteration {j} ******")
            print(f"Rewards: {list(self.episode_rewards)}")
            
            next_value = self.actor_critic.get_value(self.rollouts.obs[-1]).detach()
            
            self.rollouts.compute_returns(
                next_value,
                self.config.use_gae,
                self.config.gamma,
                self.config.gae_lambda,
                self.config.use_proper_time_limits
            )
            
            # Determine training mode
            only_value_loss = self.config.value_warmup == "yes" and j < 2
            use_kl = self.config.use_kl == "yes"
            
            value_loss, action_loss, dist_entropy, value_losses, action_losses, kls = self.agent.update(
                self.rollouts, only_value_loss=only_value_loss, kl=use_kl
            )
            
            self.lr_scheduler.step()
            self.rollouts.after_update(j)
            
            # Save checkpoint
            if self.accelerator.is_main_process and self.config.save_interval and (j + 1) % self.config.save_interval == 0:
                self._save_checkpoint(j)
            
            # Log metrics
            if len(self.episode_rewards) > 1:
                self._log_metrics(j, start_time, value_loss, action_loss, dist_entropy, kls)
        
        return self._get_final_metrics()
    
    def _process_reward(self, reward: float, done: bool, info: Dict) -> torch.Tensor:
        """Process reward into tensor."""
        if isinstance(reward, np.ndarray):
            reward = torch.from_numpy(reward)
        elif isinstance(reward, list):
            reward = torch.Tensor([reward[0]])
        else:
            reward = torch.Tensor([reward])
        return reward
    
    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint."""
        current_save_path = os.path.join(self.config.save_path, f"update_{iteration}")
        os.makedirs(current_save_path, exist_ok=True)
        
        run_name = self.config.wandb_run if self.config.wandb_run else "model"
        
        # Save base model
        self.agent.actor_critic.value_model.base_model.save_pretrained(
            os.path.join(current_save_path, f"value_head_{run_name}")
        )
        
        # Save value head
        torch.save(
            self.agent.actor_critic.value_model.value_head.state_dict(),
            os.path.join(current_save_path, f"value_head_{run_name}.pth")
        )
        
        print(f"Checkpoint saved: {current_save_path}")
    
    def _log_metrics(
        self,
        iteration: int,
        start_time: float,
        value_loss: float,
        action_loss: float,
        dist_entropy: float,
        kls: List[float]
    ):
        """Log training metrics."""
        total_num_steps = (iteration + 1) * self.config.num_processes * self.config.num_steps
        end_time = time.time()
        fps = int(total_num_steps / (end_time - start_time))
        
        print(
            f"Updates {iteration}, timesteps {total_num_steps}, FPS {fps}\n"
            f"Last {len(self.episode_rewards)} episodes: "
            f"mean/median reward {np.mean(self.episode_rewards):.2f}/{np.median(self.episode_rewards):.2f}, "
            f"min/max {np.min(self.episode_rewards):.2f}/{np.max(self.episode_rewards):.2f}, "
            f"success rate {np.mean(self.episode_success_rate):.2f}"
        )
        
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "iteration": iteration,
                "num_timesteps": total_num_steps,
                "FPS": fps,
                "episode_reward.mean": np.mean(self.episode_rewards),
                "episode_reward.median": np.median(self.episode_rewards),
                "episode_reward.min": np.min(self.episode_rewards),
                "episode_reward.max": np.max(self.episode_rewards),
                "episode_success_rate.mean": np.mean(self.episode_success_rate),
                "distribution_entropy": dist_entropy,
                "value.loss": value_loss,
                "action.loss": action_loss,
                "reward.max": self.rollouts.rewards.max().item(),
                "reward.min": self.rollouts.rewards.min().item(),
                "reward.mean": self.rollouts.rewards.mean().item(),
                "return.max": self.rollouts.returns.max().item(),
                "return.min": self.rollouts.returns.min().item(),
                "return.mean": self.rollouts.returns.mean().item(),
                "value.max": self.rollouts.value_preds.max().item(),
                "value.min": self.rollouts.value_preds.min().item(),
                "value.mean": self.rollouts.value_preds.mean().item(),
                "kl": np.array(kls).mean() if kls else 0,
            })
    
    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        return {
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "success_rate": np.mean(self.episode_success_rate) if self.episode_success_rate else 0,
        }
    
    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()


class TrainerConfig:
    """
    Configuration class for VLMTrainer.
    Can be initialized from arguments or a YAML config file.
    """
    
    def __init__(
        self,
        # Model config
        model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        cache_dir: Optional[str] = None,
        use_peft: bool = False,
        
        # Environment config
        env_name: str = "MiniWorld-OneRoom-v0",
        max_episode_steps: int = 128,
        max_image_obs_len: int = 4,
        prompt_version: str = "v1",
        
        # Training config
        seed: int = 1,
        num_processes: int = 1,
        num_steps: int = 256,
        num_env_steps: int = int(10e6),
        
        # PPO config
        ppo_epoch: int = 4,
        mini_batch_size: int = 1,
        clip_param: float = 0.1,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        kl_beta: float = 0.04,
        gamma: float = 0.9,
        gae_lambda: float = 0.95,
        use_gae: bool = False,
        use_proper_time_limits: bool = False,
        
        # Optimizer config
        init_lr: float = 1e-6,
        end_lr: float = 1e-8,
        weight_decay: float = 0,
        eps: float = 1e-7,
        lr_max_steps: int = 100,
        max_grad_norm: float = 0.01,
        grad_accum_steps: int = 2,
        
        # Generation config
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        thought_prob_coef: float = 1.0,
        
        # Training modes
        value_warmup: str = "yes",
        use_kl: str = "yes",
        stop_grad: str = "yes",
        
        # Logging config
        save_path: str = "./runs",
        log_dir: str = "./runs",
        save_interval: int = 100,
        eval_num_per_episode: int = 100,
        use_wandb: bool = False,
        wandb_project: str = "test",
        wandb_run: str = "test",
        debug: bool = False,
        
        **kwargs
    ):
        # Model
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.use_peft = use_peft
        
        # Environment
        self.env_name = env_name
        self.max_episode_steps = max_episode_steps
        self.max_image_obs_len = max_image_obs_len
        self.prompt_version = prompt_version
        
        # Training
        self.seed = seed
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.num_env_steps = num_env_steps
        
        # PPO
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.kl_beta = kl_beta
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        self.use_proper_time_limits = use_proper_time_limits
        
        # Optimizer
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.lr_max_steps = lr_max_steps
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = grad_accum_steps
        
        # Generation
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.thought_prob_coef = thought_prob_coef
        
        # Training modes
        self.value_warmup = value_warmup
        self.use_kl = use_kl
        self.stop_grad = stop_grad
        
        # Logging
        self.save_path = save_path
        self.log_dir = log_dir
        self.save_interval = save_interval
        self.eval_num_per_episode = eval_num_per_episode
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run = wandb_run
        self.debug = debug
        
        # Store extra kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainerConfig":
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args) -> "TrainerConfig":
        """Create config from argparse namespace."""
        return cls(**vars(args))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return vars(self)
    
    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

