#!/usr/bin/env python3
"""
VLM RL Training Script

This script supports two modes:
1. Legacy mode: Original training loop (default)
2. Modular mode: New modular architecture with YAML configs

Usage:
    # Legacy mode (original behavior)
    python main.py --env-name MiniWorld-OneRoom-v0 --model-path Qwen/Qwen2-VL-7B-Instruct
    
    # Modular mode with config file
    python main.py --modular --config configs/miniworld_qwen2vl.yaml
"""

import copy
import glob
import os
import time
from collections import deque
import threading

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from peft import get_peft_model, LoraConfig, TaskType
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.rl_utils import make_observation, text_projection
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage
from matplotlib import pyplot as plt
import math
import random
from functools import partial
from typing import List, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Gemma3ForConditionalGeneration
import transformers
from dotenv import load_dotenv

from tqdm import tqdm

import accelerate 
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

# S3 upload is optional
HAS_S3 = False
try:
    import boto3
    from botocore.exceptions import ClientError
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_ENDPOINT_URL:
        HAS_S3 = True
except ImportError:
    pass

if not HAS_S3:
    print("S3 upload disabled (missing credentials, endpoint, or boto3)")


def upload_to_s3(local_path, s3_bucket, s3_path, aws_access_key_id=None, aws_secret_access_key=None):
    """Upload a file to S3"""
    if not HAS_S3:
        return
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            endpoint_url=S3_ENDPOINT_URL
        )
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                s3_path = relative_path.replace("\\", "/")
                print(local_file, relative_path)
                s3_path = os.path.join("/".join(local_path.split("/")[1:]), s3_path)
                print(s3_path)
                s3_client.upload_file(local_file, s3_bucket, s3_path)
    except Exception as e:
        print(f"Error uploading to S3: {e}")


def main_legacy():
    """Legacy training loop (original implementation)."""
    args = get_args()

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    set_seed(args.seed, device_specific=True)

    model_device = device
    print(args.seed, accelerator.process_index)

    # Initialization of VLM model
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    model_path = args.model_path
    cache_dir = args.cache_dir

    print(model_path)
    while True:
        try:
            if "Qwen" in model_path:
                processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
                base = Qwen2VLForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)
            elif "gemma" in model_path.lower():
                processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
                base = Gemma3ForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)
            reference_model = None
            break
        except Exception as e:
            print(e)
            print("Model not found, trying again...")
        
    if args.use_peft:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
        base = get_peft_model(base, peft_config)
    
    print("Model max context length:{}".format(base.config.max_length))
    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    # Environment creation
    if "miniworld" in args.env_name.lower():
        env = make_vec_envs(args.env_name, args.seed, accelerator.process_index,
                             args.gamma, os.path.join(args.log_dir, args.env_name), device, False, 1, max_episode_steps=args.max_episode_steps)
    elif "gym_cards" in args.env_name.lower():
        env = make_vec_envs(args.env_name, args.seed, accelerator.process_index,
                             args.gamma, os.path.join(args.log_dir, args.env_name), device, False, 1, max_episode_steps=args.max_episode_steps)
        print("GYM CARDS SUPPORTED ONLY FOR DEBUG REASON")
    elif "webshop" in args.env_name.lower():
        env = make_vec_envs(args.env_name, args.seed, accelerator.process_index,
                             args.gamma, os.path.join(args.log_dir, args.env_name), device, False, 1, max_episode_steps=args.max_episode_steps)
    elif "alfred" in args.env_name.lower() or "alfworld" in args.env_name.lower():
        env = make_vec_envs(args.env_name, args.seed, accelerator.process_index,
                             args.gamma, os.path.join(args.log_dir, args.env_name), device, False, 1, max_episode_steps=args.max_episode_steps)
    else:
        print(f"Environment {args.env_name} not supported")
        print("Supported: miniworld, gym_cards, webshop, alfworld")
        exit(1)

    projection_f = partial(text_projection, env_name=args.env_name)
    
    actor_critic = VLMPolicy(
        accelerator=accelerator,
        processor=processor,
        value_model=value_model,
        reference_model=reference_model,
        projection_f=projection_f,
        args=args
    )
    
    optimizer = optim.Adam([
        {'params': actor_critic.value_model.parameters(), 'lr': args.init_lr},
    ], eps=args.eps, weight_decay=args.weight_decay)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)
    
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    
    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)
    if reference_model is not None:
        reference_model = accelerator.prepare_model(reference_model, evaluation_mode=True)
        actor_critic.reference_model = reference_model
    
    agent = algo.PPO(
        actor_critic,
        optimizer,
        accelerator,
        args.clip_param,
        args.ppo_epoch,
        args.mini_batch_size,
        args.value_loss_coef,
        args.entropy_coef,
        args.kl_beta,
        max_grad_norm=args.max_grad_norm,
        grad_accum_steps=args.grad_accum_steps
    )

    rollouts = RolloutStorage(args.num_steps, 1, env.action_space, args.max_new_tokens, log_path=os.path.join(args.log_dir, args.env_name, "rollouts"))

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    
    if args.use_wandb:
        import wandb
        run_name_prefix = "debug-" if args.debug else ""
        run_name = run_name_prefix + args.wandb_run
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)

    running_episode_rewards = torch.zeros(args.num_processes).flatten()
    image_observations = deque(maxlen=args.max_image_obs_len)
    counter = 0
    infos = []
    
    for j in tqdm(range(num_updates)):
        for step in tqdm(range(args.num_steps), desc="Step on epoch {}, on rank {}".format(j, accelerator.process_index)):
            # Sample actions
            if j == 0 and step == 0:
                if "webshop" in args.env_name.lower():
                    _ = env.reset(seed=args.seed)
                    image_obs = env.observation_space["image"]
                else:       
                    image_obs = utils.image_wrap(env.reset())
                image_observations.append(image_obs)
                observation = make_observation(image_observations, args.env_name, infos, args.prompt_version)
                rollouts.obs[0] = observation
        
            value, output_id, action, tokens_log_probs, text_action = actor_critic.act(observation)
            reference_log_probs = actor_critic.get_reference_model_logits(observation, output_id)
            
            if step % 10 == 0:
                print(text_action)
            
            if action is None:
                print(text_action)
            
            image_obs, reward, done, infos = env.step([action])
            image_obs = utils.image_wrap(image_obs)
            
            if isinstance(reward, np.ndarray):
                reward = torch.from_numpy(reward)
            elif isinstance(reward, list):
                reward = torch.Tensor([reward[0]])
            else:
                reward = torch.Tensor([reward])
            
            for d in done:
                if d:
                    image_observations.clear()
                    counter = 0
                    if isinstance(reward, np.ndarray):
                        reward = torch.from_numpy(reward)
                    elif isinstance(reward, list):
                        reward = torch.Tensor([reward[0]])
                    else:
                        reward = torch.Tensor([reward])
                    if "miniworld" in args.env_name.lower():
                        if reward <= 0:
                            reward = torch.Tensor([-1])
                    print(step, "Episode finished")
            
            for info in infos:
                if info.get('TimeLimit.truncated', 0):
                    image_observations.clear()
                    print(step, "Episode finished")
                    counter = 0
            
            image_observations.append(image_obs)
            observation = make_observation(image_observations, args.env_name, infos, args.prompt_version)

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            
            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    episode_success_rate.append(1 if running_episode_rewards[i] > 0 else 0)
                    running_episode_rewards[i] = 0

            bad_masks = torch.FloatTensor([[0.0] if info.get('TimeLimit.truncated', 0) else [1.0] for info in infos])
            rollouts.insert(observation, output_id, action, tokens_log_probs, reference_log_probs, value, reward, masks, bad_masks)

            counter += 1

        print("****** iteration number:{} ******".format(j))
        print("reward:{}".format(episode_rewards))
        print()
        
        next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        print(rollouts.obs[-1], next_value)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        
        if args.value_warmup == "yes":
            only_value_loss = True if j < 2 else False
        else:
            only_value_loss = False

        if args.use_kl == "yes":
            kl = True
        else:
            kl = False
        
        value_loss, action_loss, dist_entropy, value_losses, action_losses, kls = agent.update(rollouts, only_value_loss=only_value_loss, kl=kl)
        lr_scheduler.step()
        rollouts.after_update(j)

        if accelerator.is_main_process and args.save_interval and (j + 1) % args.save_interval == 0:
            current_save_path = os.path.join(args.save_path, "update_{}".format(j))
            os.makedirs(current_save_path, exist_ok=True)
            run_name = args.wandb_run if args.wandb_run else "model"
            agent.actor_critic.value_model.base_model.save_pretrained(os.path.join(current_save_path, f"value_head_{run_name}"))
            torch.save(agent.actor_critic.value_model.value_head.state_dict(), os.path.join(current_save_path, f"value_head_{run_name}.pth"))
            
            # Upload checkpoint to S3 asynchronously
            s3_bucket = "trs-gbredis"
            s3_path = ""
            upload_to_s3(current_save_path, s3_bucket, s3_path)

        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate)))
            
            if args.use_wandb:
                import wandb
                wandb.log({
                    "iteration": j,
                    "num_timesteps": total_num_steps,
                    "FPS": int(total_num_steps / (end - start)),
                    "episode_reward.mean": np.mean(episode_rewards),
                    "episode_reward.median": np.median(episode_rewards),
                    "episode_reward.min": np.min(episode_rewards),
                    "episode_reward.max": np.max(episode_rewards),
                    "episode_success_rate.mean": np.mean(episode_success_rate),
                    "distribution_entropy": dist_entropy,
                    "value.loss": value_loss,
                    "action.loss": action_loss,
                    "reward.max": rollouts.rewards.max().item(),
                    "reward.min": rollouts.rewards.min().item(),
                    "reward.mean": rollouts.rewards.mean().item(),
                    "reward.std": rollouts.rewards.std().item(),
                    "reward.median": rollouts.rewards.median().item(),
                    "return.max": rollouts.returns.max().item(),
                    "return.min": rollouts.returns.min().item(),
                    "return.mean": rollouts.returns.mean().item(),
                    "return.std": rollouts.returns.std().item(),
                    "value.max": rollouts.value_preds.max().item(),
                    "value.min": rollouts.value_preds.min().item(),
                    "value.mean": rollouts.value_preds.mean().item(),
                    "value.std": rollouts.value_preds.std().item(),
                    "kl": np.array(kls).mean()
                })


def main_modular():
    """Modular training using the new architecture."""
    from a2c_ppo_acktr.config import get_args as get_config_args
    from a2c_ppo_acktr.trainer import VLMTrainer, TrainerConfig
    
    args = get_config_args()
    config = TrainerConfig.from_args(args)
    
    print("=" * 60)
    print("VLM RL Training (Modular Mode)")
    print("=" * 60)
    print(f"Environment: {config.env_name}")
    print(f"Model: {config.model_path}")
    print(f"Seed: {config.seed}")
    print("=" * 60)
    
    trainer = VLMTrainer(config)
    
    try:
        metrics = trainer.train()
        print(f"\nTraining Complete!")
        print(f"Final mean reward: {metrics['mean_reward']:.4f}")
        print(f"Final success rate: {metrics['success_rate']:.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        trainer.close()


def main():
    """Main entry point - selects between legacy and modular mode."""
    import sys
    
    # Check if modular mode is requested
    if "--modular" in sys.argv:
        sys.argv.remove("--modular")
        main_modular()
    else:
        main_legacy()


if __name__ == "__main__":
    main()
