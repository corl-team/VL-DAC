"""
Configuration management for VLM RL training.
Supports YAML configs with nested structure and command-line overrides.
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten nested dictionary with separator."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load and flatten YAML configuration."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested structure for easy access
    flat_config = {}
    
    # Model config
    if 'model' in config:
        flat_config['model_path'] = config['model'].get('path')
        flat_config['cache_dir'] = config['model'].get('cache_dir')
        flat_config['use_peft'] = config['model'].get('use_peft', False)
        if 'peft_config' in config['model']:
            flat_config['peft_config'] = config['model']['peft_config']
    
    # Environment config
    if 'environment' in config:
        flat_config['env_name'] = config['environment'].get('name')
        flat_config['max_episode_steps'] = config['environment'].get('max_episode_steps', 128)
        flat_config['max_image_obs_len'] = config['environment'].get('max_image_obs_len', 4)
        flat_config['prompt_version'] = config['environment'].get('prompt_version', 'v1')
        if 'config_path' in config['environment']:
            flat_config['alf_config'] = config['environment']['config_path']
    
    # Training config
    if 'training' in config:
        flat_config['seed'] = config['training'].get('seed', 1)
        flat_config['num_processes'] = config['training'].get('num_processes', 1)
        flat_config['num_steps'] = config['training'].get('num_steps', 256)
        flat_config['num_env_steps'] = config['training'].get('num_env_steps', int(10e6))
    
    # PPO config
    if 'ppo' in config:
        flat_config['ppo_epoch'] = config['ppo'].get('ppo_epoch', 4)
        flat_config['mini_batch_size'] = config['ppo'].get('mini_batch_size', 1)
        flat_config['clip_param'] = config['ppo'].get('clip_param', 0.1)
        flat_config['entropy_coef'] = config['ppo'].get('entropy_coef', 0.01)
        flat_config['value_loss_coef'] = config['ppo'].get('value_loss_coef', 0.5)
        flat_config['kl_beta'] = config['ppo'].get('kl_beta', 0.04)
        flat_config['gamma'] = config['ppo'].get('gamma', 0.9)
        flat_config['gae_lambda'] = config['ppo'].get('gae_lambda', 0.95)
        flat_config['use_gae'] = config['ppo'].get('use_gae', False)
        flat_config['use_proper_time_limits'] = config['ppo'].get('use_proper_time_limits', False)
    
    # Optimizer config
    if 'optimizer' in config:
        flat_config['init_lr'] = config['optimizer'].get('init_lr', 1e-6)
        flat_config['end_lr'] = config['optimizer'].get('end_lr', 1e-8)
        flat_config['weight_decay'] = config['optimizer'].get('weight_decay', 0)
        flat_config['eps'] = config['optimizer'].get('eps', 1e-7)
        flat_config['lr_max_steps'] = config['optimizer'].get('lr_max_steps', 100)
        flat_config['max_grad_norm'] = config['optimizer'].get('max_grad_norm', 0.01)
        flat_config['grad_accum_steps'] = config['optimizer'].get('grad_accum_steps', 2)
    
    # Generation config
    if 'generation' in config:
        flat_config['max_new_tokens'] = config['generation'].get('max_new_tokens', 128)
        flat_config['temperature'] = config['generation'].get('temperature', 0.2)
        flat_config['thought_prob_coef'] = config['generation'].get('thought_prob_coef', 1.0)
    
    # Training modes
    if 'modes' in config:
        flat_config['value_warmup'] = config['modes'].get('value_warmup', 'yes')
        flat_config['use_kl'] = config['modes'].get('use_kl', 'yes')
        flat_config['stop_grad'] = config['modes'].get('stop_grad', 'yes')
    
    # Logging config
    if 'logging' in config:
        flat_config['save_path'] = config['logging'].get('save_path', './runs')
        flat_config['log_dir'] = config['logging'].get('log_dir', './runs')
        flat_config['save_interval'] = config['logging'].get('save_interval', 100)
        flat_config['eval_num_per_episode'] = config['logging'].get('eval_num_per_episode', 100)
        flat_config['use_wandb'] = config['logging'].get('use_wandb', False)
        flat_config['wandb_project'] = config['logging'].get('wandb_project', 'test')
        flat_config['wandb_run'] = config['logging'].get('wandb_run', 'test')
        flat_config['debug'] = config['logging'].get('debug', False)
    
    return flat_config


def get_args():
    """Get command line arguments with optional config file."""
    parser = argparse.ArgumentParser(description="VLM RL Training")
    
    # Config file argument
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default=None, help="Path to VLM model")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for model")
    parser.add_argument("--use-peft", action="store_true", default=False, help="Use PEFT/LoRA")
    
    # Environment arguments  
    parser.add_argument("--env-name", type=str, default="MiniWorld-OneRoom-v0", help="Environment name")
    parser.add_argument("--max-episode-steps", type=int, default=128, help="Max steps per episode")
    parser.add_argument("--max-image-obs-len", type=int, default=4, help="Max image observation history")
    parser.add_argument("--prompt-version", type=str, default="v1", help="Prompt version")
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of parallel processes")
    parser.add_argument("--num-steps", type=int, default=256, help="Steps per update")
    parser.add_argument("--num-env-steps", type=int, default=int(10e6), help="Total environment steps")
    
    # PPO arguments
    parser.add_argument("--ppo-epoch", type=int, default=4, help="PPO epochs")
    parser.add_argument("--mini-batch-size", type=int, default=1, help="Mini-batch size")
    parser.add_argument("--clip-param", type=float, default=0.1, help="PPO clip parameter")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--kl-beta", type=float, default=0.04, help="KL divergence beta")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--use-gae", action="store_true", default=False, help="Use GAE")
    parser.add_argument("--use-proper-time-limits", action="store_true", default=False)
    
    # Optimizer arguments
    parser.add_argument("--init-lr", type=float, default=1e-6, help="Initial learning rate")
    parser.add_argument("--end-lr", type=float, default=1e-8, help="Final learning rate")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--eps", type=float, default=1e-7, help="Adam epsilon")
    parser.add_argument("--lr-max-steps", type=int, default=100, help="LR scheduler max steps")
    parser.add_argument("--max-grad-norm", type=float, default=0.01, help="Max gradient norm")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps")
    
    # Generation arguments
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--thought-prob-coef", type=float, default=1.0, help="Thought probability coefficient")
    
    # Training modes
    parser.add_argument("--value-warmup", type=str, default="yes", help="Value warmup")
    parser.add_argument("--use-kl", type=str, default="yes", help="Use KL divergence")
    parser.add_argument("--stop-grad", type=str, default="yes", help="Stop gradient to value model")
    
    # Logging arguments
    parser.add_argument("--save-path", type=str, default="./runs", help="Save path")
    parser.add_argument("--log-dir", type=str, default="./runs", help="Log directory")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval")
    parser.add_argument("--eval-num-per-episode", type=int, default=100, help="Episodes for evaluation")
    parser.add_argument("--use-wandb", action="store_true", default=False, help="Use wandb logging")
    parser.add_argument("--wandb-project", type=str, default="test", help="Wandb project name")
    parser.add_argument("--wandb-run", type=str, default="test", help="Wandb run name")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
    
    # Legacy arguments for compatibility
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--alf-config", type=str, default=None)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config is not None:
        config_dict = load_yaml_config(args.config)
        
        # Override args with config values (command line takes precedence)
        for key, value in config_dict.items():
            arg_key = key.replace('-', '_')
            if hasattr(args, arg_key):
                # Only use config value if arg was not explicitly set
                if getattr(args, arg_key) is None or getattr(args, arg_key) == parser.get_default(arg_key):
                    setattr(args, arg_key, value)
    
    # Set CUDA flag
    import torch
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args


@dataclass
class Config:
    """
    Unified configuration dataclass for VLM RL training.
    """
    # Model
    model_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    cache_dir: Optional[str] = None
    use_peft: bool = False
    
    # Environment
    env_name: str = "MiniWorld-OneRoom-v0"
    max_episode_steps: int = 128
    max_image_obs_len: int = 4
    prompt_version: str = "v1"
    
    # Training
    seed: int = 1
    num_processes: int = 1
    num_steps: int = 256
    num_env_steps: int = int(10e6)
    
    # PPO
    ppo_epoch: int = 4
    mini_batch_size: int = 1
    clip_param: float = 0.1
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    kl_beta: float = 0.04
    gamma: float = 0.9
    gae_lambda: float = 0.95
    use_gae: bool = False
    use_proper_time_limits: bool = False
    
    # Optimizer
    init_lr: float = 1e-6
    end_lr: float = 1e-8
    weight_decay: float = 0
    eps: float = 1e-7
    lr_max_steps: int = 100
    max_grad_norm: float = 0.01
    grad_accum_steps: int = 2
    
    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.2
    thought_prob_coef: float = 1.0
    
    # Training modes
    value_warmup: str = "yes"
    use_kl: str = "yes"
    stop_grad: str = "yes"
    
    # Logging
    save_path: str = "./runs"
    log_dir: str = "./runs"
    save_interval: int = 100
    eval_num_per_episode: int = 100
    use_wandb: bool = False
    wandb_project: str = "test"
    wandb_run: str = "test"
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from YAML file."""
        flat_config = load_yaml_config(yaml_path)
        return cls(**flat_config)
    
    @classmethod
    def from_args(cls, args) -> "Config":
        """Create config from argparse namespace."""
        return cls(**{k: v for k, v in vars(args).items() if hasattr(cls, k.replace('-', '_'))})

