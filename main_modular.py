#!/usr/bin/env python3
"""
Modular VLM RL Training Script

This script provides a flexible training pipeline that supports:
- Multiple environments: MiniWorld, ALFWorld, WebShop, GymCards
- Multiple models: Qwen2VL, Gemma3, LLaVA
- Configuration via YAML files or command-line arguments

Usage:
    # With config file
    python main_modular.py --config configs/miniworld_qwen2vl.yaml
    
    # With command-line arguments
    python main_modular.py --env-name MiniWorld-OneRoom-v0 --model-path Qwen/Qwen2-VL-7B-Instruct
    
    # With config file and overrides
    python main_modular.py --config configs/miniworld_qwen2vl.yaml --use-wandb --seed 42
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2c_ppo_acktr.config import get_args
from a2c_ppo_acktr.trainer import VLMTrainer, TrainerConfig


def main():
    """Main training entry point."""
    # Parse arguments (with optional config file)
    args = get_args()
    
    # Create trainer config from arguments
    config = TrainerConfig.from_args(args)
    
    print("=" * 60)
    print("VLM RL Training")
    print("=" * 60)
    print(f"Environment: {config.env_name}")
    print(f"Model: {config.model_path}")
    print(f"Seed: {config.seed}")
    print(f"Steps per update: {config.num_steps}")
    print(f"Total env steps: {config.num_env_steps}")
    print("=" * 60)
    
    # Create and run trainer
    trainer = VLMTrainer(config)
    
    try:
        metrics = trainer.train()
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Final mean reward: {metrics['mean_reward']:.4f}")
        print(f"Final success rate: {metrics['success_rate']:.4f}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

