import os
try:
    import miniworld
except ImportError:
    print("miniworld not found")


try:
    import gym_cards
except ImportError:
    print("gym_cards not found. IF YOU TRAINING ON MINIWORLD, IGNORE THIS MESSAGE")


try:
    from web_agent_site.envs import WebAgentSiteEnv
except Exception as e:
    print(e)
    print("WebAgentSiteEnv not found. IF YOU NOT TRAINING ON WEBSHOP, IGNORE THIS MESSAGE")


# import gym
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.box import Box
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_


def make_env(
    env_id,
    seed,
    rank,
    log_dir,
    allow_early_resets,
    use_cnn=False,
    record_video=True,
    max_episode_steps=128,
):
    def _thunk():
        if "gym_cards" in env_id.lower():
            env = gym.make(env_id, max_episode_steps=max_episode_steps)
            print("DEBUGGING")
        elif "miniworld" in env_id.lower():
            env = gym.make(
                env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps
            )
        elif "webshop" in env_id.lower():
            env = WebAgentSiteEnv(
                env_id,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
            )
        
        _ = np.random.get_state()


        # if time_limit is not None:
        #     env._max_episode_steps = time_limit
        #     env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets,
            )
            if record_video and not ("gym_cards" in env_id.lower()):
                # idx = 0
                # while os.path.exists(f"runs/video_{idx}"):
                #     idx += 1
                trigger = lambda t: True
                # env = RecordVideo(
                #     env,
                #     video_folder=os.path.join(log_dir, f"{env_id}-{max_episode_steps}/video_rank_{torch.distributed.get_rank()}"),
                #     episode_trigger=trigger,
                # )
        return env

    return _thunk


def make_vec_envs(
    env_name,
    seed,
    rank,
    gamma,
    log_dir,
    device,
    allow_early_resets,
    num_frame_stack=None,
    use_cnn=False,
    max_episode_steps=128,
):
    env = DummyVecEnv(
        [
            make_env(
                env_name,
                seed,
                rank, 
                log_dir,
                True,
                use_cnn,
                max_episode_steps=max_episode_steps,
            )
        ]
    )
    env.seed(seed + rank)
    return env


# Checks whether done was caused by timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if (
            terminated or truncated
        ) and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]
        self.shape_dim1 = wos.shape[1]

        low = np.repeat(wos.low, self.nstack, axis=1)
        high = np.repeat(wos.high, self.nstack, axis=1)

        if device is None:
            device = torch.device("cpu")
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)
        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :, : -self.shape_dim1] = self.stacked_obs[
            :, :, self.shape_dim1 :
        ].clone()
        for i, new in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, :, -self.shape_dim1 :] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, :, -self.shape_dim1 :] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
