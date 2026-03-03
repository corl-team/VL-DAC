import os
import yaml
import torchvision.transforms as T
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
import gymnasium as gym
from gymnasium import spaces
import alfworld.agents.environment as environment
from PIL import Image
from typing import Optional
import numpy as np
import torch
import random


ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    current_frames = env.get_frames()
    images = [frame for frame in current_frames]
    print(len(images), images[0].size, type(images[0]))
    return images

class AlfEnv(gym.Env):
    def __init__(self, config_file):
        config = load_config_file(config_file)
        env_type = config['env']['type']
        env = getattr(environment, env_type)(config, train_eval='train')
        self.env = env.init_env(batch_size=1)
        self.action_space = spaces.Discrete(len(ALF_ACTION_LIST))
        self.observation_space = spaces.Box(low=0, high=255, shape=(300, 300, 3), dtype=np.uint8)
        # Add the previous admissible commands for step
        self.prev_admissible_commands = None
        self.num_envs = 1
    def step(self, action):
        ## SZ.3.4: sanity checking legal action as rewards
        action, legal_action = process_action(self.env, action, self.prev_admissible_commands)
        obs, scores, dones, infos = self.env.step(action)
        infos['observation_text'] = obs
        reward = compute_reward(infos, legal_action)
        self.prev_admissible_commands = list(infos['admissible_commands'])[0]
        return self._get_obs(), reward, dones, infos

    def reset(
        self,
        seed=42,
    ):
        self.env.seed(seed)
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        self.prev_admissible_commands = list(infos['admissible_commands'])[0]
        return self._get_obs(), infos

    def _get_obs(self):
        image = get_obs_image(self.env)
        return image

def process_action(env, action=None, action_list=None):
    """
    An function to process the action
    env: the environment should be of type AlfredThorEnv
    action: the list of action to be processeed, it is a list of strings.
    """
    if type(env) != AlfEnv and type(env) != AlfredThorEnv:
        pass
    else:
        legal_action = False
        for i in range(len(action)):
            action[i] = action[i].lower()
            # TODO: need to figure this out
            if len(action[i]) == 0:
                print("Action is empty!!!!")
                # randomly choose an action from the action list if illegal
                action[i] = action_list[random.randint(0, len(action_list)-1)]
            else:
                try:
                    action_index = action[i].find('"action":')
                    # string has the following format '"action": "look"\n}'
                    if action_index == -1:
                        # if we cannot find "action":, then we pick the last 30 characters
                        string = action[i][-30:]
                    else:
                        string = action[i][action_index:]
                    # post processing by removing the first and last part of the string
                    for act in action_list:
                        if act in string:
                            action[i] = act
                            # if found legal action, set legal_action = True
                            legal_action = True
                            break
                except:
                    # randomly choose an action from the action list if illegal
                    action[i] = action_list[random.randint(0, len(action_list)-1)]

    return action, legal_action


def compute_reward(infos, legal_action):
    # A function to compute the shaped reward for the alfworld environment
    # infos: the info returned by the environment
    # legal_action: a boolean value to indicate if the action is legal
    ## Tentative rewards: r = success_reward * 10 + goal_conditioned_r - 1*illegal_action
    reward = float(infos['won'][0]) + float(infos['goal_condition_success_rate'][0])
    if not legal_action:
        # adding a reward penalty to illegal actions
        reward -= 1
    reward = [reward]
    return torch.tensor(reward)

def get_encoded_text(observation_text, tokenizer, model):

    encoded_input = tokenizer(observation_text, return_tensors='pt')
    outputs = model(**encoded_input)
    cls_embeddings = outputs.last_hidden_state[:,0,:]

    return cls_embeddings

def get_concat(obs, infos, tokenizer, model, device):
    assert 'observation_text' in infos.keys(), 'observation_text not in infos!'
    obs_text = infos['observation_text']
    obs_text_encode = get_encoded_text(obs_text, tokenizer, model)
    obs_text_encode = obs_text_encode.to(device)
    obs_cat = torch.cat((obs.flatten(start_dim=1), obs_text_encode), dim=1)
    return obs_cat

def get_cards_concat(obs, infos, tokenizer, model, device):
    ## Need to move these codes to a CNN utils or something
    assert 'Formula' in infos[0].keys(), 'Formula not in infos!'
    infos = infos[0]
    formula_list = infos['Formula']
    formula = "".join([str("".join([str(x) for x in formula_list]))])
    obs_text_encode = get_encoded_text(formula, tokenizer, model).to(device)
    obs_cat = torch.cat((obs.flatten(start_dim=1), obs_text_encode), dim=1)
    return obs_cat