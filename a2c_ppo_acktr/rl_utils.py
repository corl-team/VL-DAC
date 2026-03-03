import torch
import random
from typing import List
import numpy as np

env_to_task = {
    "MiniWorld-CollectHealth-v0": "to collect health kits and stay alive as long as possible",
    "MiniWorld-FourRooms-v0": {"v1": "to go to a red box in four rooms within as few steps as possible", "v2": "to go to a red box in four rooms within as few steps as possible", "v3": "to go to red box"},
    "MiniWorld-Hallway-v0": {"v1": "to go to a red box at the end of a hallway within as few steps as possible", "v2": "to go to a red box at the end of a hallway within as few steps as possible", "v3": "to go to red box"},
    "MiniWorld-Maze-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-MazeS2-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-MazeS3-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-MazeS3Fast-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-OneRoom-v0": {"v1": "to go to a red box randomly placed in one big room", "v2": "to go to a red box randomly placed in one big room", "v3": "to go to red box"},
    "MiniWorld-OneRoomS6-v0": "to go to a red box placed randomly in one big room",
    "MiniWorld-OneRoomS6Fast-v0": "to go to a red box placed randomly in one big room",
    "MiniWorld-PickupObjects-v0": "to collect as many objects as possible",
    "MiniWorld-PutNext-v0": "to put a red box next to a yellow box",
    "MiniWorld-RoomObjects-v0": "to collect as many objects as possible",  # TODO: check this
    "MiniWorld-Sidewalk-v0": "to walk on a sidewalk up to an object to be collected. Don't walk into the street. The goal is to reach the object in as few steps as possible",
    "MiniWorld-Sign-v0": "to read the sign and follow the instructions",  # TODO: check this
    "MiniWorld-TMaze-v0": "to reach the red box within as few steps as possible.",
    "MiniWorld-TMazeLeft-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-TMazeRight-v0": "to reach the red box within as few steps as possible",
    "MiniWorld-ThreeRooms-v0": "to reach the red box within as few steps as possible",  # TODO: check this
    "MiniWorld-WallGap-v0": {"v1": "to go to a red box behind a wall within as little steps as possible", "v2": "to go to a red box behind a wall within as little steps as possible", "v3": "to go to red box"},
    "MiniWorld-YMaze-v0": "to go to a red box within as little steps as possible",
    "MiniWorld-YMazeLeft-v0": "to go to a red box within as little steps as possible",
    "MiniWorld-YMazeRight-v0": "to go to a red box within as little steps as possible",
}


# Define the function that processes the list of strings according to the specified rules
def text_projection(text_actions: List[str], env_name):
    try:
        action = eval(text_actions[0]).get("action")
        if "miniworld" in env_name.lower():
            if isinstance(action, str):
                print(f"Current action is string: {action}")
                action = int(action)
        if env_name == 'gym_cards/NumberLine-v0':
            action = {"+": 1, "-": 0}[action]
        elif env_name == 'gym_cards/EZPoints-v0':
            action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
            if isinstance(action, int):
                action = str(action)
            action = action_list.index(action)
            if action == -1:
                action = np.random.choice(list(range(12)))
        elif env_name == "gym_cards/Blackjack-v0":
            action_list = ["stand", "hit"]
            if isinstance(action, int):
                action = str(action)
            action = action_list.index(action)
            if action == -1:
                print(f"ATTENTION: Cannot find correct action, {text_actions[0]}")
                action = np.random.choice(list(range(2)))
        elif env_name == "gym_cards/Points24-v0":
            action_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
            if isinstance(action, int):
                action = str(action)
            action = action_list.index(action)
            if action == -1:
                action = np.random.choice(list(range(16)))
    except Exception as e:
        #print("error on parsing")
        action = np.random.choice(list(range(2)))
        print(f"Cannot find correct action, {text_actions[0]}", e)
    return action


def make_observation(past_images, task_name, infos=None, prompt_version="v1"):
    messages = []

    if task_name == 'gym_cards/NumberLine-v0':
        question = "You are playing a game called number line. You will see a target number and a current number in the image. "
        question = question + "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        question = question + "You can return one of two actions: '-' or '+'.  Also describe current observation and your thoughts."
        question = question + "The output format should be as follows: "
        question = question + """{{"thoughts": <any thougths that will lead you to the goal>, "action": <"+" or "-">}}"""
    elif task_name == 'gym_cards/EZPoints-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        question = "You are an expert card game player. You are observing two cards in the image. "
        question = question + f"You are observing the current formula: {text_formula}."
        question = question + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. You should choose only one symbol from given list. "
        question = question + "The chosen symbol will be appended to the current formula. If formula is empty choose random symbol from given list. "
        question = question + "Note that 'J', 'Q', and 'K' count as '10'. "
        question = question + 'Your goal is to output a formula that evaluates to 12, and each number can only be used once.'
        question = question + 'If the current formula "z" is complete, output "=". '
        question = question + 'Otherwise consider which number or operator should be appended to the current formula to make it equal 12. Return your thoughts and action. Your thoughts must be concise. '
        question = question + 'The output format must be as follows: {"thoughts": <any thoughts that will lead you to the goal>, "action": <your chosen symbol>}'
    elif task_name == 'gym_cards/Blackjack-v0':
        question = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        question = question + "Your response should be a valid json file in the following format: \n{\n "
        question = question + "\"thoughts\": \"{first describe your total points and the dealer's total points then think about which action to choose}\", \n"
        question = question + "\"action\": \"stand\" or \"hit\" \n}"
    elif task_name == 'gym_cards/Points24-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula']).strip()
        except:
            text_formula = ''

        current_choice = "['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']"  if not text_formula[-1:].isdigit() else "['+', '-', '*', '/', '(', ')', '=']"
        question = "You are an expert 24 points card game player. You are observing the four cards in the image. Your goal is to make 24 using points that correspond to given cards. "
        question = question + f"{'You are observing the current formula: ' if text_formula else 'Right now you are on the first step to make a formula'}{text_formula}. Final formula after all steps should be in format x+o+p+q=24, where [x, o, p, q] are cards from observation and pluses might be any possible operations. "
        question = question + f"You can choose between {current_choice} symbols. You should choose only {'one symbol from given list that correspond to one of the cards or operator.' if not text_formula[-1:].isdigit() else 'one symbol corresponding to operator.'} "
        question = question + "The chosen symbol will be appended to the current formula. "
        question = question + "Note that 'J', 'Q', and 'K' count as '10'. "
        question = question + "Your goal is to output a formula that evaluates to 24, and each number (card) can only be used once. "
        question = question + "If the current formula equals 24, output '='. "
        question = question + "Otherwise consider which number or operator should be appended to the current formula to make it equal 24. Your thoughts must be concise. "
        question = question + "Your response must be in the following format: "
        question = question + '{"thoughts": <any thoughts that will lead you to the goal>, "action": <your chosen card by number or operator>}'
        print(question)
    elif "miniworld" in task_name.lower():
        task = env_to_task[task_name] if isinstance(env_to_task[task_name], str) else env_to_task[task_name][prompt_version]
        if prompt_version == "v1":
            question = f"""# Instructions
You are operating in a simulator. Your objective is to complete the task. To complete the task, you need to take actions. Upon completing the task, the simulation will end, and you will receive a reward. If you will not solve the task, you will get reward 0.
TASK: {task}.
Take ONE action based on the current observation. Current observation is {f'the state after {len(past_images)} previous actions' if len(past_images) > 1 else 'starting state'}. If you cannot determine how to solve the task, you may turn around or explore the environment to identify the appropriate action.
# Available actions:
0 : turn left
1 : turn right
2 : move forward
3 : move back

First, describe what you observe on the last state using a text description. Try to understand your position relative to the goal, walls, and other objects. Then, carefully consider which action will help you complete the task. Think step by step to understand the environment. After that, choose only one action. Return current scene description, thoughts, and the chosen action.

# ADDITIONAL INSTRUCTIONS:
- If you're stuck against a wall, try to turn around and explore the environment.
- If you can't see the goal, try to explore the environment.

The output format should be as follows:
{{"description": <description>, "thoughts": <thoughts>, "action": <action_number>}}
    """
        elif prompt_version in ["v2", "v3"]:
            question = f"""# Instructions
You are operating in a simulator. Your objective is to complete the task. To complete the task, you need to take actions. Upon completing the task, the simulation will end, and you will receive a reward. If you will not solve the task, you will get reward 0.
TASK: {task}.
Take ONE action based on the current observation. Current observation is {f'the state after {len(past_images)} previous actions' if len(past_images) > 1 else 'starting state'}. If you cannot determine how to solve the task, you may turn around or explore the environment to identify the appropriate action.
# Available actions:
0 : turn left
1 : turn right
2 : move forward
3 : move back

Think through current observation and choose one action to take.

# ADDITIONAL INSTRUCTIONS:
- If you're stuck against a wall, try to turn around and explore the environment.
- If you can't see the goal, try to explore the environment.

The output format should be as follows:
{{"thoughts": <any thoughts that will lead you to the goal>, "action": <action_number>}}
"""
    # if len(past_images) > 1:
    #     messages.append(
    #         {
    #             "type": "video",
    #             "video": list(past_images),
    #             "fps": 30.0,
    #         }
    #     )
    # elif len(past_images) == 1:
    #     messages.append({"type": "image", "image": list(past_images)[0]})
    if len(past_images):
        for image in past_images:
            messages.append({"type": "image", "image": image})
    else:
        raise ValueError("No video frames found.")
    messages.append({"type": "text", "text": question})

    conversation = [
        {
            "role": "user",
            "content": messages,
        }
    ]
    return conversation
