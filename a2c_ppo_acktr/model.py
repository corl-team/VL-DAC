import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.model_interface import model_evaluate, model_generate, model_evaluate_reference
import torch.nn.init as init

try:
    from qwen_vl_utils import process_vision_info
except:
    pass


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VLMValue(nn.Module):
    """
    Value network for VLMPolicy, utilizing the base model for feature extraction.
    """

    def __init__(self, base):
        super(VLMValue, self).__init__()
        self.base_model = base
        # Define a more complex value head for improved performance
        if "text_config" in self.base_model.config:
            output_dim = self.base_model.config.text_config.hidden_size
        else:
            output_dim = self.base_model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(output_dim, 3072),  # First layer
            nn.ReLU(),  # Non-linearity
            nn.Linear(3072, 2048),  # Second layer
            nn.ReLU(),  # Non-linearity
            nn.Linear(2048, 1),  # Output layer
        ).to(
            base.device, dtype=torch.float16
        )  # Move to specified device with dtype

    def forward(self, **inputs):
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        outputs = self.base_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        values = self.value_head(hidden_states[-1][:, -1])
        return values


class VLMPolicy(nn.Module):
    def __init__(self, processor, value_model, reference_model, args, projection_f, accelerator, base_kwargs=None):
        """
        Policy network for VLM, handling action selection and evaluation.

        Args:
            tokenizer: Tokenizer for processing text inputs.
            image_processor: Processor for handling image inputs.
            value_model: Instance of VLMValue for value estimation.
            args: Configuration arguments.
            INPUT_IDS: Initial input IDs for the model.
            projection_f: Function to project text actions to discrete actions.
            base_kwargs: Additional keyword arguments for the base model.
        """
        super(VLMPolicy, self).__init__()
        self.args = args
        self.processor = processor
        self.value_model = value_model
        self.reference_model = reference_model

        self.base = value_model.base_model
        self.projection_f = projection_f
        self.accelerator = accelerator

    def process_obs(self, conversation):
        """
        Process observations using the image processor to extract necessary tensors.

        Args:
            obs: Raw observations from the environment.

        Returns:
            A dictionary containing processed tensors like pixel_values, image_grid_thw, etc.
        """
        if "Qwen" in self.args.model_path:
            image_inputs, video_inputs = process_vision_info(conversation)
            text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        elif "gemma" in self.args.model_path:
            inputs = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.base.device, dtype=torch.bfloat16)
        else:
            raise ValueError(f"Model path {self.args.model_path} not supported")
        return inputs

    def act(self, conversation):
        """
        Select an action based on the current inputs.

        Args:
            inputs: Raw observation inputs.
            deterministic: Whether to use deterministic action selection.
            INPUT_IDS: Optional input IDs to override the default.

        Returns:
            Tuple containing value estimates, output token IDs, selected action, and log probabilities.
        """
        inputs = self.process_obs(conversation)

        outputs = model_generate(
            **inputs,
            value_model=self.value_model,
            processor=self.processor,
            args=self.args
        )
        values, output_ids, text_action, tokens_log_probs = (
            outputs
        )

        action = self.projection_f(text_action)
        return values, output_ids, action, tokens_log_probs, text_action

    def get_reference_model_logits(self, conversation, output_ids):

        if self.reference_model is None:
            inputs = self.process_obs(conversation)
            with self.accelerator.unwrap_model(self.base).disable_adapter():
                    outputs = model_evaluate_reference(
                        **inputs,
                        output_ids=output_ids,
                        model=self.base,
                        processor=self.processor,
                        temperature=self.args.temperature,
                        thought_prob_coef=self.args.thought_prob_coef,
                    )
                    logits = outputs
        else:
            inputs = self.process_obs(conversation)
            outputs = model_evaluate_reference(
                **inputs,
                output_ids=output_ids,
                base_model=self.reference_model,
                processor=self.processor,
                temperature=self.args.temperature,
                thought_prob_coef=self.args.thought_prob_coef,
            )
            logits = outputs
        return logits

    def get_value(self, conversation):
        """
        Get the value estimate for the given inputs.

        Args:
            inputs: Raw observation inputs.
            INPUT_IDS: Optional input IDs to override the default.

        Returns:
            Value estimates from the value model.
        """
        inputs = self.process_obs(conversation)
        return self.value_model(**inputs)

    def evaluate_actions(self, conversation, output_ids):
        """
        Evaluate the log probabilities and value estimates for given actions.

        Args:
            inputs: Raw observation inputs.
            output_ids: Output token IDs corresponding to the actions.
            INPUT_IDS: Optional input IDs to override the default.

        Returns:
            Tuple containing value estimates and action log probabilities.
        """
        inputs = self.process_obs(conversation)
        outputs = model_evaluate(
            **inputs,
            output_ids=output_ids,
            value_model=self.value_model,
            processor=self.processor,
            temperature=self.args.temperature,
            thought_prob_coef=self.args.thought_prob_coef,
            mode="train"
        )
        value, tokens_log_probs = outputs
        return value, tokens_log_probs
