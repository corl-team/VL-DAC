"""
Qwen2VL model adapter implementation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from transformers import GenerationConfig

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False

from .base import BaseVLMAdapter
from .registry import ModelRegistry


@ModelRegistry.register("qwen")
@ModelRegistry.register("qwen2vl")
@ModelRegistry.register("Qwen2-VL")
class Qwen2VLAdapter(BaseVLMAdapter):
    """
    Adapter for Qwen2VL model.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_model(self) -> Tuple[Any, Any]:
        """Load Qwen2VL model and processor."""
        if not HAS_QWEN:
            raise ImportError(
                "Qwen2VL dependencies not installed. "
                "Please install transformers and qwen_vl_utils."
            )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            cache_dir=self.cache_dir
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
        
        if self.use_peft:
            self.model = self.apply_peft(self.model)
        
        if self.device is not None:
            self.model = self.model.to(self.device)
        
        return self.model, self.processor
    
    def process_inputs(self, conversation: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process conversation into Qwen2VL inputs."""
        image_inputs, video_inputs = process_vision_info(conversation)
        
        text_prompt = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        return inputs
    
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        **kwargs
    ) -> Tuple[torch.Tensor, List[str]]:
        """Generate outputs from Qwen2VL."""
        # Move inputs to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.0,
            top_p=0.001,
            top_k=1,
        )
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        
        output_ids = outputs.sequences
        input_len = inputs['input_ids'].size(1)
        generated_ids = output_ids[:, input_len:]
        
        decoded_outputs = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_ids, decoded_outputs
    
    def get_log_probs(
        self,
        inputs: Dict[str, torch.Tensor],
        output_ids: torch.Tensor,
        temperature: float = 1.0,
        mode: str = "eval",
        **kwargs
    ) -> torch.Tensor:
        """Get log probabilities for given output tokens."""
        # Move inputs to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        output_ids = output_ids.to(self.device)
        
        input_ids = inputs['input_ids']
        if output_ids.size(0) != 1:
            input_ids = input_ids.expand(output_ids.size(0), -1)
        
        combined_ids = torch.cat([input_ids, output_ids], dim=1)
        
        # Update attention mask
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            if attention_mask.size(0) == 1 and output_ids.size(0) > 1:
                attention_mask = attention_mask.expand(output_ids.size(0), -1)
            output_mask = torch.ones_like(output_ids, dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([attention_mask, output_mask], dim=1)
        
        forward_inputs = {
            "input_ids": combined_ids,
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "pixel_values_videos": inputs.get("pixel_values_videos"),
            "video_grid_thw": inputs.get("video_grid_thw"),
        }
        
        outputs = self.model(**forward_inputs, output_hidden_states=True)
        logits = outputs.logits
        
        if mode == "eval":
            logits = logits.detach()
        
        scores = logits.to(torch.float32)
        log_probs = torch.nn.functional.log_softmax(scores, dim=-1).to(torch.bfloat16)
        
        output_ids_mask = (output_ids != 0)[:, 1:]
        tokens_log_probs = output_ids_mask * torch.gather(
            log_probs[:, input_ids.size(1):-1], 2, output_ids[:, 1:].unsqueeze(2)
        ).squeeze(2)
        
        return tokens_log_probs
    
    def get_hidden_states(
        self,
        inputs: Dict[str, torch.Tensor],
        output_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Get hidden states from Qwen2VL."""
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        if output_ids is not None:
            output_ids = output_ids.to(self.device)
            input_ids = inputs['input_ids']
            if output_ids.size(0) != 1:
                input_ids = input_ids.expand(output_ids.size(0), -1)
            combined_ids = torch.cat([input_ids, output_ids], dim=1)
            inputs['input_ids'] = combined_ids
        
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]
    
    def evaluate(
        self,
        inputs: Dict[str, torch.Tensor],
        output_ids: torch.Tensor,
        value_model: nn.Module,
        temperature: float = 1.0,
        value_stopgrad: bool = True,
        mode: str = "eval",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate outputs and return values and log probs.
        Combined forward pass for efficiency.
        """
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        output_ids = output_ids.to(self.device)
        
        input_ids = inputs['input_ids']
        if output_ids.size(0) != 1:
            input_ids = input_ids.expand(output_ids.size(0), -1)
        
        combined_ids = torch.cat([input_ids, output_ids], dim=1)
        
        forward_inputs = {
            "input_ids": combined_ids,
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "pixel_values_videos": inputs.get("pixel_values_videos"),
            "video_grid_thw": inputs.get("video_grid_thw"),
        }
        
        outputs = self.model(**forward_inputs, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        if mode == "eval":
            logits = logits.detach()
            hidden_states = hidden_states.detach()
        
        # Compute values
        input_token_len = input_ids.size(1)
        if value_stopgrad:
            hidden_state_to_value = hidden_states[:, input_token_len - 1].detach()
        else:
            hidden_state_to_value = hidden_states[:, input_token_len - 1]
        values = value_model.get_value_from_hidden(hidden_state_to_value)
        
        if mode == "eval":
            values = values.detach()
        
        # Compute log probs
        scores = logits.to(torch.float32)
        log_probs = torch.nn.functional.log_softmax(scores, dim=-1).to(torch.bfloat16)
        
        output_ids_mask = (output_ids != 0)[:, 1:]
        tokens_log_probs = output_ids_mask * torch.gather(
            log_probs[:, input_ids.size(1):-1], 2, output_ids[:, 1:].unsqueeze(2)
        ).squeeze(2)
        
        return values, tokens_log_probs

