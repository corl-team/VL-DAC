"""
Base model adapter class defining the interface for all VLM models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseVLMAdapter(ABC):
    """
    Abstract base class for VLM model adapters.
    Provides a unified interface for different VLM architectures.
    """
    
    def __init__(
        self,
        model_path: str,
        cache_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_peft: bool = False,
        peft_config: Optional[Dict] = None,
        **kwargs
    ):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_peft = use_peft
        self.peft_config = peft_config or {}
        
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self) -> Tuple[Any, Any]:
        """Load and return the model and processor."""
        pass
    
    @abstractmethod
    def process_inputs(self, conversation: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process conversation into model inputs."""
        pass
    
    @abstractmethod
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        **kwargs
    ) -> Tuple[torch.Tensor, List[str]]:
        """Generate outputs from the model."""
        pass
    
    @abstractmethod
    def get_log_probs(
        self,
        inputs: Dict[str, torch.Tensor],
        output_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Get log probabilities for given output tokens."""
        pass
    
    @abstractmethod
    def get_hidden_states(
        self,
        inputs: Dict[str, torch.Tensor],
        output_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Get hidden states from the model."""
        pass
    
    def apply_peft(self, model: nn.Module) -> nn.Module:
        """Apply PEFT (LoRA) to the model."""
        if not self.use_peft:
            return model
        
        from peft import get_peft_model, LoraConfig, TaskType
        
        target_modules = self.peft_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.peft_config.get("r", 16),
            lora_alpha=self.peft_config.get("lora_alpha", 32),
            lora_dropout=self.peft_config.get("lora_dropout", 0.1),
            target_modules=target_modules,
            bias=self.peft_config.get("bias", "none"),
        )
        
        return get_peft_model(model, peft_config)
    
    @property
    def hidden_size(self) -> int:
        """Return hidden size of the model."""
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'text_config'):
                return self.model.config.text_config.hidden_size
            return self.model.config.hidden_size
        raise NotImplementedError("Hidden size not available")
    
    def to(self, device: torch.device) -> "BaseVLMAdapter":
        """Move model to device."""
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self


class VLMValueModel(nn.Module):
    """
    Value model wrapper for VLM that adds a value head.
    """
    
    def __init__(self, base_model: nn.Module, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 3072),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
        ).to(base_model.device, dtype=torch.float16)
    
    def forward(self, **inputs) -> torch.Tensor:
        """Forward pass returning value estimate."""
        inputs = {k: v.to(self.base_model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        outputs = self.base_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states[:, -1])
        return values
    
    def get_value_from_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get value from hidden state tensor."""
        return self.value_head(hidden_state)

