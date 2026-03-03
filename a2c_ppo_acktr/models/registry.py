"""
Model registry for dynamic model loading.
"""

from typing import Dict, Type, Optional, Any
from .base import BaseVLMAdapter


class ModelRegistry:
    """
    Registry for VLM model adapters.
    Allows dynamic registration and instantiation of models.
    """
    
    _registry: Dict[str, Type[BaseVLMAdapter]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model adapter class."""
        def decorator(adapter_class: Type[BaseVLMAdapter]):
            cls._registry[name.lower()] = adapter_class
            return adapter_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseVLMAdapter]]:
        """Get model adapter class by name."""
        # Try exact match first
        if name.lower() in cls._registry:
            return cls._registry[name.lower()]
        
        # Try partial match
        for key, adapter_class in cls._registry.items():
            if key in name.lower():
                return adapter_class
        
        return None
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._registry.keys())
    
    @classmethod
    def create(
        cls,
        model_path: str,
        cache_dir: Optional[str] = None,
        device: Optional[Any] = None,
        use_peft: bool = False,
        peft_config: Optional[Dict] = None,
        **kwargs
    ) -> BaseVLMAdapter:
        """Create a model adapter instance."""
        adapter_class = cls.get(model_path)
        
        if adapter_class is None:
            raise ValueError(
                f"Model '{model_path}' not found in registry. "
                f"Available: {cls.list_models()}"
            )
        
        adapter = adapter_class(
            model_path=model_path,
            cache_dir=cache_dir,
            device=device,
            use_peft=use_peft,
            peft_config=peft_config,
            **kwargs
        )
        
        adapter.load_model()
        return adapter


def get_model(
    model_path: str,
    cache_dir: Optional[str] = None,
    device: Optional[Any] = None,
    use_peft: bool = False,
    peft_config: Optional[Dict] = None,
    **kwargs
) -> BaseVLMAdapter:
    """
    Convenience function to get a model adapter by path.
    
    Args:
        model_path: Path or name of the model
        cache_dir: Directory for caching model weights
        device: Device to load model on
        use_peft: Whether to apply PEFT/LoRA
        peft_config: PEFT configuration
        **kwargs: Additional model-specific arguments
    
    Returns:
        BaseVLMAdapter instance
    """
    return ModelRegistry.create(
        model_path=model_path,
        cache_dir=cache_dir,
        device=device,
        use_peft=use_peft,
        peft_config=peft_config,
        **kwargs
    )

