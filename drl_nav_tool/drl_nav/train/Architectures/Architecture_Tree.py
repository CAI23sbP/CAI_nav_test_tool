from typing import Callable
import torch.nn as nn

class ArchitectureTree:
    registry = {}

    @classmethod
    def register(cls, name) -> Callable:
        def check_exist(wrapped_archi_class)-> Callable:
            assert name not in cls.registry, f"Architector '{name}' already exists!"
            assert issubclass(wrapped_archi_class, nn.Module) or  issubclass(wrapped_archi_class, object) , f"Wrapped class {wrapped_archi_class.__name__} is not of type 'nn.Module'!"
            cls.registry[name] = wrapped_archi_class
            
            return wrapped_archi_class
        
        return check_exist

    @classmethod
    def instantiate(
        cls, name: str
    ):
        
        assert name in cls.registry, f"Architector '{name}' is not registered!"
        archi_class = cls.registry[name]
        
        return archi_class
