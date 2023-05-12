from typing import Callable, Type, Union
# import pickle

class SimulatorTree:
    registry = {}
    # REGISTRY_FILE = "registry.pkl"

    @classmethod
    def register(cls, name) -> Callable:
        def check_exist(wrapped_sim_class)-> Callable:
            assert name not in cls.registry, f"Simulator '{name}' already exists!"
            issubclass(wrapped_sim_class, object) , f"Wrapped class {wrapped_sim_class.__name__} is not of type 'nn.Module'!"
            cls.registry[name] = wrapped_sim_class
            
            return wrapped_sim_class
        
        return check_exist

    @classmethod
    def instantiate(
        cls, name: str ):
        
        assert name in cls.registry, f"Simulator '{name}' is not registered!"
        sim_class = cls.registry[name]
        
        return sim_class
