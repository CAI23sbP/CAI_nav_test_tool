from typing import Callable

from Base_human import BaseHuman

class HumanTree:
    registry = {}

    @classmethod
    def register(cls, name) -> Callable:
        def check_exist(human_class)-> Callable:
            assert name not in cls.registry, f"Human type '{name}' already exists!"
            assert issubclass(human_class, BaseHuman)

            cls.registry[name] = human_class
            
            return human_class
        
        return check_exist

    @classmethod
    def instantiate(
        cls, name: str
    ):
        
        assert name in cls.registry, f"Human type '{name}' is not registered!"
        human_class = cls.registry[name]
        
        return human_class
