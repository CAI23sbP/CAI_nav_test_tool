from typing import Callable

from Base_static import BaseStatic

class StaticTree:
    registry = {}

    @classmethod
    def register(cls, name) -> Callable:
        def check_exist(static_class)-> Callable:
            assert name not in cls.registry, f"Static type '{name}' already exists!"
            assert issubclass(static_class, BaseStatic)

            cls.registry[name] = static_class
            
            return static_class
        
        return check_exist

    @classmethod
    def instantiate(
        cls, name: str
    ):
        
        assert name in cls.registry, f"Static type '{name}' is not registered!"
        static_class = cls.registry[name]
        
        return static_class
