import os

class BaseStatic():
    def __init__(self, namespace):
        self._namespace = namespace
        self._ns_prefix = lambda *topic: os.path.join(self._namespace, *topic)
    
    def reset_all_static(self):
        """
        echo to flatland reset all static
        """
        raise NotImplementedError()

    def spawn_static_obstacles(self, **obstacles):
        """
        spawn static in flatland
        """
        raise NotImplementedError()
    
    
    def remove_all_static_obstacles(self):
        """
        remove all static in flatland
        """
        raise NotImplementedError()
    

    def _delete_static_obstacles(self):
        """
        remove_all_static is need this part
        remove one by one in flatland 

        """
        raise NotImplementedError()