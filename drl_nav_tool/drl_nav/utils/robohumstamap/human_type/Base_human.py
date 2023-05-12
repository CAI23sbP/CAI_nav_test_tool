import os

class BaseHuman():
    def __init__(self, namespace):
        self._namespace = namespace
        self._ns_prefix = lambda *topic: os.path.join(self._namespace, *topic)
    
    def reset_all_human(self):
        """
        echo to flatland reset all human
        """
        raise NotImplementedError()

    def spawn_human_agents(self, **dynamic_obstacles):
        """
        spawn human in flatland
        """
        raise NotImplementedError()
    
    
    def remove_all_humans(self):
        """
        remove all human in flatland
        """
        raise NotImplementedError()
    

    def _delete_human(self):
        """
        remove_all_human is need this part
        remove one by one in flatland 

        """
        raise NotImplementedError()