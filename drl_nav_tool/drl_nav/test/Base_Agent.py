

class BaseAgent():
    def __init__(self):
        pass
    
    def cbGlobalGoal(self, msg):
        raise NotImplementedError()

    def cbPose(self,msg):
        raise NotImplementedError()

    def cbObserv(self,msg):
        raise NotImplementedError()

    def goalReached(self):
        raise NotImplementedError()
        
    def stop_moving(self):
        raise NotImplementedError()

    def update_action(self, action):
        raise NotImplementedError()

    def cbControl(self, event):
        raise NotImplementedError()
    
    def performAction(self, action):
        raise NotImplementedError()

    def cbComputeActionArena(self, event):
        raise NotImplementedError()
        