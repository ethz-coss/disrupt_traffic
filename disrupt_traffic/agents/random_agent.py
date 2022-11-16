from agents.fixed_agent import Fixed_Agent
import random

class Random_Agent(Fixed_Agent):
    """
    The random agent selecting phases randomly
    """
    def __init__(self, env, ID='', **kwargs):
        super().__init__(env, ID)
        self.agents_type = 'random'
    
    def choose_act(self, eng, time):
        """
        selects a random phase
        """
        phaseID = random.randint(1, len(self.phases))
        return phaseID
