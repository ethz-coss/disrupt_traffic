from agents.fixed_agent import Fixed_Agent
import random

class Random_Agent(Fixed_Agent):
    """
    The random agent selecting phases randomly
    """
    def __init__(self, env, ID='', **kwargs):
        super().__init__(env, ID)
        self.agents_type = 'random'
    
    def act(self, lanes_count):
        """
        selects a random phase
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        phaseID = random.randint(1, len(self.phases))
        return self.phases[phaseID]
