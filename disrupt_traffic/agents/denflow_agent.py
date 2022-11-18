import numpy as np
import torch

from agents.hybrid_agent import Hybrid_Agent


class Denflow_Agent(Hybrid_Agent):

    def __init__(self, env, ID='', in_roads=[], out_roads=[], n_states=None, lr=None, batch_size=None):
        super().__init__(env, ID, in_roads, out_roads, n_states, lr, batch_size)
        self.agents_type = 'denflow'
        self.assigned_cluster = None

    def observe(self, *args, **kwargs):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        time = self.env.time
        lanes_count = self.env.lanes_count
        flow_change, density_change = self.get_density_flow(time, lanes_count)
        observations = [flow_change, density_change]
        
        return observations