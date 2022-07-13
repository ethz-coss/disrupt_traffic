import random
import numpy as np
import torch

from hybrid_agent import Hybrid_Agent


class Denflow_Agent(Hybrid_Agent):

    def __init__(self, eng, ID='', in_roads=[], out_roads=[], n_states=None, lr=None, batch_size=None):
        super().__init__(eng, ID, in_roads, out_roads, n_states, lr, batch_size)
        self.agents_type = 'denflow'
        self.assigned_cluster = None
            
    def step(self, eng, time, lane_vehs, lanes_count, veh_distance, eps, memory, local_net, done):
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        if time % self.action_freq == 0:
            if self.action_type == "reward":
                reward = self.get_reward(lanes_count)
                self.reward = reward
                self.total_rewards += [reward]
                self.reward_count += 1
                reward = torch.tensor([reward], dtype=torch.float)
                next_state = torch.FloatTensor(self.observe(eng, time, lanes_count)).unsqueeze(0)

                memory.add(self.state, self.action.ID, reward, next_state, done)
                self.action_type = "act"

            if self.action_type == "act":
                self.state = np.asarray(self.observe(eng, time, lanes_count))
                
                self.action = self.act(local_net, self.state, time, lanes_count, eps=eps)
                self.green_time = 10

                if self.action != self.phase:
                    self.update_wait_time(time, self.action, self.phase, lanes_count)
                    self.set_phase(eng, self.clearing_phase)
                    self.action_type = "update"
                    self.action_freq = time + self.clearing_time
                    
                else:
                    self.action_type = "reward"
                    self.action_freq = time + self.green_time

            elif self.action_type == "update":
                self.set_phase(eng, self.action)
                self.action_type = "reward"
                self.action_freq = time + self.green_time



    def observe(self, eng, time, lanes_count):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        flow_change, density_change = self.get_density_flow(time, lanes_count)
        observations = [flow_change, density_change]
        
        return observations
