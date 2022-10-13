import cityflow
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from models.mlp import MLP
from models.dqn import ReplayMemory#, optimize_model
from agents.learning_agent import Learning_Agent
from agents.analytical_agent import Analytical_Agent
# from demand_agent import Demand_Agent
# from hybrid_agent import Hybrid_Agent
# from presslight_agent import Presslight_Agent
# from fixed_agent import Fixed_Agent
# from random_agent import Random_Agent
# from denflow_agent import Denflow_Agent

# from policy_agent import DPGN, Policy_Agent
from engine.cityflow.intersection import Lane

class Environment:
    """
    The class Environment represents the environment in which the agents operate in this case it is a city
    consisting of roads, lanes and intersections which are controled by the agents
    """
    def __init__(self, args, ID=0, n_actions=9, n_states=44):
        """
        initialises the environment with the arguments parsed from the user input
        :param args: the arguments input by the user
        :param n_actions: the number of possible actions for the learning agent, corresponds to the number of available phases
        :param n_states: the size of the state space for the learning agent
        """
        self.eng = cityflow.Engine(args.sim_config, thread_num=8)
        self.ID = ID
        
        self.update_freq = args.update_freq      # how often to update the network
        self.batch_size = args.batch_size
        
        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay= args.eps_decay
        self.eps_update = args.eps_update
        
        self.eps = self.eps_start

        self.agents = []

        random.seed(2)

        self.agents_type = args.agents_type
        
        
        agent_ids = [x for x in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(x)]
        for agent_id in agent_ids:
            if self.agents_type == 'analytical':
                new_agent = Analytical_Agent(self.eng, ID=agent_id)
            elif self.agents_type == 'learning':
                new_agent = Learning_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=self.batch_size)
            # elif self.agents_type == 'demand':
            #     new_agent = Demand_Agent(self.eng, ID=agent_id)
            # elif self.agents_type == 'hybrid':
            #     new_agent = Hybrid_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=self.batch_size)
            # elif self.agents_type == 'presslight':
            #     new_agent = Presslight_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=self.batch_size)
            # elif self.agents_type == 'fixed':
            #     new_agent = Fixed_Agent(self.eng, ID=agent_id)
            # elif self.agents_type == 'random':
            #     new_agent = Random_Agent(self.eng, ID=agent_id)
            # elif self.agents_type == 'denflow':
            #     new_agent = Denflow_Agent(self.eng, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=self.batch_size)
            else:
                raise Exception("The specified agent type:", args.agents_type, "is incorrect, choose from: analytical/learning/demand/hybrid/fixed/random")  
            self.agents.append(new_agent)
            
        self.action_freq = 10   #typical update freq for agents

        
        self.n_actions = len(self.agents[0].phases)
        self.n_states = n_states
        
        # if self.agents_type == 'cluster':
        #     self.cluster_models = Cluster_Models(n_states=n_states, n_actions=self.n_actions, lr=args.lr, batch_size=self.batch_size)
        #     # self.cluster_algo = SOStream.sostream.SOStream(alpha=0, min_pts=9, merge_threshold=0.01)
        #     self.cluster_algo = Mfd_Clustering(self.cluster_models)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.load:
            self.local_net = MLP(self.n_states, self.n_actions, seed=2).to(self.device)
            self.local_net.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
            self.local_net.eval()
            
            self.target_net = MLP(self.n_states, self.n_actions, seed=2).to(self.device)
            self.target_net.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
            self.target_net.eval()
        else:
            self.local_net = MLP(self.n_states, self.n_actions, seed=2).to(self.device)
            self.target_net = MLP(self.n_states, self.n_actions, seed=2).to(self.device)

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=args.lr, amsgrad=True)
        self.memory = ReplayMemory(self.n_actions, batch_size=args.batch_size)

        self.mfd_data = []
        self.agent_history = []

        self.lanes = []

        for lane_id in self.eng.get_lane_vehicles().keys():
            self.lanes.append(Lane(self.eng, ID=lane_id))

        self.speeds = []
        self.stops = []
        self.stopped = {}
        
    def step(self, time, done):
        """
        represents a single step of the simulation for the analytical agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode, used for learning, here for interchangability of the two steps
        """
        
        # print(time)

        veh_ids = self.eng.get_vehicles()
        speeds = []
        stops = 0
        
        for veh_id in veh_ids:
            speed = self.eng.get_vehicle_info(veh_id)['speed']
            speeds.append(float(speed))
            
            if float(speed) <= 0.1 and veh_id not in self.stopped.keys():
                self.stopped.update({veh_id : 1})
                stops += 1
            elif float(speed) > 0.1 and veh_id in self.stopped.keys():
                self.stopped.pop(veh_id)

        self.speeds.append(np.mean(speeds))
        self.stops.append(stops)

        
        lane_vehs = self.eng.get_lane_vehicles()
        lanes_count = self.eng.get_lane_vehicle_count()
        
        self.flow = []
        self.density = []
        
        for lane in self.lanes:
            lane.update_flow_data(self.eng, lane_vehs)
        # flow, density = get_mfd_data(time, lanes_count, self.lanes)
        # if flow != None and density != None and flow != [] and density != []:
        #     self.flow += flow
        #     self.density += density
        # if self.flow != [] and self.density !=[]:
        #     self.mfd_data.append((self.density, self.flow))
        

        veh_distance = 0
        if self.agents_type == "hybrid" or self.agents_type == "learning" or self.agents_type == 'cluster' or self.agents_type == 'presslight':
            veh_distance = self.eng.get_vehicle_distance()

        for agent in self.agents:        
            if agent.agents_type == "cluster":
                agent.step(self.eng, time, lane_vehs, lanes_count, veh_distance, self.eps, self.cluster_algo, self.cluster_models, done)
            else:
                agent.step(self.eng, time, lane_vehs, lanes_count, veh_distance, self.eps, self.memory, self.local_net, done)

        if time % self.action_freq == 0: self.eps = max(self.eps-self.eps_decay,self.eps_end)
        # if time % self.eps_update == 0: self.eps = max(self.eps*self.eps_decay,self.eps_end)

        self.eng.next_step()

    def reset(self):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        self.eng.reset(seed=False)

        for agent in self.agents:
            agent.reset_movements()
            agent.total_rewards = []
            agent.action_type = 'act'



def get_mfd_data(time, lanes_count, lanes):
    flow = []
    density = []

    for lane in lanes:
        if time >= 60:
            f = np.sum(lane.arr_vehs_num[time-60: time]) / 60
        else:
            f = np.sum(lane.arr_vehs_num[0: time]) / time
        d = lanes_count[lane.ID] / lane.length
            
        flow.append(f)
        density.append(d)

    return (flow, density)
