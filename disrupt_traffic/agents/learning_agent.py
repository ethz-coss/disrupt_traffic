import torch
import numpy as np
import random

from agents.agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learning_Agent(Agent):
    """
    The class defining an agent which controls the traffic lights using reinforcement learning approach called PressureLight
    """
    def __init__(self, env, ID='', in_roads=[], out_roads=[], n_states=0, lr=None, batch_size=None):
        """
        initialises the Learning Agent
        :param ID: the unique ID of the agent corresponding to the ID of the intersection it represents 
        :param eng: the cityflow simulation engine
        """
        super().__init__(env, ID)

        self.in_roads = in_roads
        self.out_roads = out_roads

        self.init_phases_vectors(self.env.eng)
        self.n_actions = len(self.phases)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.local_net = DQN(n_states, self.n_actions, seed=2).to(self.device)
        # self.target_net = DQN(n_states, self.n_actions, seed=2).to(self.device)

        # self.optimizer = Adam(self.local_net.parameters(), lr=lr, amsgrad=True)
        # self.memory = ReplayMemory(self.n_actions, batch_size=batch_size)
        self.agents_type = 'learning'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                
    def init_phases_vectors(self, eng):
        """
        initialises vector representation of the phases
        :param eng: the cityflow simulation engine
        """
        idx = 1
        vec = np.zeros(len(self.phases))
        self.clearing_phase.vector = vec.tolist()
        for phase in self.phases.values():
            vec = np.zeros(len(self.phases))
            if idx != 0:
                vec[idx-1] = 1
            phase.vector = vec.tolist()
            idx+=1    


    def step(self, eng, time, lane_vehs, lanes_count, veh_distance, eps, policy, done):
        local_net = policy.net_local
        memory = policy.memory
        if time % self.action_freq == 0:
            if self.action_type == "reward":
                reward = self.get_reward(lanes_count)
                self.reward = reward
                self.total_rewards += [reward]
                reward = torch.tensor([reward], dtype=torch.float, device=device)
                done = torch.tensor([done], dtype=torch.bool, device=device)
                action = torch.tensor([self.action.ID], device=device)
                next_state = torch.FloatTensor(self.observe(eng, time, lanes_count, lane_vehs, veh_distance),
                                               device=device)
                memory.add(self.state, action, reward, next_state, done)
                self.action_type = "act"

            if self.action_type == "act":
                self.state = torch.FloatTensor(self.observe(eng, time, lanes_count, lane_vehs, veh_distance), device=device)
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

    def observe(self, eng, time, lanes_count, lane_vehs, vehs_distance):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        observations = self.phase.vector + self.get_in_lanes_veh_num(eng, lane_vehs, vehs_distance) + self.get_out_lanes_veh_num(eng, lanes_count)
        return observations

    def act(self, net_local, state, lanes_count, time, eps = 0):
        """
        generates the action to be taken by the agent
        :param net_local: the neural network used in the decision making process
        :param state: the current state of the intersection, given by observe
        :param eps: the epsilon value used in the epsilon greedy learing
        """
        if random.random() > eps:
            state = state.unsqueeze(0)
            net_local.eval()
            with torch.no_grad():
                action_values = net_local(state)
            net_local.train()

            action = action_values.max(1)[1].item()
            return self.phases[action]
        else:
            return self.phases[random.choice(np.arange(self.n_actions))]

        
    def get_out_lanes_veh_num(self, eng, lanes_count):
        """
        gets the number of vehicles on the outgoing lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        lanes_veh_num = []            
        for road in self.out_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                length = self.out_lanes_length[lane]
                lanes_veh_num.append(lanes_count[lane] * 5 / length)
                # lanes_veh_num.append(lanes_count[lane])
        return lanes_veh_num

    def get_in_lanes_veh_num(self, eng, lanes_veh, vehs_distance):
        """
        gets the number of vehicles on the incoming lanes of the intersection
        :param eng: the cityflow simulation engine
        :param lanes_veh: a dictionary with lane ids as keys and list of vehicle ids as values
        :param vehs_distance: dictionary with vehicle ids as keys and their distance on their current lane as value
        """
        lanes_veh_num = []
        for road in self.in_roads:
            lanes = eng.get_road_lanes(road)
            for lane in lanes:
                length = self.in_lanes_length[lane]
                seg1 = 0
                seg2 = 0
                seg3 = 0
                vehs = lanes_veh[lane]
                for veh in vehs:
                    if veh in vehs_distance.keys():
                        if vehs_distance[veh] / length >= 0.66:
                            seg1 += 1
                        elif vehs_distance[veh] / length >= 0.33:
                            seg2 += 1
                        else:
                            seg3 +=1
     
                lanes_veh_num.append(seg1 * (5 / (length/3)))
                lanes_veh_num.append(seg2 * (5 / (length/3)))
                lanes_veh_num.append(seg3 * (5 / (length/3)))


        return lanes_veh_num

