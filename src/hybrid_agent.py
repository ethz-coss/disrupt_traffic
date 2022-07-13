import random
import numpy as np
import torch
import operator
import queue

from learning_agent import Learning_Agent


class Hybrid_Agent(Learning_Agent):

    def __init__(self, eng, ID='', in_roads=[], out_roads=[], n_states=None, lr=None, batch_size=None):
        super().__init__(eng, ID, in_roads, out_roads, n_states, lr, batch_size)
        self.agents_type = 'hybrid'

    def act(self, eng, net_local, state, time, lanes_count, eps = 0):
        """
        generates the action to be taken by the agent
        :param net_local: the neural network used in the decision making process
        :param state: the current state of the intersection, given by observe
        :param eps: the epsilon value used in the epsilon greedy learing
        """

        # self.stabilise(time, lanes_count)
        # if not self.action_queue.empty():
        #     phase = self.action_queue.get()
        #     return phase
        
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            net_local.eval()
            with torch.no_grad():
                action_values = net_local(state)
            net_local.train()
            return self.phases[np.argmax(action_values.cpu().data.numpy())]
        else:
            if random.random() > eps:
                #explore analytically

                # self.stabilise(time, lanes_count)
                # if not self.action_queue.empty():
                #     phase = self.action_queue.get()
                #     return phase

                self.update_clear_green_time(time, eng)
                self.update_priority_idx(time)
                
                phases_priority = {}
                for phase in self.phases.values():
                    movements = [x for x in phase.movements if x not in self.clearing_phase.movements]
                    phase_prioirty = 0
                    for moveID in movements:
                        phase_prioirty += self.movements[moveID].priority

                    phases_priority.update({phase.ID : phase_prioirty})

                action = self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
                return action
            else:
                #explore randomly
                return self.phases[random.choice(np.arange(self.n_actions))]

    # def get_reward(self, eng, lane_vehs):
    #     sum_distance = 0
    #     num_vehs = 0
    #     for lane in self.in_lanes:
    #         for veh in lane_vehs[lane]:
    #             leader = eng.get_leader(veh)
    #             if veh != '' and leader != '':
    #                 leader_distance = float(eng.get_vehicle_info(leader)['distance'])
    #                 sum_distance += abs(leader_distance - float(eng.get_vehicle_info(veh)['distance']))
    #                 num_vehs += 1
    #     for lane in self.out_lanes:
    #         for veh in lane_vehs[lane]:
    #             leader = eng.get_leader(veh)
    #             if veh != '' and leader != '':
    #                 leader_distance = float(eng.get_vehicle_info(leader)['distance'])
    #                 sum_distance += abs(leader_distance - float(eng.get_vehicle_info(veh)['distance'])) 
    #                 num_vehs += 1
    #     if num_vehs == 0:
    #         return self.movements[0].in_length
    #     else:
    #         return sum_distance / num_vehs

    def step(self, eng, time, lane_vehs, lanes_count, veh_distance, eps, memory, local_net, done):
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        if time % self.action_freq == 0:
            if self.action_type == "reward":
                reward = self.get_reward(lanes_count)
                self.reward = reward
                self.total_rewards += [reward]
                self.reward_count += 1
                reward = torch.tensor([reward], dtype=torch.float)
                next_state = torch.FloatTensor(self.observe(eng, time, lanes_count, lane_vehs, veh_distance)).unsqueeze(0)

                memory.add(self.state, self.action.ID, reward, next_state, done)
                self.action_type = "act"

            if self.action_type == "act":
                self.state = np.asarray(self.observe(eng, time, lanes_count, lane_vehs, veh_distance))
                self.action = self.act(eng, local_net, self.state, time, lanes_count, eps=eps)
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
            
    def stabilise(self, time, lanes_count):
        """
        Implements the stabilisation mechanism of the algorithm, updates the action queue with phases that need to be prioritiesd
        :param time: the time in the simulation, at this moment only integer values are supported
        """
        def add_phase_to_queue(priority_list):
            """
            helper function called recursievely to add phases which need stabilising to the queue
            """
            phases_score = {}
            for elem in priority_list:
                for phaseID in elem.phases:
                    if phaseID in phases_score.keys():
                        phases_score.update({phaseID : phases_score[phaseID] + 1})
                    else:
                        phases_score.update({phaseID : 1})

            if [x for x in phases_score.keys() if phases_score[x] != 0]:
                idx = max(phases_score.items(), key=operator.itemgetter(1))[0]
                self.action_queue.put(self.phases[idx])

        T = 180
        T_max = 240        
        priority_list = []
        
        for movement in [x for x in self.movements.values() if x.ID not in self.phase.movements]:              
            Q = movement.get_arr_veh_num(0, time) / time
            if movement.last_on_time == -1:
                waiting_time = 0
            else:
                waiting_time = time - movement.last_on_time

            z = 10 + movement.clearing_time + waiting_time
            n_crit = Q * T * ((T_max - z) / (T_max - T))

            waiting = sum([lanes_count[x] for x in movement.in_lanes])

           
            if waiting > n_crit:
                priority_list.append(movement)
            
        add_phase_to_queue(priority_list)
     
