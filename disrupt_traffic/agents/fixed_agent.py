from engine.cityflow.intersection import Phase
from agents.agent import Agent
import numpy as np

class Fixed_Agent(Agent):

    def __init__(self, env, ID='', **kwargs):
        super().__init__(env, ID)
        self.agents_type = 'fixed'


    def init_phases(self, eng):
        """
        initialises the phases of the Agent based on the intersection phases extracted from the simulation data
        :param eng: the cityflow simulation engine
        """
        for idx, phase_tuple in enumerate(eng.get_intersection_phases(self.ID)):
            phases = phase_tuple[0]
            types = phase_tuple[1]
            empty_phases = []
            
            new_phase_moves = []
            for move, move_type in zip(phases, types):
                key = tuple(move)
                self.movements[key].move_type = move_type
                new_phase_moves.append(self.movements[key].ID)

            if types and all(x == 1 for x in types): #1 -> turn right
                self.clearing_phase = Phase(idx, new_phase_moves)

            elif new_phase_moves:
                if set(new_phase_moves) not in [set(x.movements) for x in self.phases.values()]:
                    new_phase = Phase(idx, new_phase_moves)                    
                    self.phases.update({idx : new_phase})
            else:
                empty_phases.append(idx)

            if empty_phases:
                self.clearing_phase = Phase(empty_phases[0], [])
                self.phases.update({empty_phases[0] : self.clearing_phase})

        self.phase = self.clearing_phase
        temp_moves = dict(self.movements)
        self.movements.clear()
        for move in temp_moves.values():
            move.phases = []
            self.movements.update({move.ID : move})
            
        for phase in self.phases.values():
            for move in phase.movements:
                if phase.ID not in self.movements[move].phases:
                    self.movements[move].phases.append(phase.ID)
    

    def act(self, lanes_count):
        phaseID = self.action.ID % len(self.phases)
        phaseID += 1
        return self.phases[phaseID]
      


    def step(self, eng, time, lane_vehs, lanes_count, veh_distance, eps, memory, local_net, done):
        """
        represents a single step of the simulation for the analytical agent
        :param time: the current timestep
        :param done: flag indicating weather this has been the last step of the episode, used for learning, here for interchangability of the two steps
        """
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        if time % self.action_freq == 0:
            if self.action_type == "act":
                self.total_rewards += self.get_reward(lanes_count)
                self.reward_count += 1
                self.action = self.act(lanes_count)
                
                self.green_time = int(np.min([self.movements[move_id].get_green_time(time, [], eng) for move_id in self.action.movements]))
                loop_start_action_id = self.action.ID
                while self.green_time == 0:
                    self.action = self.act(lanes_count)
                    self.green_time = int(np.max([self.movements[move_id].get_green_time(time, [], eng) for move_id in self.action.movements]))
                    if self.action.ID == loop_start_action_id:
                        self.green_time = 10
                        break
                # self.green_time = 10

                if self.phase.ID != self.action.ID:
                    self.update_wait_time(time, self.action, self.phase, lanes_count)
                    self.set_phase(eng, self.clearing_phase)
                    self.action_freq = time + self.clearing_time
                    self.action_type = "update"
                    
                else:
                    self.action_freq = time + self.green_time

            elif self.action_type == "update":
                self.set_phase(eng, self.action)
                self.action_freq = time + self.green_time
                self.action_type = "act"
