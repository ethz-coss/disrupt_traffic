import random
import numpy as np
import torch
import operator

from agents.learning_agent import Learning_Agent


class Hybrid_Agent(Learning_Agent):

    def __init__(self, eng, ID='', in_roads=[], out_roads=[], lr=None, batch_size=None):
        super().__init__(eng, ID, in_roads, out_roads, lr, batch_size)
        self.agents_type = 'hybrid'

    def apply_action(self, eng, action, time, lane_vehs, lanes_count):
        self.update_arr_dep_veh_num(lane_vehs, lanes_count)
        super().apply_action(eng, action, time, lane_vehs, lanes_count)

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
                        phases_score.update(
                            {phaseID: phases_score[phaseID] + 1})
                    else:
                        phases_score.update({phaseID: 1})

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
