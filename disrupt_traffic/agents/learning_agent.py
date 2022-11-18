import torch
import numpy as np
import random

from agents.agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CAR_LENGTH = 5  # m


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

        self.agents_type = 'learning'

        vehs_distance = env.eng.get_vehicle_distance()
        self.observation = self.observe(vehs_distance)

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
            idx += 1

    def observe(self, vehs_distance):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        eng = self.env.eng
        lane_vehs = self.env.lane_vehs
        lanes_count = self.env.lanes_count
        observations = self.phase.vector + self.get_in_lanes_veh_num(
            eng, lane_vehs, vehs_distance) + self.get_out_lanes_veh_num(eng, lanes_count)
        return np.array(observations)

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
                        if vehs_distance[veh] / length >= (2/3):
                            seg1 += 1
                        elif vehs_distance[veh] / length >= (1/3):
                            seg2 += 1
                        else:
                            seg3 += 1

                lanes_veh_num.append(seg1 * (5 / (length/3)))
                lanes_veh_num.append(seg2 * (5 / (length/3)))
                lanes_veh_num.append(seg3 * (5 / (length/3)))

        return lanes_veh_num
