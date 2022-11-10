import cityflow

import numpy as np
import random
import os


# from policy_agent import DPGN, Policy_Agent
from engine.cityflow.intersection import Lane
import gym
from gym import spaces


class Environment(gym.Env):
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
        self.eng = cityflow.Engine(args.sim_config, thread_num=os.cpu_count())
        self.ID = ID
        self.num_sim_steps = args.num_sim_steps
        self.update_freq = args.update_freq      # how often to update the network
        self.batch_size = args.batch_size

        self.eps_start = args.eps_start
        self.eps_end = args.eps_end
        self.eps_decay = args.eps_decay
        self.eps_update = args.eps_update

        self.eps = self.eps_start

        self.agents = []
        self.time = 0
        random.seed(2)

        self.agents_type = args.agents_type

        self.action_freq = 10  # typical update freq for agents

        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Discrete(n_actions)
                for agent_id in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(agent_id)
            }
        )
        self.observation_space = spaces.Dict(
            {
                agent_id: spaces.utils.flatten_space(
                                spaces.Tuple((spaces.Box(0, 1, shape=(n_actions,), dtype=int),
                                              spaces.Box(0, 1, shape=(48,), dtype=float)))
                )
                # TODO: reduce observation space dimensionality
                # agent_id: spaces.Discrete(n_actions, dtype=int)+spaces.Box(0, np.inf, shape=(48,), dtype=float)
                for agent_id in self.eng.get_intersection_ids() if not self.eng.is_intersection_virtual(agent_id)
            }
        )


        if self.agents_type == 'cluster':
            self.cluster_models = Cluster_Models(
                n_states=n_states, n_actions=self.n_actions, lr=args.lr, batch_size=self.batch_size)
            # self.cluster_algo = SOStream.sostream.SOStream(alpha=0, min_pts=9, merge_threshold=0.01)
            self.cluster_algo = Mfd_Clustering(self.cluster_models)

        self.mfd_data = []
        self.agent_history = []

        self.lanes = {}

        for lane_id in self.eng.get_lane_vehicles().keys():
            self.lanes[lane_id] = Lane(self.eng, ID=lane_id)

        self.speeds = []
        self.stops = []
        self.stopped = {}

    def step(self, actions):

        veh_ids = self.eng.get_vehicles()
        speeds = []
        stops = 0

        lane_vehs = self.eng.get_lane_vehicles()
        lanes_count = self.eng.get_lane_vehicle_count()

        self.flow = []
        self.density = []

        for lane_id, lane in self.lanes.items():
            lane.update_flow_data(self.eng, lane_vehs)
            lane.update_speeds(self, lane_vehs[lane_id])

        for veh_id in veh_ids:
            veh_info = self.eng.get_vehicle_info(veh_id)
            speed = float(veh_info['speed'])
            veh_lane = veh_info['drivable']
            speeds.append(speed)
            if veh_lane in self.lanes:
                self.lanes[veh_lane].speeds[-1].append(speed)

            if speed <= 0.1 and veh_id not in self.stopped.keys():
                self.stopped.update({veh_id: 1})
                stops += 1
            elif speed > 0.1 and veh_id in self.stopped.keys():
                self.stopped.pop(veh_id)

        self.speeds.append(np.mean(speeds))
        self.stops.append(stops)


        veh_distance = 0
        if self.agents_type == "hybrid" or self.agents_type == "learning" or self.agents_type == 'cluster' or self.agents_type == 'presslight':
            veh_distance = self.eng.get_vehicle_distance()

        # self._step(self, self.time, None, policy_mapper=None)
        rewards = {}
        for agent in self.agents:
            agent_id = agent.ID
            action = actions[agent_id]

            # if policy_mapper:
            #     policy = policy_mapper(agent.ID)
            # else:
            #     policy = None
            if agent.agents_type == "cluster":
                agent.step(self.eng, self.time, lane_vehs, lanes_count, veh_distance,
                           self.eps, self.cluster_algo, self.cluster_models)
            else:
                reward = agent.step(self.eng, action, self.time, lane_vehs, lanes_count,
                           veh_distance, self.eps)
                rewards[agent_id] = reward

        if self.time % self.action_freq == 0: # TODO: move outside to training
            self.eps = max(self.eps-self.eps_decay, self.eps_end)

        self.eng.next_step()
        self.time += 1
        done = self.time==self.num_sim_steps
        info = {}
        observations = self._get_obs()

        return observations, rewards, done, False, info


    def _get_obs(self):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        lane_vehs = self.eng.get_lane_vehicles()
        lanes_count = self.eng.get_lane_vehicle_count()
        vehs_distance = self.eng.get_vehicle_distance()

        obs = {}
        for agent in self.agents:
            if self.agents_type in ['learning', 'hybrid', 'presslight']:
                obs[agent.ID] = np.array(agent.phase.vector + agent.get_in_lanes_veh_num(self.eng, lane_vehs, vehs_distance) + agent.get_out_lanes_veh_num(self.eng, lanes_count))
        return obs


    def reset(self, seed=None, options=None):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        super().reset(seed=seed)
        self.eng.reset(seed=False)
        self.time = 0
        for agent in self.agents:
            agent.reset_movements()
            agent.total_rewards = []
            agent.action_type = 'act'
        
        obs = self._get_obs()
        info = {}
        return obs, info


    def get_mfd_data(self, time_window=60):
        mfd_detailed = {}

        for lane_id in self.eng.get_lane_vehicles().keys():
            mfd_detailed[lane_id] = {"speed": [], "density": []}
                    
        data = mfd_detailed[lane_id]

        for lane_id, lane in self.lanes.items():
            data = mfd_detailed[lane_id]
            speed = data['speed']
            density = data['density']
            
        #     _lanespeeds = sum(lane.speeds, [])
            _lanedensity = np.subtract(lane.arr_vehs_num, lane.dep_vehs_num).cumsum()
            for t in range(3600):
                time_window = min(time_window, t+1)
                idx_start = t
                idx_end = t+time_window

                s = np.mean(sum(lane.speeds[idx_start:idx_end], []))
                d = _lanedensity[idx_start:idx_end].mean() / lane.length 

                speed.append(s)
                density.append(d)

        return mfd_detailed


def get_mfd_data(time, lanes_count, lanes):
    # TODO: revise/remove
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
