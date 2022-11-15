import cityflow

import numpy as np
import random
import os
import functools

# from policy_agent import DPGN, Policy_Agent
from engine.cityflow.intersection import Lane
from gym import spaces
from gym import utils

from pettingzoo.utils.env import ParallelEnv


class Environment(ParallelEnv, utils.EzPickle):
    """
    The class Environment represents the environment in which the agents operate in this case it is a city
    consisting of roads, lanes and intersections which are controled by the agents
    """

    metadata = {"name": "cityflow"}

    def __init__(self, args, ID=0, n_actions=9, n_states=44, AgentClass=None):
        """
        initialises the environment with the arguments parsed from the user input
        :param args: the arguments input by the user
        :param n_actions: the number of possible actions for the learning agent, corresponds to the number of available phases
        :param n_states: the size of the state space for the learning agent
        """
        utils.EzPickle.__init__(self, args, ID, n_actions, n_states, AgentClass)
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

        self._agents = []
        self.time = 0
        random.seed(2)

        self.agents_type = args.agents_type

        self.action_freq = 10  # typical update freq for agents

        self.possible_agents = [x for x in self.eng.get_intersection_ids()
                 if not self.eng.is_intersection_virtual(x)]

        self.agents = self.possible_agents
        self._agents=[]
        for agent_id in self.possible_agents:
            new_agent = AgentClass(self, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(
                agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=args.batch_size)
            self._agents.append(new_agent)

        self.action_spaces = spaces.Dict({
                agent_id: spaces.Discrete(n_actions)
                for agent_id in self.possible_agents
            })
        self.observation_spaces = spaces.Dict({
                agent_id: spaces.utils.flatten_space(
                                spaces.Tuple((spaces.Box(0, 1, shape=(n_actions,), dtype=int),
                                              spaces.Box(0, 100, shape=(48,), dtype=float)))
                )
                # TODO: reduce observation space dimensionality
                # agent_id: spaces.Discrete(n_actions, dtype=int)+spaces.Box(0, np.inf, shape=(48,), dtype=float)
                for agent_id in self.possible_agents
            })



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

        # metrics
        self.speeds = []
        self.stops = []
        self.waiting_times = []
        self.stopped = {}

        # for detailed logging. DO NOT USE WHEN TRAINING
        self.vehicles = {}
        self.prev_vehs = set()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions):

        veh_speeds = self.eng.get_vehicle_speed()
        stops = 0

        lane_vehs = self.eng.get_lane_vehicles()
        lanes_count = self.eng.get_lane_vehicle_count()

        self.flow = []
        self.density = []

        for lane_id, lane in self.lanes.items():
            lane.update_flow_data(self.eng, lane_vehs)
            # lane.update_speeds(self, lane_vehs[lane_id], veh_speeds)

        for veh_id, speed in veh_speeds.items():

            if speed <= 0.1:
                veh_stop = self.stopped.setdefault(veh_id, 0) + 1
                if veh_stop==1: stops += 1 # first stop
            elif speed > 0.1 and veh_id in self.stopped.keys():
                self.waiting_times.append(self.stopped[veh_id])
                self.stopped.pop(veh_id)

        self.speeds.append(np.mean(list(veh_speeds.values())))
        self.stops.append(stops)


        veh_distance = 0
        if self.agents_type == "hybrid" or self.agents_type == "learning" or self.agents_type == 'cluster' or self.agents_type == 'presslight':
            veh_distance = self.eng.get_vehicle_distance()

        # self._step(self, self.time, None, policy_mapper=None)
        for agent in self._agents:
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
                agent.apply_action(self.eng, action, self.time, lane_vehs, lanes_count,
                           veh_distance, self.eps)

        if self.time % self.update_freq == 0: # TODO: move outside to training
            self.eps = max(self.eps-self.eps_decay, self.eps_end)

        self.eng.next_step()
        self.time += 1

        observations = self._get_obs()
        rewards = {agent.ID:agent.calculate_reward(lanes_count) for agent in self._agents}
        done = {id:self.time==self.num_sim_steps if x is not None else None for id, x in observations.items()}
        info = {id:self.time==self.num_sim_steps if x is not None else None for id, x in observations.items()}
        info['__all__'] = None

        return observations, rewards, info, done#, info


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
        for agent in self._agents:
            obs_agent = None
            if self.agents_type in ['learning', 'hybrid', 'presslight']:
                obs_agent = agent.observation # NOTE: observations are only updated when the agent acts
                # obs_agent = np.array(agent.phase.vector + agent.get_in_lanes_veh_num(self.eng, lane_vehs, vehs_distance) + agent.get_out_lanes_veh_num(self.eng, lanes_count))
            obs[agent.ID] = obs_agent
        return obs

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return self._get_obs['agent']


    def reset(self, seed=None, options=None):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        # super().reset(seed=seed)
        if seed is None:
            seed = random.randint(1,1e6)
        self.eng.reset(seed=False)
        self.eng.set_random_seed(seed)
        self.time = 0
        for agent in self._agents:
            agent.reset_movements()
            agent.total_rewards = []
            agent.action_type = 'act'
            agent.action_freq = self.action_freq

        # self.speeds = []
        # self.stops = []
        self.waiting_times = []
        self.stopped = {}
        self.vehicles = {}
        self.prev_vehs = set()

        obs = self._get_obs()
        info = {}
        return obs#, info


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

    def detailed_log(self):
        current_vehs = set(self.eng.get_vehicles())
        finished_vehs = self.prev_vehs - current_vehs
        new_vehs = current_vehs - self.prev_vehs

        # for veh_id in current_vehs:
        #     veh_info = self.vehicles[veh_id]
        #     veh_info[]

        for veh_id in finished_vehs:
            veh_info = self.vehicles[veh_id]
            veh_info['end_time'] = self.time
            # veh_info['stops'] = 

        for veh_id in new_vehs:            
            veh_info = self.vehicles.setdefault(veh_id, {})
            veh_info['flow_id'] = veh_id.rsplit('_', 1)[0]
            veh_info['start_time'] = self.time

        self.prev_vehs = current_vehs

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
