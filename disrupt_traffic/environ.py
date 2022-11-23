import cityflow

import numpy as np
import random
import os
import functools

from engine.cityflow.intersection import Lane
from gym import utils

from pettingzoo.utils.env import ParallelEnv, AECEnv

class Environment(Parallel, utils.EzPickle):
    """
    The class Environment represents the environment in which the agents operate in this case it is a city
    consisting of roads, lanes and intersections which are controled by the agents
    """

    metadata = {"name": "cityflow"}

    def __init__(self, args=None, ID=0, n_actions=9, n_states=44, AgentClass=None):
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

        self.time = 0
        random.seed(2)

        self.lane_vehs = self.eng.get_lane_vehicles()
        self.lanes_count = self.eng.get_lane_vehicle_count()

        self.agents_type = args.agents_type

        self.action_freq = 10  # typical update freq for agents

        self.agent_ids = [x for x in self.eng.get_intersection_ids()
                                if not self.eng.is_intersection_virtual(x)]
        self.agents = []
        self._agents_dict = {}
        for agent_id in self.agent_ids:
            new_agent = AgentClass(self, ID=agent_id, in_roads=self.eng.get_intersection_in_roads(
                agent_id), out_roads=self.eng.get_intersection_out_roads(agent_id), n_states=n_states, lr=args.lr, batch_size=args.batch_size)
            self.agents.append(new_agent)
            self._agents_dict[agent_id] = new_agent



        self.observations = {agent_id: None for agent_id in self.agent_ids}
        self.actions = {agent_id: None for agent_id in self.agent_ids}
        self.action_probs = {
            agent_id: None for agent_id in self.agent_ids}
        self.rewards = {agent_id: None for agent_id in self.agent_ids}
        # self.dones = {a_id: False for a_id in self.agent_ids}
        # self.dones['__all__'] = False

        if self.agents_type == 'cluster':
            raise NotImplementedError 
            # self.cluster_models = Cluster_Models(
            #     n_states=n_states, n_actions=self.n_actions, lr=args.lr, batch_size=self.batch_size)
            # # self.cluster_algo = SOStream.sostream.SOStream(alpha=0, min_pts=9, merge_threshold=0.01)
            # self.cluster_algo = Mfd_Clustering(self.cluster_models)

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

    @property
    def observation_space(self):
        return self.agents[0].observation_space
    
    @property
    def action_space(self):
        return self.agents[0].action_space

    def observation_spaces(self, ts_id):
        return self._agents_dict[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self._agents_dict[ts_id].action_space

    def step(self, actions):
        self._apply_actions(actions)
        self.sub_steps()

        rewards = {agent.ID: agent.calculate_reward(self.lanes_count) for agent in self.agents
                   if agent.time_to_act}
        observations = self._get_obs()
        info = {}
        dones = {a_id: self.time == self.num_sim_steps for a_id in self.agent_ids}
        dones['__all__'] = self.time == self.num_sim_steps

        return observations, rewards, dones, info

    def sub_steps(self):
        time_to_act = False
        while not time_to_act:
            self.eng.next_step()
            self.time += 1

            stops = 0
            veh_speeds = self.eng.get_vehicle_speed()
            self.lane_vehs = self.eng.get_lane_vehicles()
            self.lanes_count = self.eng.get_lane_vehicle_count()

            for lane_id, lane in self.lanes.items():
                lane.update_flow_data(self.eng, self.lane_vehs)
                lane.update_speeds(self, self.lane_vehs[lane_id], veh_speeds)

            for veh_id, speed in veh_speeds.items():
                if speed <= 0.1:
                    veh_stop = self.stopped.setdefault(veh_id, 0) + 1
                    if veh_stop == 1:
                        stops += 1  # first stop
                elif speed > 0.1 and veh_id in self.stopped.keys():
                    self.waiting_times.append(self.stopped[veh_id])
                    self.stopped.pop(veh_id)

            self.speeds.append(np.mean(list(veh_speeds.values())))
            self.stops.append(stops)

            if self.time % self.update_freq == 0:  # TODO: move outside to training
                self.eps = max(self.eps-self.eps_decay, self.eps_end)

            for agent in self.agents:
                if agent.agents_type == "cluster":
                    raise NotImplementedError
                agent.update()
                if agent.time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        for agent in self.agents:
            if agent.time_to_act:
                action = actions[agent.ID]
                agent.apply_action(self.eng, action, self.time,
                                   self.lane_vehs, self.lanes_count)

    def _get_obs(self):
        """
        generates the observations made by the agents
        :param eng: the cityflow simulation engine
        :param time: the time of the simulation
        :param lanes_count: a dictionary with lane ids as keys and vehicle count as values
        """
        vehs_distance = self.eng.get_vehicle_distance()

        self.observations.update({agent.ID: agent.observe(
            vehs_distance) for agent in self.agents if agent.time_to_act})
        return self.observations.copy()

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self._get_obs['agent']

    def reset(self, seed=None, options=None):
        """
        resets the movements amd rewards for each agent and the simulation environment, should be called after each episode
        """
        # super().reset(seed=seed)
        if seed is None:
            seed = random.randint(1, 1e6)
        self.eng.reset(seed=False)
        self.eng.set_random_seed(seed)
        self.time = 0
        for agent in self.agents:
            agent.reset()
            agent.next_act_time = self.action_freq

        for lane_id, lane in self.lanes.items():
            lane.speeds = []
            lane.dep_vehs_num = []
            lane.arr_vehs_num = []
            lane.prev_vehs = set()

        self.lane_vehs = self.eng.get_lane_vehicles()
        self.lanes_count = self.eng.get_lane_vehicle_count()
        # self.speeds = []
        # self.stops = []
        self.waiting_times = []
        self.stopped = {}
        self.vehicles = {}
        self.prev_vehs = set()

        obs = self._get_obs()
        info = {}
        return obs

    def get_mfd_data(self, time_window=60):
        mfd_detailed = {}

        for lane_id in self.eng.get_lane_vehicles().keys():
            mfd_detailed[lane_id] = {"speed": [], "density": []}

        for lane_id, lane in self.lanes.items():
            data = mfd_detailed[lane_id]
            speed = data['speed']
            density = data['density']

        #     _lanespeeds = sum(lane.speeds, [])
            _lanedensity = np.subtract(
                lane.arr_vehs_num, lane.dep_vehs_num).cumsum()
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



class EnvironmentParallel(AECEnv, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': "sumo_rl_v0", 'is_parallelizable': True}

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = Environment(**self._kwargs)

        self.agents = self.env.agent_ids
        self.possible_agents = self.env.agent_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        obs = self.env.observations[agent].copy()
        return obs

    def state(self):
        raise NotImplementedError('Method state() currently not implemented.')

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)
    
    # def save_csv(self, out_csv_name, run):
    #     self.env.save_csv(out_csv_name, run)

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception('Action for agent {} must be in Discrete({}).'
                            'It is currently {}'.format(agent, self.action_spaces[agent].n, action))

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.env._compute_info()
        else:
            self._clear_rewards()
        
        done = self.env.time == self.env.num_sim_steps
        self.truncations = {a : done for a in self.agents}
        self.dones = {a : done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()