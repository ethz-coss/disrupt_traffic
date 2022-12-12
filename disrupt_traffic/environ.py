import cityflow

import numpy as np
import random
import os
import functools

from engine.cityflow.intersection import Lane
from gym import utils
import gym
from pettingzoo.utils.env import ParallelEnv, AECEnv
from pettingzoo.utils import agent_selector

class Environment(gym.Env):
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



        self.observations = {agent_id: np.zeros(n_states) for agent_id in self.agent_ids}
        self.actions = {agent_id: None for agent_id in self.agent_ids}
        self.action_probs = {
            agent_id: None for agent_id in self.agent_ids}
        self.rewards = {agent_id: None for agent_id in self.agent_ids}
        self._cumulative_rewards = {agent_id: None for agent_id in self.agent_ids}
        self.dones = {a_id: False for a_id in self.agent_ids}
        self.dones['__all__'] = False
        self.infos =  {agent: False for agent in self.agent_ids}
        if self.agents_type == 'cluster':
            raise NotImplementedError 

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
        assert actions is not None
        self._apply_actions(actions)
        self.sub_steps()

        rewards = self._compute_rewards()
        observations = self._get_obs()
        info = self.infos
        dones = self._compute_dones()

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
        for agent_id, action in actions.items():
            agent = self._agents_dict[agent_id]
            if agent.time_to_act:
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

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.agent_ids}
        dones['__all__'] = self.time > self.num_sim_steps
        return dones

    def _compute_rewards(self):
        self.rewards.update({agent.ID: agent.calculate_reward(self.lanes_count) for agent in self.agents if agent.time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self._agents_dict[ts].time_to_act}

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self.observations[agent]

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

        for veh_id in finished_vehs:
            veh_info = self.vehicles[veh_id]
            veh_info['end_time'] = self.time
            # veh_info['stops'] =

        for veh_id in new_vehs:
            veh_info = self.vehicles.setdefault(veh_id, {})
            veh_info['flow_id'] = veh_id.rsplit('_', 1)[0]
            veh_info['start_time'] = self.time

        self.prev_vehs = current_vehs



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
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def reset(self, seed=None, options=None):
        tts = self.env.eng.get_average_travel_time()
        vehs = self.env.eng.get_finished_vehicle_count()
        rew = sum([np.mean(agent.total_rewards) for agent in self.env.agents])
        print(f'reward: {rew}\ttravel time: {tts}\tvehicles: {vehs}')

        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {a: False for a in self.agents}
        self.cycled_all_agents = False
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
    
    def _was_done_step(self, action):
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert (
            self.dones[agent]
        ), "an agent that was not dead as attempted to be removed"
        del self.dones[agent]
        if agent in self.rewards:
            del self.rewards[agent]
            del self._cumulative_rewards[agent]
            del self.infos[agent]
        self.agents.remove(agent)

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.dones[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                assert self._skip_agent_selection is not None
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()


    def step(self, action):
        if (
            self.dones[self.agent_selection]
        ):
            return self._was_done_step(action=action) # pretend the action is None

        agent = self.agent_selection

        if not self.action_spaces[agent].contains(action):
            raise Exception('Action for agent {} must be in Discrete({}).'
                            'It is currently {}'.format(agent, self.action_spaces[agent].n, action))

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last() or self.cycled_all_agents:
            self.env.sub_steps()
            self.env._get_obs()
            self.rewards = self.env._compute_rewards()
        else:
            self._clear_rewards()
        
        done = self.env._compute_dones()['__all__']
        self.dones = {a: done for a in self.agents}
        self.dones['__all__'] = done

        self.agent_selection = self._agent_selector.next()
        while not self.env._agents_dict[self.agent_selection].time_to_act:
            # print('agent', self.env.time, self.agent_selection, self.env._agents_dict[self.agent_selection].next_act_time)
            if self._agent_selector.is_last():
                self.env.sub_steps()
                self.env._get_obs()
                self.rewards = self.env._compute_rewards()
            self.agent_selection = self._agent_selector.next()

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
