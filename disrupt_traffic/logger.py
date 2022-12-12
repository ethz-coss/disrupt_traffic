import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import random
import pickle
import dill
from network_parser import get_network

class Logger:
    """
    The Logger class is responsible for logging data, building representations and saving them in a specified location
    """

    def __init__(self, args):
        """
        Initialises the logger object
        :param args: the arguments passed by the user
        """

        self.args = args

        self.veh_count = []
        self.travel_time = []
        self.losses = []
        self.plot_rewards = []
        self.episode_losses = []
        self.delays = []
        self.travel_times = []

        self.reward = 0

        config_dir, config_file = os.path.split(args.sim_config)
        scenario_name = os.path.basename(config_dir)
        exp_name = f"{os.path.splitext(config_file)[0]}_{args.agents_type}"

        if args.load != None or args.load_cluster != None:
            scenario_name += "_load"

        self.log_path = os.path.join(args.path, scenario_name, exp_name)
        head, tail = os.path.split(self.log_path)
        i = 1

        if args.ID:
            self.log_path = os.path.join(head, f'{tail}({args.ID})')
        else:
            while os.path.exists(self.log_path):
                self.log_path = os.path.join(head, f'{tail}({i})')
                i += 1
        print(f'saving to {self.log_path}')
        os.makedirs(self.log_path)

    def log_measures(self, environ):
        """
        Logs measures such as reward, vehicle count, average travel time and q losses, works for learning agents and aggregates over episodes
        :param environ: the environment in which the model was run
        """
        self.reward = 0
        for agent in environ.agents:
            self.reward += np.mean(agent.total_rewards)

        self.plot_rewards.append(self.reward)
        self.veh_count.append(environ.eng.get_finished_vehicle_count())
        self.travel_time.append(environ.eng.get_average_travel_time())
        self.episode_losses.append(np.mean(self.losses))

    def log_mfd(self, environ, time_window=60):
        data = environ.get_mfd_data(time_window=time_window)
        road_dict = {}
        for lane_id, lane in environ.lanes.items():
            road_id, direction = lane_id.rsplit('_',1)
            road_data = road_dict.setdefault(road_id, {'density': [], 'speed': []})
            road_data['density'].append(data[lane_id]['density'])
            road_data['speed'].append(data[lane_id]['speed'])

        output = {}
        for road_id, road_data in road_dict.items():
            output[road_id] = {}
            density = np.vstack(road_data['density'])
            speed = np.vstack(road_data['speed'])
            output[road_id]['speed'] = np.nansum(speed*density, axis=0)/np.nansum(density, axis=0)
            output[road_id]['density'] = np.nanmean(density, axis=0)

        self.mfd = output

    def log_delays(self, config, environ):
        network, roads, flows = get_network(config)
        delays = []
        travel_times = []
        for veh_id, veh_data in environ.vehicles.items():
            if 'end_time' in veh_data.keys():
                tt = veh_data['end_time'] - veh_data['start_time']
            else:
                continue # ignore unfinished vehicles
                # tt = environ.time - veh_data['start_time']
            flow_id = veh_data['flow_id']
            delay = (tt - flows[flow_id]['freeflow_time'])/flows[flow_id]['routelength']*1000 # secs/km
            delays.append(delay)
            travel_times.append(tt)
        self.delays.append(delays)
        self.travel_times.append(travel_times)
        return delays, travel_times

    def serialise_data(self, environ, policy=None):
        """
        Serialises the waiting times data and rewards for the agents as dictionaries with agent ID as a key
        """
        waiting_time_dict = {}
        reward_dict = {}

        self.log_mfd(environ, time_window=60)

        for agent in environ.agents:
            waiting_time_dict.update({agent.ID: {}})
            reward_dict.update({agent.ID: agent.total_rewards})
            for move in agent.movements.values():
                waiting_time_dict[agent.ID].update(
                    {move.ID: (move.max_waiting_time, move.waiting_time_list)})
        if policy:
            with open(os.path.join(self.log_path, "memory.dill"), "wb") as f:
                dill.dump(policy.memory.memory, f)

        with open(os.path.join(self.log_path, "agent_history.dill"), "wb") as f:
            pickle.dump(environ.agent_history, f)
        with open(os.path.join(self.log_path, "waiting_time.pickle"), "wb") as f:
            pickle.dump(waiting_time_dict, f)

        with open(os.path.join(self.log_path, "agents_rewards.pickle"), "wb") as f:
            pickle.dump(reward_dict, f)

        with open(os.path.join(self.log_path, "mfd.pickle"), "wb") as f:
            pickle.dump(self.mfd, f)

        with open(os.path.join(self.log_path, "stops.pickle"), "wb") as f:
            pickle.dump(environ.stops, f)
        with open(os.path.join(self.log_path, "speeds.pickle"), "wb") as f:
            pickle.dump(environ.speeds, f)

        with open(os.path.join(self.log_path, "delays.pickle"), "wb") as f:
            pickle.dump(self.delays, f)

        with open(os.path.join(self.log_path, "travel_times.pickle"), "wb") as f:
            pickle.dump(self.travel_times, f)

        if environ.agents_type in ['learning', 'hybrid', 'presslight', 'policy', 'denflow']:
            with open(os.path.join(self.log_path, "episode_rewards.pickle"), "wb") as f:
                pickle.dump(self.plot_rewards, f)

            with open(os.path.join(self.log_path, "episode_veh_count.pickle"), "wb") as f:
                pickle.dump(self.veh_count, f)

            with open(os.path.join(self.log_path, "episode_travel_time.pickle"), "wb") as f:
                pickle.dump(self.travel_time, f)

    def save_log_file(self, environ):
        """
        Creates and saves a log file with information about the experiment in a .txt format
        :param environ: the environment in which the model was run
        """
        log_file = open(self.log_path + "/logs.txt", "w+")

        log_file.write(str(self.args.sim_config))
        log_file.write("\n")
        log_file.write(str(self.args.num_episodes))
        log_file.write("\n")
        log_file.write(str(self.args.num_sim_steps))
        log_file.write("\n")
        log_file.write(str(self.args.update_freq))
        log_file.write("\n")
        log_file.write(str(self.args.batch_size))
        log_file.write("\n")
        log_file.write(str(self.args.lr))
        log_file.write("\n")

        log_file.write("mean vehicle count: " + str(np.mean(self.veh_count[self.args.num_episodes-10:])) + " with sd: " + str(np.std(self.veh_count[self.args.num_episodes-10:])) +
                       "\nmean travel time: " + str(np.mean(self.travel_time[self.args.num_episodes-10:])) +
                       " with sd: " + str(np.std(self.travel_time[self.args.num_episodes-10:])) +
                       "\nmax vehicle time: " + str(np.max(self.veh_count)) +
                       "\nmin travel time: " + str(np.min(self.travel_time))
                       )
        log_file.write("\n")
        log_file.write("best epoch: " + str(environ.best_epoch))
        log_file.write("\n")
        log_file.write("\n")

        for agent in environ.agents:
            log_file.write(agent.ID + "\n")
            for move in agent.movements.values():
                log_file.write("movement " + str(move.ID) +
                               " max wait time: " + str(move.max_waiting_time) + "\n")
                if not move.waiting_time_list:
                    move.waiting_time_list = [0]
                log_file.write("movement " + str(move.ID) + " avg wait time: " +
                               str(np.mean(move.waiting_time_list)) + "\n")
            log_file.write("\n")

        log_file.write("\n")

        log_file.close()

    def save_models(self, policies, flag):
        """
        Saves machine learning models (for now just neural networks)
        :param environ: the environment in which the model was run
        :param flag: the flag indicating which model to save - throughput based or avg. travel time based
        """
        for policy in policies:
            policy.save(self.log_path, flag)

    def save_clusters(self, environ):
        if not os.path.isdir(self.log_path + '/cluster_nets'):
            os.mkdir(self.log_path + '/cluster_nets')
        for key, model in zip(environ.cluster_models.model_dict.keys(), environ.cluster_models.model_dict.values()):
            torch.save(model[0].state_dict(), self.log_path +
                       '/cluster_nets/cluster' + str(key) + '_q_net.pt')
            torch.save(model[1].state_dict(), self.log_path +
                       '/cluster_nets/cluster' + str(key) + '_target_net.pt')

        for key, memory in zip(environ.cluster_models.memory_dict.keys(), environ.cluster_models.memory_dict.values()):
            with open(os.path.join(self.log_path, f"cluster_nets/memory{key}.dill"), "wb") as f:
                dill.dump(memory, f)

        # environ.clustering.M = [environ.clustering.M[-1]]

        with open(os.path.join(self.log_path, "clustering.dill"), "wb") as f:
            dill.dump(environ.cluster_algo, f)

    def plot_pressure(self, environ):
        """
        plots pressure as a function of time, both avg pressure of all intersections and individual pressure
        :param environ: the environment, after some simulation steps
        """
        plot_avg = []
        plot_std = []
        for data in environ.log_pressure:
            plot_avg.append(np.mean(data))
            plot_std.append(np.std(data))

        # plt.errorbar(range(len(plot_avg)), plot_avg, yerr=plot_std)
        plt.plot(plot_avg)
        plt.savefig(self.log_path + '/avg_pressure.png')
        plt.clf()

        plt.plot(environ.log_pressure)
        plt.savefig(self.log_path + '/partial_pressure.png')
        plt.clf()

        with open(os.path.join(self.log_path, "avg_pressure.pickle"), "wb") as f:
            pickle.dump(plot_avg, f)
        with open(os.path.join(self.log_path, "partial_pressure.pickle"), "wb") as f:
            pickle.dump(environ.log_pressure, f)
