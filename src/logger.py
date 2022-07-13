import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import random
import pickle
import dill

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

        self.reward = 0

        self.log_path = args.path + "/" + args.sim_config.split('/')[2] +'_' + 'config' + '_' + str(args.agents_type)

        if args.load != None or args.load_cluster != None:
            self.log_path += "_load"
            
        
        old_path = self.log_path
        i = 1

        if args.ID:
            self.log_path = self.log_path + '(' + str(args.ID) + ')'
        else:
            while os.path.exists(self.log_path):
                self.log_path = old_path + "(" + str(i) + ")"
                i += 1

        os.mkdir(self.log_path)


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
        
    def serialise_data(self, environ):
        """
        Serialises the waiting times data and rewards for the agents as dictionaries with agent ID as a key
        """
        waiting_time_dict = {}
        reward_dict = {}
        
        for agent in environ.agents:
            waiting_time_dict.update({agent.ID : {}})
            reward_dict.update({agent.ID : agent.total_rewards})
            for move in agent.movements.values():
                waiting_time_dict[agent.ID].update({move.ID : (move.max_waiting_time, move.waiting_time_list)})


        with open(self.log_path + "/" + "memory.dill", "wb") as f:
            dill.dump(environ.memory.memory, f)
            
        with open(self.log_path + "/" + "agent_history.dill", "wb") as f:
            pickle.dump(environ.agent_history, f)
        with open(self.log_path + "/" + "waiting_time.pickle", "wb") as f:
            pickle.dump(waiting_time_dict, f)
            
        with open(self.log_path + "/" + "agents_rewards.pickle", "wb") as f:
            pickle.dump(reward_dict, f)
            
        with open(self.log_path + "/" + "mfd.pickle", "wb") as f:
            pickle.dump(environ.mfd_data, f)

        with open(self.log_path + "/" + "stops.pickle", "wb") as f:
            pickle.dump(environ.stops, f)
        with open(self.log_path + "/" + "speeds.pickle", "wb") as f:
            pickle.dump(environ.speeds, f) 

        if environ.agents_type == 'learning' or environ.agents_type == 'hybrid' or environ.agents_type == 'presslight' or environ.agents_type == 'policy':
            with open(self.log_path + "/" + "episode_rewards.pickle", "wb") as f:
                pickle.dump(self.plot_rewards, f)

            with open(self.log_path + "/" + "episode_veh_count.pickle", "wb") as f:
                pickle.dump(self.veh_count, f)

            with open(self.log_path + "/" + "episode_travel_time.pickle", "wb") as f:
                pickle.dump(self.travel_time, f)
            
    def save_log_file(self, environ):
        """
        Creates and saves a log file with information about the experiment in a .txt format
        :param environ: the environment in which the model was run
        """
        log_file = open(self.log_path + "/logs.txt","w+")

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
                log_file.write("movement " + str(move.ID) + " max wait time: " + str(move.max_waiting_time) + "\n")
                if not move.waiting_time_list: move.waiting_time_list = [0]
                log_file.write("movement " + str(move.ID) + " avg wait time: " + str(np.mean(move.waiting_time_list)) + "\n")
            log_file.write("\n")
    
        log_file.write("\n")
        
        log_file.close()


    def save_models(self, environ, flag):
        """
        Saves machine learning models (for now just neural networks)
        :param environ: the environment in which the model was run
        :param flag: the flag indicating which model to save - throughput based or avg. travel time based
        """
        if flag is None:
            torch.save(environ.local_net.state_dict(), self.log_path + '/reward_q_net.pt')
            torch.save(environ.target_net.state_dict(), self.log_path + '/reward_target_net.pt') 
        elif flag:
            torch.save(environ.local_net.state_dict(), self.log_path + '/throughput_q_net.pt')
            torch.save(environ.target_net.state_dict(), self.log_path + '/throughput_target_net.pt')
        else: 
            torch.save(environ.local_net.state_dict(), self.log_path + '/time_q_net.pt')
            torch.save(environ.target_net.state_dict(), self.log_path + '/time_target_net.pt')



    def save_clusters(self, environ):
        if not os.path.isdir(self.log_path + '/cluster_nets'):
            os.mkdir(self.log_path + '/cluster_nets')
        for key, model in zip(environ.cluster_models.model_dict.keys(), environ.cluster_models.model_dict.values()):
            torch.save(model[0].state_dict(), self.log_path + '/cluster_nets/cluster' + str(key) + '_q_net.pt')
            torch.save(model[1].state_dict(), self.log_path + '/cluster_nets/cluster' + str(key) + '_target_net.pt')

        for key, memory in zip(environ.cluster_models.memory_dict.keys(), environ.cluster_models.memory_dict.values()):
            with open(self.log_path + "/cluster_nets/memory" + str(key) + ".dill", "wb") as f:
                dill.dump(memory, f)

            
        # environ.clustering.M = [environ.clustering.M[-1]]
        
        with open(self.log_path + "/" + "clustering.dill", "wb") as f:
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

        with open(self.log_path + "/" + "avg_pressure.pickle", "wb") as f:
            pickle.dump(plot_avg, f)
        with open(self.log_path + "/" + "partial_pressure.pickle", "wb") as f:
            pickle.dump(environ.log_pressure, f)
