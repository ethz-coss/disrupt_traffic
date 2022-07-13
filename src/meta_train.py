import cityflow
import numpy as np
import torch

import math
import random
import itertools

import argparse
import os
import json

from population import Env_Config, populate_envs, populate_agents

from dqn import DQN, ReplayMemory, optimize_model
from environ import Environment
from logger import Logger

from learning_agent import Learning_Agent
from analytical_agent import Analytical_Agent
from hybrid_agent import Hybrid_Agent


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", default='../scenarios/1x1/',  type=str, help="the relative path to the simulation config file")

    parser.add_argument("--num_episodes", default=1, type=int,
                        help="the number of episodes to run (one episosde consists of a full simulation run for num_sim_steps)")
    parser.add_argument("--num_sim_steps", default=300, type=int, help="the number of simulation steps, one step corresponds to 1 second")
    parser.add_argument("--agents_type", default='learning', type=str, help="the type of agents learning/policy/analytical/hybrid/demand")

    parser.add_argument("--update_freq", default=10, type=int,
                        help="the frequency of the updates (training pass) of the deep-q-network, default=10")
    parser.add_argument("--batch_size", default=64, type=int, help="the size of the mini-batch used to train the deep-q-network, default=64")
    parser.add_argument("--lr", default=5e-4, type=float, help="the learning rate for the dqn, default=5e-4")
    parser.add_argument("--eps_start", default=1, type=float, help="the epsilon start")
    parser.add_argument("--eps_end", default=0.01, type=float, help="the epsilon decay")
    parser.add_argument("--eps_decay", default=0.95, type=float, help="the epsilon decay")
    # parser.add_argument("--eps_decay", default=5e-5, type=float, help="the epsilon decay")
    parser.add_argument("--eps_update", default=1799, type=float, help="how frequently epsilon is decayed")
    parser.add_argument("--load", default=None, type=str, help="path to the model to be loaded")

    return parser.parse_args()

def generate_baseline(pop_envs):
    total_freeflow_throughput = []
    for env in pop_envs:
        agent_ids = [x for x in env.eng.get_intersection_ids() if not env.eng.is_intersection_virtual(x)]
        agent_id = agent_ids[0]
        baseline_agent = Analytical_Agent(env.eng, ID=agent_id)
        env.analytic_reward, env.analytic_avg_travel_time, env.analytic_throughput = train(baseline_agent, env, args)
        total_freeflow_throughput += [env.freeflow_throughput]
        
    analytic_reward = np.mean([env.analytic_reward for env in pop_envs])
    analytic_reward_std = np.std([env.analytic_reward for env in pop_envs])
    analytic_avg_travel_time = np.mean([env.analytic_avg_travel_time for env in pop_envs])
    analytic_throughput = np.mean([env.analytic_throughput for env in pop_envs])

    return analytic_reward, analytic_reward_std, analytic_avg_travel_time, analytic_throughput, total_freeflow_throughput


def train(agent, environ, args, test=False):
    logger = Logger(args)
    step = 0
    environ.agents = [agent]
    for i_episode in range(args.num_episodes):
        done = False
        environ.reset()
        t = 0

        while t < args.num_sim_steps:
            if t >= args.num_sim_steps-1: done = True
                            
            environ.step(t, done)   
            t += 1
      
            step = (step+1) % environ.update_freq
            if test == False and (agent.agents_type == 'learning' or agent.agents_type == 'hybrid') and step == 0:
                for agent in environ.agents:
                    if len(agent.memory)>environ.batch_size:
                        experience = agent.memory.sample()
                        logger.losses.append(optimize_model(experience, agent.local_net, agent.target_net, agent.optimizer))

        if agent.agents_type == 'analytical':
            break
        
    logger.log_measures(environ)
    return logger.reward, environ.eng.get_average_travel_time(), environ.eng.get_finished_vehicle_count()

def meta_train_loop(meta_train_episodes, pop_agents, pop_envs, args):
    analytic_reward, analytic_reward_std, analytic_avg_travel_time, analytic_throughput, freeflow_throughput = generate_baseline(pop_envs)
    for i in range(meta_train_episodes):
        print("META EPISODE", i)
        results = []
        for agent, env in zip(pop_agents, pop_envs):
        # for agent, env in itertools.zip_longest(pop_agents, pop_envs, fillvalue=pop_envs[0]):
            reward, avg_travel_time, throughput = train(agent, env, args)
            agent.fit_scores.append(reward)
            results.append((reward, avg_travel_time, throughput))
        learning_results = results
        
        print(np.mean([x[0] for x in learning_results]), np.std([x[0] for x in learning_results]), np.mean([x[1] for x in learning_results]), np.mean([x[2] for x in learning_results]))
        print(analytic_reward, analytic_reward_std, analytic_avg_travel_time, analytic_throughput, np.mean(freeflow_throughput), np.std(freeflow_throughput))

        random.shuffle(pop_agents)
        random.shuffle(pop_envs)


def evolve_agents(agents, environs, args):
    for agent in agents:
        # rewards = []
        # for env in environs:
        #     reward, avg_travel_time, throughput = train(agent, env, args, test=True)
        #     rewards.append(reward)
        agent.fitness = np.mean(agent.fit_scores) #if reward is std sort is reveresed

    agents.sort(key=lambda x: x.fitness, reverse=True)
    pop_parents = agents[0:int(len(agents)/2)]
    probs = [1 / x.fitness for x in pop_parents]
    probs = [x / np.sum(probs) for x in probs]
    print(probs)
    
    # # pop_parents = agents[0:int(len(environs)/2)]
    new_population = []
    
    eng = cityflow.Engine(args.path + "0.config", thread_num=8)
    agent_ids = [x for x in eng.get_intersection_ids() if not eng.is_intersection_virtual(x)]
    agent_id = agent_ids[0]

    # while len(new_population) != len(agents):    
    #     # parents = random.sample(pop_parents, 2)
    #     parents = np.random.choice(pop_parents, 2, p=probs, replace=False)
            
    #     child = Hybrid_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id),
    #                          out_roads=eng.get_intersection_out_roads(agent_id), n_states=args.n_states, lr=args.lr, batch_size=args.batch_size)
        
    #     for param1, param2 in zip(parents[0].local_net.named_parameters(), parents[1].local_net.named_parameters()):
    #         name1 = param1[0]
    #         name2 = param2[0]
    #         param1 = param1[1]
    #         param2 = param2[1]

    #         dim = param1.shape[0]
    #         # crossover_point = random.randrange(0, dim)
    #         # new_params = torch.cat((param1[0:crossover_point], param2[crossover_point:]))
            
    #         if random.random() <= 0.1:
    #             #mutation
    #             mutation_point = random.randrange(0, dim)
    #             param1[mutation_point] += np.random.normal(0, 0.2)

    #         beta = parents[0].fitness / (parents[0].fitness + parents[1].fitness)
    #         child_params = dict(child.local_net.named_parameters())
    #         child_params[name1].data.copy_(beta * param1.data + (1-beta) * param2.data)

    #     new_population.append(child)


    child = Hybrid_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id),
                         out_roads=eng.get_intersection_out_roads(agent_id), n_states=args.n_states, lr=args.lr, batch_size=args.batch_size)
    child_params = dict(child.local_net.named_parameters())
    for name, param in child.local_net.named_parameters():
        child_params[name].data.fill_(0)
        
    for parent, weight in zip(pop_parents, probs):
        for name, param in parent.local_net.named_parameters():
            child_params[name].data.copy_(child_params[name].data + param)
    for name, _ in child.local_net.named_parameters():
        child_params[name].data.copy_(child_params[name] / len(pop_parents))
            
    rewards = []
    att = []
    tt = []
    for env in environs:
        reward, avg_travel_time, throughput = train(agent, env, args, test=True)
        rewards.append(reward)
        att.append(avg_travel_time)
        tt.append(throughput)
    print("super agent:", np.mean(rewards), np.mean(att), np.mean(tt))
    
    # while len(new_population) < len(agents):
    #     child = Hybrid_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id),
    #             out_roads=eng.get_intersection_out_roads(agent_id), n_states=args.n_states, lr=args.lr, batch_size=args.batch_size)
    #     child_params = dict(child.local_net.named_parameters())
    #     for name, param in child.local_net.named_parameters():
    #         child_params[name].data.fill_(0)
            
    #     for parent, weight in zip(pop_parents, probs):
    #         for name, param in parent.local_net.named_parameters():
    #             child_params[name].data.copy_(child_params[name].data + param)

    #     for name, _ in child.local_net.named_parameters():
    #         child_params[name].data.copy_(child_params[name] / len(pop_parents))

    #     new_population.append(child)
        
    return agents             

def evolve_environs(environs):
    for env in environs:
        env.fitness = env.analytic_throughput / env.freeflow_throughput
        
    environs.sort(key=lambda x: x.fitness, reverse=False)
    pop_parents = environs[0:int(len(environs))]

    genotypes = []
    new_population = []

    while len(new_population) != len(environs):    
        parents = random.sample(pop_parents, 2)
        crossover_point = random.randrange(0, 11)
        new_genotype = parents[0].genotype[0:crossover_point] + parents[1].genotype[crossover_point:]
        if random.random() <= 0.1:
            #mutation
            mutation_point = random.randrange(0, 11)
            if random.random() > 0.5:
                new_genotype[mutation_point] *= 1.5
            else:
                new_genotype[mutation_point] /= 1.5
        new_population.append(new_genotype)
        
    return new_population

        
if __name__ == '__main__':
    np.random.seed(2)
    random.seed(2)
    args = parse_args()

    pop_agent_size = 10
    pop_envs_size = 10

    genotypes = []
    for _ in range(pop_envs_size):
        genotypes.append(list(np.abs(np.random.normal(0.05, 0.05, size=12))))
    
    pop_envs = populate_envs(pop_envs_size, args, genotypes)
    pop_agents = populate_agents(pop_agent_size, args)
    
    meta_train_episodes = 5
    epochs = 40
    
    for epoch in range(epochs):
        print(epoch)
        meta_train_loop(meta_train_episodes, pop_agents, pop_envs, args)
        # genotypes = evolve_environs(pop_envs)
        # pop_envs = populate_envs(pop_envs_size, args, genotypes)
        pop_agents = evolve_agents(pop_agents, pop_envs, args)
