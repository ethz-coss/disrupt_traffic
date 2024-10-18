
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

PATHROOT = os.environ.get('SCRATCH', '../../../../scratch/mkorecki/')
PATHROOT = '../'
pretrained = ['4x4mount']
paths = ['4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
ds = ['dis1', 'dis2', 'dis3', 'dis4']
method = 'hybrid_load_4x4mount'

numtrials = 50

# for pretrain in pretrained:
#     data_dict = {}
#     print(pretrain)
#     for path in paths:
#         print(path)
#         for d in ds:
#             avg_time = []
#             for i in range(10):
#                 with open("../pretrained_hybrid/" + path + '_pretrained_' + pretrain + "/" + d + "_" + str(i) + "_" + method + "/logs.txt", "r") as log_file:
#                     data = log_file.read().split("\n")
#                     time = float(data[7].split(":")[1].split("with")[0])
#                     avg_time.append(time)  

#             data_dict.update({(path, d) : (np.mean(avg_time), np.std(avg_time))})
#             # print(d, "{0:.2f}".format(np.mean(avg_time)), "{0:.2f}".format(np.std(avg_time)))

    # print('pre-' + pretrain)
    # for path in paths:
    #     string = path
    #     for d in ds:
    #         if path == '2x2' and d != '1' and d != 'dis1':
    #             break
    #         string += " & " + "{0:.2f}".format(data_dict[(path, d)][0]) + " (" + "{0:.2f}".format(data_dict[(path, d)][1]) + ")"
    #     print(string + "\\\\")
            
paths = ['2x2', '4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
ds = ['1', 'dis1', 'dis2', 'dis3', 'dis4']
methods = ['random', 'fixed', 'demand', 'analytical', 'presslight_load', 'hybrid_load']
# printing results
# for path in paths:
#     data_dict = {}
#     print(path)
#     for d in ds:
#         if path == '2x2' and d != '1' and d != 'dis1':
#             break
#         for method in methods:
#             avg_time = []
#             for i in range(0, 10):
#                 if d == '1' and i > 0:
#                     break
#                 if d == '1':
#                     with open("../runs/runs_" + path + '/' + d + "_" + method + "/logs.txt", "r") as log_file:
#                         data = log_file.read().split("\n")
#                         time = float(data[7].split(":")[1].split("with")[0])
#                         avg_time.append(time)
#                 else:
#                     with open("../runs/runs_" + path + '/' + d + "_" + str(i) + "_" + method + "/logs.txt", "r") as log_file:
#                         data = log_file.read().split("\n")
#                         time = float(data[7].split(":")[1].split("with")[0])
#                         avg_time.append(time)

#             data_dict.update({(method, d) : (np.mean(avg_time), np.std(avg_time))})
#             # print(d, method, np.mean(avg_time), np.std(avg_time))

#     table_names = ['Random', 'Fixed', 'Demand', 'Analytic+', 'Presslight', 'Hybrid']
#     for method, name in zip(methods, table_names):
#         string = name
#         for d in ds:
#             if path == '2x2' and d != '1' and d != 'dis1':
#                 break
#             string += " & " + "{0:.2f}".format(data_dict[(method, d)][0]) + " (" + "{0:.2f}".format(data_dict[(method, d)][1]) + ")"
#         print(string + "\\\\")
        
#running the experiments

# paths = ['4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48quad']
# paths = ['ny48quad']
# ds = ['dis1', 'dis2', 'dis3', 'dis4', 'dis24', 'dis48']
# ds = ['dis24', 'dis48']
# methods = ['hybrid', 'presslight', 'analytical', 'fixed', 'random', 'demand']

# for method in methods:
#     for path in paths:
#         calls = []
#         # results_path = '../runs_' + path + '_' + method
#         # results_path = '../runs_' + path + '_' + method

#         results_path = os.path.join(PATHROOT, f'runs')

#         model_path = '../models/' + path +'_' + method + '/reward_q_net.pt'
#         # model_paths = ['../models/hangzhou_' + method + '/reward_q_net.pt']
        
#         # for model_path in model_paths:
#         for d in ds:
#             if d in ['dis24', 'dis48'] and 'ny48' not in path:
#                 break
#             for i in range(10):
#                 if method == 'hybrid' or method == 'presslight':
#                     calls.append("python runner.py --sim_config ../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + f" --path {results_path}")
#                     # os.system("sbatch -n 8 --time=8:00:00 --wrap \"python runner.py --sim_config ../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+ "_" + method + "\"")
#                 else:
#                     calls.append("python runner.py --sim_config " + "../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + f" --path {results_path}")

#         #             os.system("sbatch -n 8 --time=8:00:00 --wrap \"python runner.py --sim_config " + "../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+"_" + method + "\"")
#         if calls:
#             pycalls = "\n".join(calls)
#             os.system(f"""sbatch -n 8 --time=8:00:00 --wrap '{pycalls}'""")


# paths = ['2x2', '4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
# # paths = ['2x2']
# ds = ['1']
# methods = ['hybrid', 'presslight', 'analytical', 'fixed', 'random', 'demand']


# for method in methods:
#     for path in paths:
#         calls = []

#         # results_path = '../runs_' + path + '_' + method
#         # results_path = '../runs_' + path + '_' + method

#         results_path = os.path.join(PATHROOT, f'runs')
#         # os.mkdir(results_path)

#         model_path = '../models/' + path +'_' + method + '/reward_q_net.pt'
        
#         for d in ds:
#             for i in range(numtrials, numtrials*2): #range(numtrials):
#                 if method == 'hybrid' or method == 'presslight':
#                     calls.append("python runner.py --sim_config ../scenarios/" + path + f'/{d}.config' + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + f" --path {results_path} --ID {i} --seed {i}")
#                 else:
#                     calls.append("python runner.py --sim_config " + "../scenarios/" + path + f'/{d}.config' + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + f" --path {results_path} --ID {i} --seed {i}")
#         pycalls = "\n".join(calls)
#         os.system(f"""sbatch -n 8 --time=8:00:00 --wrap '{pycalls}'""")


# paths = ['2x2']
# ds = ['dis1']
# methods = ['hybrid', 'presslight', 'analytical', 'fixed', 'random', 'demand']

# for method in methods:
#     for path in paths:
#         calls = []
#         # results_path = '../runs_' + path + '_' + method
#         # results_path = '../runs_' + path + '_' + method
#         results_path = os.path.join(PATHROOT, f'runs')

#         # results_path = '../test_results/runs_' + path + '_' + method
#         # results_path = '../test_results/runs_' + path + '_' + method 


#         model_path = '../models/' + path +'_' + method + '/reward_q_net.pt'
        
#         for d in ds:
#             for i in range(10):
#                 if method == 'hybrid' or method == 'presslight':
#                     # os.system("sbatch -n 8 --time=8:00:00 --wrap \'python runner.py --sim_config ../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+ "_" + method + "/\'")
#                     calls.append("python runner.py --sim_config ../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + f" --path {results_path}")
#                 else:
#                     # os.system("sbatch -n 8 --time=8:00:00 --wrap \'python runner.py --sim_config " + "../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+"_" + method + "/\'")
#                     calls.append("python runner.py --sim_config " + "../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + f" --path {results_path}")
#         pycalls = "\n".join(calls)
#         os.system(f"""sbatch -n 8 --time=8:00:00 --wrap '{pycalls}'""")

## PRETRAINED RUNS

paths = ['4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48quad']
paths = ['4x4']

# paths = ['ny48quad']
ds = ['dis1', 'dis2', 'dis3', 'dis4', 'dis24', 'dis48']
methods = ['hybrid', 'analytical']
pretrain_methods = ['4x4', '4x4mount']

all_calls = []
for method in (methods+pretrain_methods):
    for path in paths:
        calls = []

        # results_path = '../runs_' + path + '_' + method
        # results_path = '../runs_' + path + '_' + method

        results_path = os.path.join(PATHROOT, f'runs/pretrained_hybrid')
        os.makedirs(results_path, exist_ok=True)

        if method in pretrain_methods:
            _meth = 'hybrid'
            model_path = f'../models/{method}_{_meth}/reward_q_net.pt'
        else:    
            model_path = f'../models/{path}_{method}/reward_q_net.pt'
        
        for d in ds:
            if d in ['dis24', 'dis48'] and 'ny48' not in path:
                break
            for i in range(10): 
                if method in (['hybrid', 'presslight'] + pretrain_methods):
                    agent_type = method
                    if method in pretrain_methods:
                        agent_type = 'hybrid'
                    # os.system("sbatch -n 8 --time=8:00:00 --wrap \"python runner.py --sim_config ../scenarios/" + path + '/' + d + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + " --path " + results_path+"/"+d+ "_" + method + "\"")
                    command = "python runner.py --sim_config ../scenarios/" + path + f'/{d}_{i}.config' + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + f" --agents_type {agent_type}"  + f" --path {results_path}{'_pretrain_'+str(method) if method in pretrain_methods else''} --seed {i}"
                else:
                    # os.system("sbatch -n 8 --time=8:00:00 --wrap \"python runner.py --sim_config " + "../scenarios/" + path + '/' + d + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + " --path " + results_path+"/"+d+"_" + method + "\"")
                    command = "python runner.py --sim_config ../scenarios/" + path + f'/{d}_{i}.config' + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + f" --path {results_path} --ID {i} --seed {i}"
                calls.append(command)
                all_calls.append(command)

        pycalls = "\n".join(calls)
        os.system(f"""sbatch -n 8 --time=8:00:00 --wrap '{pycalls}'""")
