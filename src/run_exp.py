
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation



import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation


paths = ['4x4mount', 'hangzhou', 'ny48', '2x2']
ds = ['1']
methods = ['hybrid', 'presslight', 'analytical', 'fixed', 'demand']


for method in methods:
    for path in paths:
        old_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method + '_1'
        results_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method + '_1'
        i = 0
        while os.path.exists(results_path):
            results_path = old_path + "(" + str(i) + ")"
            i += 1
        os.mkdir(results_path)
        model_path = '../models/' + path +'_' + method + '/reward_q_net.pt'
        
        for d in ds:
            if method == 'hybrid' or method == 'presslight':
                os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config ../scenarios/" + path + '/' + d + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --replay True --num_episodes 1 --load " + model_path  + " --agents_type " + method + " --path " + results_path+"/"+d+ "_" + method + "\"")
            else:
                os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config " + "../scenarios/" + path + '/' + d + ".config" + " --num_sim_steps 3600 --num_episodes 1  --replay True --agents_type " + method + " --path " + results_path+"/"+d+"_" + method + "\"")



# pretrained = ['4x4mount']
# paths = ['4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
# ds = ['dis1', 'dis2', 'dis3', 'dis4']
# method = 'hybrid_load_4x4mount'

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

#     print('pre-' + pretrain)
#     for path in paths:
#         string = path
#         for d in ds:
#             if path == '2x2' and d != '1' and d != 'dis1':
#                 break
#             string += " & " + "{0:.2f}".format(data_dict[(path, d)][0]) + " (" + "{0:.2f}".format(data_dict[(path, d)][1]) + ")"
#         print(string + "\\\\")
            
# paths = ['2x2', '4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
# ds = ['1', 'dis1', 'dis2', 'dis3', 'dis4']
# methods = ['random', 'fixed', 'demand', 'analytical', 'presslight_load', 'hybrid_load']
#printing results
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
#                     with open("../run_exp_" + path + '/' + d + "_" + method + "/logs.txt", "r") as log_file:
#                         data = log_file.read().split("\n")
#                         time = float(data[7].split(":")[1].split("with")[0])
#                         avg_time.append(time)
#                 else:
#                     with open("../run_exp_" + path + '/' + d + "_" + str(i) + "_" + method + "/logs.txt", "r") as log_file:
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

# paths = ['4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
# ds = ['dis1', 'dis2', 'dis3', 'dis4']
# methods = ['hybrid', 'presslight', 'analytical', 'fixed', 'random', 'demand']

# for method in methods:
#     for path in paths:
#         # old_path = '../run_exp_' + path + '_' + method
#         # results_path = '../run_exp_' + path + '_' + method

#         old_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method
#         results_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method
#         i = 0
#         while os.path.exists(results_path):
#             results_path = old_path + "(" + str(i) + ")"
#             i += 1
#         os.mkdir(results_path)

#         model_path = '../models/' + path +'_' + method + '/reward_q_net.pt'
#         # model_paths = ['../models/hangzhou_' + method + '/reward_q_net.pt']
        
#         # for model_path in model_paths:
#         for d in ds:
#             for i in range(10):
#                 if method == 'hybrid' or method == 'presslight':
#                     os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config ../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+ "_" + method + "\"")
#                 else:
#                     os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config " + "../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+"_" + method + "\"")



# paths = ['4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
# ds = ['1']
# methods = ['hybrid', 'presslight', 'analytical', 'fixed', 'random', 'demand']


# for method in methods:
#     for path in paths:
#         # old_path = '../run_exp_' + path + '_' + method
#         # results_path = '../run_exp_' + path + '_' + method

#         old_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method + '_1'
#         results_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method + '_1'
#         i = 0
#         while os.path.exists(results_path):
#             results_path = old_path + "(" + str(i) + ")"
#             i += 1
#         os.mkdir(results_path)

#         model_path = '../models/' + path +'_' + method + '/reward_q_net.pt'
        
#         for d in ds:
#             if method == 'hybrid' or method == 'presslight':
#                 os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config ../scenarios/" + path + '/' + d + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + " --path " + results_path+"/"+d+ "_" + method + "\"")
#             else:
#                 os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config " + "../scenarios/" + path + '/' + d + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + " --path " + results_path+"/"+d+"_" + method + "\"")



# paths = ['2x2']
# ds = ['dis1']
# methods = ['hybrid', 'presslight', 'analytical', 'fixed', 'random', 'demand']

# for method in methods:
#     for path in paths:
#         # old_path = '../run_exp_' + path + '_' + method
#         # results_path = '../run_exp_' + path + '_' + method

#         old_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method
#         results_path = '../../../../scratch/mkorecki/run_exp_' + path + '_' + method 
#         i = 0
#         while os.path.exists(results_path):
#             results_path = old_path + "(" + str(i) + ")"
#             i += 1
#         os.mkdir(results_path)

#         model_path = '../models/' + path +'_' + method + '/reward_q_net.pt'
        
#         for d in ds:
#             for i in range(10):
#                 if method == 'hybrid' or method == 'presslight':
#                     os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config ../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --eps_start 0 --lr 0.0005 --mode test --num_episodes 1 --load " + model_path  + " --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+ "_" + method + "\"")
#                 else:
#                     os.system("sbatch -n 8 --wrap \"python traffic_sim.py --sim_config " + "../scenarios/" + path + '/' + d + "_" + str(i) + ".config" + " --num_sim_steps 3600 --num_episodes 1 --agents_type " + method + " --path " + results_path+"/"+d+"_"+str(i)+"_" + method + "\"")
