
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

path = '../scenarios/RRL_TLC/flow_1x5/'
ds = ['0', '01', '001', '005', '0005']

# old_path = '../run_exp'
# results_path = '../run_exp'
# i = 0
# while os.path.exists(results_path):
#     results_path = old_path + "(" + str(i) + ")"
#     i += 1
# os.mkdir(results_path)

# for d in ds:
#     os.mkdir(results_path+"/"+d)
#     for i in range(10):
#         with open(path + d + '.config', "r") as config:
#             data = json.load(config)
#             data['flowFile'] = data['flowFile'].split('_')[0] + '_' + str(i+1) + '.json'
#         with open(path + d + '.config', "w") as config:
#             json.dump(data, config)
            
#         os.system("python3 traffic_sim.py --sim_config \"" + path + d + ".config\"" + " --num_sim_steps 3600 --num_episodes 1 --lr 0.0005 --agents_type analytical --path \"" + results_path+"/"+d + "\"")

        # os.system("python traffic_sim.py --sim_config \"" + path + d + ".config\"" + " --num_sim_steps 3600 --num_episodes 1 --lr 0.0005 --agents_type hybrid --load \"../RRL_TLC_config_hybrid/time_q_net.pt\" --mode \"test\" --eps_start 0 --eps_end 0 --path \"" + results_path+"/"+ d + "\"")





path = '../results/run_exp_hangzhou240_360/'
results = {}

for folder in os.listdir(path):
    avg_times_list = []
    if os.path.isdir(path + folder):
        print(folder)
        for f in os.listdir(path + folder):
            if os.path.isdir(path + folder + '/' + f):
                logs = path + folder + '/' + f + '/logs.txt'
                with open(logs, "r") as f:
                    data = f.read().split("\n")
                    analytic_avg = float(data[7].split(":")[1].split("with")[0])
                    avg_times_list.append(analytic_avg)
        print("{:.1f}".format(max(avg_times_list)), "{:.1f}".format(min(avg_times_list)), "{:.1f}".format(sum(avg_times_list)/float(len(avg_times_list))))
        results.update({str(folder) : avg_times_list})

# # Hangzhou
generalight_means = [380.0, 402.0, 432.2, 493.5, 622.5]
generalight_min = [376.3, 393.6, 409.4, 443.5, 579.8]
generalight_max = [385.1, 410.2, 458.7, 582.0, 652.2]
generalight_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(generalight_means, generalight_max, generalight_min)]))

colight_means = [383.9, 456.4, 537.8, 624.1, 880.2]
colight_min = [368.7, 387.7, 402.7, 517.1, 726.2]
colight_max = [425.6, 603.0, 721.5, 887.9, 1040.4]
colight_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(colight_means, colight_max, colight_min)]))

metalight_means = [497.2, 545.6, 637.8, 767.4, 861.1]
metalight_min = [475.0, 512.0, 580.5, 719.9, 814.2]
metalight_max = [514.4, 587.3, 671.6, 843.3, 937.9] 
metalight_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(metalight_means, metalight_max, metalight_min)]))

free_means = [331.8, 332.2, 329.9, 338.6, 349.9]
free_min = [331.4, 332.5, 329.2, 339.1, 350.1]
free_max = [332.3, 331.9, 330.2, 338.2, 349.7]
free_err = np.rot90(([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(free_means, free_max, free_min)]))

# Atlanta
# generalight_means = [180.3, 181.6, 253.9, 329.0, 554.2]
# generalight_min = [165.1, 167.7, 232.8, 363.3, 409.0]
# generalight_max = [194.5, 200.0, 283.4, 298.5, 646.6]
# generalight_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(generalight_means, generalight_max, generalight_min)]))

# generalight_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(generalight_means, generalight_max, generalight_min)]))

# colight_means = [433.3, 435.2,  579.1, 797.5, 849.2]
# colight_min = [195.5, 187.1, 256.6, 421.9, 526.5]
# colight_max = [580.7, 593.6, 785.4, 903.4, 911.1]
# colight_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(colight_means, colight_max, colight_min)]))

# metalight_means = [196.5, 209.6, 339.9, 429.9, 819.2]
# metalight_min = [176.8, 177.0, 279.9, 386.3, 616.5]
# metalight_max = [206.9, 258.4, 463.1, 583.3, 991.2] 
# metalight_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(metalight_means, metalight_max, metalight_min)]))

# free_means = [59.4, 59.4, 58.5 , 60.2, 52.3]
# free_min = [59.3,  59.4, 58.5, 60.2, 52.3]
# free_max = [59.4, 59.4, 58.6, 60.3, 52.3]
# free_err = np.rot90(np.asarray([[gmax-gmean, gmean-gmin] for gmean, gmax, gmin in zip(free_means, free_max, free_min)]))



ds = ['0', '0005', '001', '005', '01']
data = []
for i in ds:
    data.append(results[i])
        
fig, ax = plt.subplots()
# ax.boxplot(data)
    
err = np.rot90(np.asarray([[max(d)-np.mean(d), np.mean(d)-min(d)] for d in data]))
mins = [min(d) for d in data]
maxs = [max(d) for d in data]

trans1 = ax.transData + ScaledTranslation(-15/72, 0, fig.dpi_scale_trans)
trans2 = ax.transData + ScaledTranslation(-5/72, 0, fig.dpi_scale_trans)
trans3 = ax.transData + ScaledTranslation(+5/72, 0, fig.dpi_scale_trans)
trans4 = ax.transData + ScaledTranslation(+15/72, 0, fig.dpi_scale_trans)

ds = ['0', '0.005', '0.01', '0.05', '0.1']

new_gen = [(z-x)/(x+z) for x, z in zip(generalight_means, [np.mean(y) for y in data])]
new_gen_min = [(z-x)/(x+z) for x, z in zip(generalight_min, mins)]
new_gen_max = [(z-x)/(x+z) for x, z in zip(generalight_max, maxs)]
new_gen_err = np.rot90(np.asarray([[abs(gmax-gmean), abs(gmean-gmin)] for gmean, gmax, gmin in zip(new_gen, new_gen_max, new_gen_min)]))

new_col = [(z-x)/(x+z) for x, z in zip(colight_means, [np.mean(y) for y in data])]
new_col_min = [(z-x)/(x+z) for x, z in zip(colight_min, mins)]
new_col_max = [(z-x)/(x+z) for x, z in zip(colight_max, maxs)]
new_col_err = np.rot90(np.asarray([[abs(gmax-gmean), abs(gmean-gmin)] for gmean, gmax, gmin in zip(new_col, new_col_max, new_col_min)]))

new_met = [(z-x)/(x+z) for x, z in zip(metalight_means, [np.mean(y) for y in data])]
new_met_min = [(z-x)/(x+z) for x, z in zip(metalight_min, mins)]
new_met_max = [(z-x)/(x+z) for x, z in zip(metalight_max, maxs)]
new_met_err = np.rot90(np.asarray([[abs(gmax-gmean), abs(gmean-gmin)] for gmean, gmax, gmin in zip(new_met, new_met_max, new_met_min)]))




# ax.errorbar(np.asarray(ds), new_gen, yerr=new_gen_err, marker='x', linestyle='dashed', transform=trans2, label='Generalight', capsize=3, ms=3, c ='xkcd:bubblegum')
# ax.errorbar(np.asarray(ds), new_col, yerr=new_col_err, marker='o', linestyle='dotted', transform=trans4, label='Colight', capsize=3, ms=3, c='xkcd:steel')
# ax.errorbar(np.asarray(ds), new_met, yerr=new_met_err, marker='D', linestyle='dashdot', transform=trans3, label='Metalight', capsize=3, ms=3, c='xkcd:pale purple')

ax.errorbar(np.asarray(ds), [np.mean(y) for y in data], yerr=err, marker='+', linestyle='solid', transform=trans1, label='Analytic+', capsize=3, ms=3, c='xkcd:aqua')
ax.errorbar(np.asarray(ds), generalight_means, yerr=generalight_err, marker='x', linestyle='dashed', transform=trans2, label='Generalight', capsize=3, ms=3, c ='xkcd:bubblegum')
ax.errorbar(np.asarray(ds), colight_means, yerr=colight_err, marker='o', linestyle='dotted', transform=trans4, label='Colight', capsize=3, ms=3, c='xkcd:steel')
ax.errorbar(np.asarray(ds), metalight_means, yerr=metalight_err, marker='D', linestyle='dashdot', transform=trans3, label='Metalight', capsize=3, ms=3, c='xkcd:pale purple')

ax.set_xlabel('Wasserstein Distance From Training Scenario', fontsize=10)
ax.set_ylabel("Avg. Travel Time(s)", fontsize=10)

ax.legend(loc='upper left', fontsize=7)
ax.set(xlim=(-25/72, 4+(25/72)))
plt.title('Hangzhou')

fig.set_size_inches(5, 2.5)
plt.tight_layout()
plt.savefig("hangzhou.pdf", format="pdf", bbox_inches="tight")
    
