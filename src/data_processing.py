
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

pretrained = ['4x4mount']
paths = ['4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
ds = ['dis1', 'dis2', 'dis3', 'dis4']
method = 'hybrid_load_4x4mount'

for pretrain in pretrained:
    data_dict_pretrained = {}
    for path in paths:
        print(path)
        for d in ds:
            avg_time = []
            for i in range(10):
                with open("../pretrained_hybrid/" + path + '_pretrained_' + pretrain + "/" + d + "_" + str(i) + "_" + method + "/logs.txt", "r") as log_file:
                    data = log_file.read().split("\n")
                    time = float(data[7].split(":")[1].split("with")[0])
                    avg_time.append(time)  

            data_dict_pretrained.update({(path, d) : (np.mean(avg_time), np.std(avg_time))})

            
paths = ['2x2', '4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
# ds = ['1', 'dis1', 'dis2', 'dis3', 'dis4']
ds = ['dis1', 'dis2', 'dis3', 'dis4']
methods = ['random', 'fixed', 'demand', 'analytical', 'presslight', 'hybrid']

data_dict = {}
for path in paths:
    print(path)
    for d in ds:
        if path == '2x2' and d != '1' and d != 'dis1':
            break
        for method in methods:
            avg_time = []
            for i in range(0, 10):
                if d == '1' and i > 0:
                    break
                if d == '1':
                    with open(f"../run_exp_{path}_{method}/{path}/{d}" + "_" + method + "/logs.txt", "r") as log_file:
                    # with open("../run_exp_" + path + '/' + d + "_" + method + "/logs.txt", "r") as log_file:
                        data = log_file.read().split("\n")
                        time = float(data[7].split(":")[1].split("with")[0])
                        avg_time.append(time)
                else:
                    load = ''
                    if method in ['presslight', 'hybrid']:
                        load = '_load'
                    fp = f"../runs/run_exp_{path}_{method}/{path}{load}/{d}_{i}_{method}/logs.txt"
                    if not os.path.exists(fp): continue
                    with open(fp, "r") as log_file:
                    # with open("../run_exp_" + path + '/' + d + "_" + str(i) + "_" + method + "/logs.txt", "r") as log_file:
                        data = log_file.read().split("\n")
                        time = float(data[7].split(":")[1].split("with")[0])
                        avg_time.append(time)

            data_dict.update({(path, method, d) : (np.mean(avg_time), np.std(avg_time))})
    

x_2x2names = ['0.0625'] 

# paths = ['2x2', '4x4mount', 'hangzhou', 'ny48', 'ny48double', 'ny48triple']
path_names = ['2x2', '4x4', 'Hangzhou', 'NY48', 'NY48 double', 'NY48 triple']

# ds = ['1', 'dis1', 'dis2', 'dis3', 'dis4']
# methods = ['random', 'fixed', 'demand', 'analytical', 'presslight_load', 'hybrid_load']
names = ['random', 'fixed', 'demand', 'analytical+', 'presslight', 'hybrid']

avg_times = []
errs = []
colors = ['xkcd:light blue green', 'xkcd:azure', 'xkcd:steel', 'xkcd:aqua', 'xkcd:bubblegum', 'xkcd:pale purple']
markers = ["o", "^", "2", "x", "d", "*"]
lines = [(0,(5,10)),(0, (1,10)), "dashdot", "solid", "dotted", "dashed"]

for path, path_name in zip(paths, path_names):
    fig, ax = plt.subplots()
    trans1 = ax.transData + ScaledTranslation(-7.5/72, 0, fig.dpi_scale_trans)
    trans2 = ax.transData + ScaledTranslation(-5/72, 0, fig.dpi_scale_trans)
    trans3 = ax.transData + ScaledTranslation(-2.5/72, 0, fig.dpi_scale_trans)
    trans4 = ax.transData + ScaledTranslation(+2.5/72, 0, fig.dpi_scale_trans)
    trans5 = ax.transData + ScaledTranslation(+5/72, 0, fig.dpi_scale_trans)
    trans6 = ax.transData + ScaledTranslation(+7.5/72, 0, fig.dpi_scale_trans)
    transforms = [trans1, trans2, trans3, trans4, trans5, trans6]
    x_names = ['0.0625', '0.125', '0.1875', '0.25']

    for method, color, line, marker, trans, name in zip(methods, colors, lines, markers, transforms, names):
        avg_times = []
        errs = []
        for d in ds:
            if path == '2x2' and d != '1' and d != 'dis1':
                x_names = x_2x2names
                break
            avg_times += [data_dict[(path, method, d)][0]]
            errs += [(data_dict[(path, method, d)][1], data_dict[(path, method, d)][1])]

        ax.errorbar(np.asarray(x_names),
            np.asarray(avg_times),
            yerr=np.rot90(np.asarray(errs)),
            marker=marker,
            linestyle=line,
            transform=trans,
            label=name,
            capsize=3,
            ms=3,
            c=color
            )

    ax.set_xlabel('Ratio of disrupted links', fontsize=10)
    ax.set_ylabel("Avg. Travel Time(s)", fontsize=10)

    if path != '2x2':
        ax.set(xlim=(-25/72, 4+(25/72)))
    else:
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0),
              ncol=6, fontsize=6)
    plt.title(path_name)


    fig.set_size_inches(7.5, 2.5)
    plt.savefig(path+".pdf", format="pdf", bbox_inches="tight")
    

