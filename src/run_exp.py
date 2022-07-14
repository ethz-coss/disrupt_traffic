
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

paths = ['2x2', 'hangzhou', 'ny48']
# ds = ['dis1', 'dis2', 'dis3', 'dis4',]
ds = ['1']

for path in paths:
    old_path = '../run_exp_' + path
    results_path = '../run_exp_' + path
    i = 0
    while os.path.exists(results_path):
        results_path = old_path + "(" + str(i) + ")"
        i += 1
    os.mkdir(results_path)
    
    for d in ds:
        os.mkdir(results_path+"/"+d)
        os.system("bsub -n 8 -W 24:00 -R \"select[model==EPYC_7742]\" -R \"select[gpu_model0==NVIDIAGeForceRTX2080Ti]\" -R \"rusage[ngpus_excl_p=1]\" -R \"rusage[mem= 28096]\" python traffic_sim.py --sim_config \"" + "../scenarios/" + path + '/' + d + ".config\"" + " --num_sim_steps 3600 --num_episodes 150 --lr 0.0005 --agents_type hybrid --mode \"train\" --path \"" + results_path + "\"")
