import cityflow

import os
import json
import numpy as np

# path = '../scenarios/RRL_TLC/flow_4x4/'
# ds = ['0', '0.1', '0.01', '0.05', '0.005']


        
# for d in ds:
#     veh_num = []
#     veh_times = []
#     for flow in os.scandir(path+d):
#         if flow.is_file():
#             with open(flow, "r") as flow_file:
#                 data = json.load(flow_file)
#                 veh_num.append(len(data))

#                 for veh in data:
#                     route = veh['route']
#                     start_time = veh['startTime']
#                     end_time = veh['endTime']

#                     if start_time == end_time:
#                         veh_times.append(start_time)
#                     else:
#                         print('this is not good')
                    
#     print(d, veh_num, np.mean(veh_times), np.median(veh_times))

path = '../scenarios/ny48/'
flow = path + 'flow.json'

with open(flow, "r") as flow_file:
    data = json.load(flow_file)
    data += data
    
with open(path + "/flow_doubled.json", "w") as jsonFile:
    json.dump(data, jsonFile)


