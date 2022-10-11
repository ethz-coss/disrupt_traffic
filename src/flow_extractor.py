import cityflow

import os
import json
import numpy as np

paths = ['../scenarios/4x4mount/', '../scenarios/ny48/', '../scenarios/ny48double/', '../scenarios/ny48triple/']
ds = ['dis1', 'dis2', 'dis3', 'dis4']

for path in paths:
    print(path)
    for d in ds:
        veh_num = []
        veh_times = []
        routes = []
        routes_std = []
        with open(path + d + '.config', "r") as cfg_file:
            data = json.load(cfg_file)
            flow = data['flowFile']
            roadnet = data['roadnetFile']

            with open(path+roadnet, "r") as roadnet_file:
                roadnet = json.load(roadnet_file)

            flow_routes = []
            for flow in range(0, 10):
                with open(path+d+"/flow_disrupted_"+str(flow)+".json", "r") as flow_file:
                    data = json.load(flow_file)
                    veh_num.append(len(data))

                    for veh in data:
                        route = veh['route']
                        start_time = veh['startTime']
                        end_time = veh['endTime']

                        flow_routes.append(len(route))
                        if start_time == end_time:
                            veh_times.append(start_time)
                        elif end_time == -1:
                            times = list(range(start_time, 3600))
                            veh_times.append(times)
                        else:
                            print('this is not good')
                routes.append(np.mean(flow_routes))
                routes_std.append(np.std(flow_routes))

        print(d, 'number of vehicles, median, rate of arrival:', veh_num[0], np.median(veh_times), veh_num[0]/3600)
        print(d, 'length of path(in links), mean, std, sum: {:.2f} ({:.2f})'.format(np.mean(routes), np.mean(routes_std)), np.sum(routes))

# path = '../scenarios/hangzhou/'
# flow = path + 'flow.json'

# with open(flow, "r") as flow_file:
#     data = json.load(flow_file)
#     data += data
    
# with open(path + "/flow_doubled.json", "w") as jsonFile:
#     json.dump(data, jsonFile)


