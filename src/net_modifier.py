import cityflow

import argparse
import os
import random
import json
import queue
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default='../scenarios/4x4mount/',  type=str, help="the relative directory of the sim files")
    parser.add_argument("--sim_config", default='../scenarios/4x4mount/1.config',  type=str, help="the relative path to the simulation config file")
    parser.add_argument("--roadnet", default='../scenarios/4x4mount/roadnet_m_m.json',  type=str, help="the relative path to the simulation roadnet file")
    parser.add_argument("--flow", default='../scenarios/4x4mount/anon_4_4_700_0.3.json',  type=str, help="the relative path to the flow file")

    return parser.parse_args()



"""
an array of helper functions for modifying/disrupting the flow, vehicles speeds and network topology as well as drawing the network
"""


args = parse_args()

num_points = [0, 1, 2, 3, 4, 5]
div = {1:2, 2:4, 3:5, 4:6, 5:7}




vehicle = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 11.11,
    "headwayTime": 1.5
}

def generate_route(turning_ratios, sim_time, road_link_dict, road_len_dict):
    route = []
    options = ['turn_left', 'turn_right', 'go_straight']
    total_time = 0
    
    current_road = random.choice(list(road_link_dict))
    time_to_cross = (road_len_dict[current_road] / 20)
    total_time += time_to_cross
    route.append(current_road)
    
    while total_time < sim_time:
        direction = random.choices(list(options), weights=turning_ratios)[0]
        current_road = road_link_dict[current_road][direction]
        time_to_cross = (road_len_dict[current_road] / 20)
        total_time += time_to_cross
        route.append(current_road)

    return route

def generate_flow_file(path, roadnet, num_vehs, turning_ratios, sim_time):
    with open(roadnet, "r") as roadnet_file:
        roadnet_data = json.load(roadnet_file)

    flow_data = []
    
    road_len_dict = {}
    road_link_dict = {}
    
    for road in roadnet_data['roads']:
        point1 = road['points'][0]
        point2 = road['points'][1]

        diff_x = point1['x'] - point2['x']
        diff_y = point1['y'] - point2['y']

        length = np.sqrt(diff_x**2 + diff_y**2)
        road_len_dict[road['id']] = length

    for intersection in roadnet_data['intersections']:
        for link in intersection['roadLinks']:
            if link["startRoad"] not in road_link_dict.keys():
                road_link_dict[link["startRoad"]] = {link["type"] : link["endRoad"]}
            else:
                road_link_dict[link["startRoad"]].update({link["type"] : link["endRoad"]})

    random.seed(datetime.now())
    for veh in range(num_vehs):
        veh_data = {'vehicle' : vehicle}
        veh_data['interval'] = 1
        veh_data['startTime'] = 0
        veh_data['endTime'] = 0
        route = generate_route(turning_ratios, sim_time, road_link_dict, road_len_dict)
        veh_data['route'] = route
        flow_data.append(veh_data)

    save_path = path + "flow.json"
    i = 0
    while os.path.exists(save_path):
        save_path = path + "flow(" + str(i) + ").json"
        i += 1
        
    with open(save_path, 'w') as flow:
        json.dump(flow_data, flow)


# order of turning is [left, right, straight]
# generate_flow_file(args.dir, args.roadnet, 120, [1, 1, 1], 1800)



        

def disrupt_veh_speed(args):
    with open(args.flow, "r") as flow_file:
        data = json.load(flow_file)

        mu, sigma = 15, 5
        lower, upper = 9, 20
        
        for vehicle in data:
            vehicle['vehicle']['maxSpeed'] =  stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=15, scale=5)

    with open(args.dir + "flow_m.json", "w") as flow_file:
        json.dump(data, flow_file)

        
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)
        for road in data['roads']:
            road['maxSpeed'] = stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=15, scale=5)
            for lane in road['lanes']:
                lane['maxSpeed'] = road['maxSpeed']

            
    with open(args.dir + "roadnet_m_m_m.json", "w") as roadnet_file:
        json.dump(data, roadnet_file)
        
def disrupt_road_topology(args):
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)
        intersections = {}
        points = {}

        for road in data['roads']:
            flag = False
            point1 = road['points'][0]
            point2 = road['points'][1]

            diff_x = point1['x'] - point2['x']
            diff_y = point1['y'] - point2['y']


            num_point = random.sample(num_points, 1)[0]

            if random.random() > 0.5: change = 50
            else: change = -50


            if (road['endIntersection'], road['startIntersection']) in points.keys():
                num_point = points.get((road['endIntersection'], road['startIntersection']))
            points.update({(road['startIntersection'], road['endIntersection']) : num_point})

            if (road['endIntersection'], road['startIntersection']) in intersections.keys():
                change = intersections.get((road['endIntersection'], road['startIntersection']))
                flag = True
            intersections.update({(road['startIntersection'], road['endIntersection']) : change})

            new_points = []

            if not (diff_x == 0 and diff_y == 0):
                for i in range(num_point):
                    if diff_x == 0:
                        new_point = {'x' : point1['x'] + change, 'y' : min(point1['y'], point2['y']) + (i+1) * abs(point1['y'] - point2['y']) / div[num_point]}
                    if diff_y == 0:
                        new_point = {'x' : min(point1['x'], point2['x']) + (i+1) * abs(point1['x'] - point2['x']) / div[num_point], 'y' : point1['y'] + change}
                    new_points.append(new_point)
                    change *= -1

                if new_points:
                    print(point1, point2)
                    print(new_points)
                    if flag:
                        new_points.reverse()
                    road['points'][1:1] = new_points

    with open(args.dir + "roadnet_m_m.json", "w") as roadnet_file:
        json.dump(data, roadnet_file)


def draw_tikzpicture(args):
    scale = 1/100
    latex_string = "\\begin{tikzpicture}\n"
    
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)
        
        for road in data['roads']:
            for i in range(len(road['points']) - 1):
                latex_string += "\draw " + "(" + str(road['points'][i]['x'] * scale) + "," + str(road['points'][i]['y'] * scale) + ")" + " -- " + "(" + str(road['points'][i+1]['x'] * scale) + "," + str(road['points'][i+1]['y'] * scale) + ")" ";"


        for intersection in data['intersections']:
            if not intersection['virtual']:
                latex_string += "\\filldraw [blue]" + "(" + str(intersection['point']['x'] * scale) + ", " + str(intersection['point']['y'] * scale) + ")" + "circle (8pt);"

    latex_string += "\n\end{tikzpicture}"
    with open("../tikzpicture.txt", "w") as save_file:
        save_file.write(latex_string)



def get_road_lengths(args):
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)

    road_lengths = []
    for road in data['roads']:
        road_length = 0
        for i in range(len(road['points']) - 1):
            road_length += np.sqrt((road['points'][i+1]['x'] - road['points'][i]['x'])**2 + (road['points'][i+1]['y'] - road['points'][i]['y'])**2)
        road_lengths.append(road_length)

    print(np.mean(road_lengths), np.std(road_lengths))


def get_flow_rates(args):
    with open(args.flow, "r") as flow_file:
        data = json.load(flow_file)

    length = 3600
    vehs_time_array = np.zeros(length)
    last_time = -1

    starting_points = {}
    
    for elem in data:
        if elem['startTime'] >= length:
            continue
        if elem['route'][0] not in starting_points.keys():
            starting_points.update({elem['route'][0] :  np.zeros(length)})
        for i in range(elem['startTime'], elem['endTime']+1):
            vehs_time_array[i] += 1
            starting_points[elem['route'][0]][i] += 1

    arr_rate = np.zeros(length)
    for i in range(length):
        arr_rate[i] = np.sum(vehs_time_array[0:i+1]) / (i+1)

    arr_points_sum = []
    for val in starting_points.values():
        print(np.sum(val), np.mean(val), np.var(val))
        arr_points_sum.append(np.sum(val))

    print(np.mean(arr_points_sum), np.var(arr_points_sum))
    print(np.sum(vehs_time_array), np.mean(vehs_time_array), np.var(vehs_time_array))




def draw_mfd(ax, name, path1, path2, path3):
    init = 0
    sample_rate = 20


    # if name == "I":
    #     with open('../4x4_100/mfds/4x4_100_config100_1_analytical+/mfd.pickle', "rb") as f:
    #         mfd_data = pickle.load(f)
    #         mfd_data = mfd_data[init::sample_rate]

    #         density = [np.mean(x[0]) for x in mfd_data]
    #         flow = [np.mean(x[1]) for x in mfd_data]
    #         density = [x*1000 for x in density]
    #         ax.scatter(density, flow, s=2, c='tab:green', label='Analytic+')
            
    with open(path1, "rb") as f:
        #hybrid
        mfd_data = pickle.load(f)
        mfd_data = mfd_data[init::sample_rate]

        if name == "NY196" or name == "Jinan" or name == "Hangzhou":
            density = [np.mean([x for x in x[0] if x>0.01]) for x in mfd_data]
            flow = [np.mean([x for x in x[1] if x>0.01]) for x in mfd_data]
        elif name == "NY16":
            density = [np.mean([x for x in x[0] if x>0.01]) for x in mfd_data]
            flow = [np.mean(x[1]) for x in mfd_data]
        else:
            density = [np.mean(x[0]) for x in mfd_data]
            flow = [np.mean(x[1]) for x in mfd_data]
        density = [x*1000 for x in density]
        flow = [x*3600 for x in flow]

        ax.scatter(density, flow, s=2, c='tab:cyan', label='GuidedLight')

    with open(path2, "rb") as f:
        #analytical
        mfd_data = pickle.load(f)
        mfd_data = mfd_data[init::sample_rate]

        if name == "NY196" or name == "Jinan" or name == "Hangzhou":
            density = [np.mean([x for x in x[0] if x>0.01]) for x in mfd_data]
            flow = [np.mean([x for x in x[1] if x>0.01]) for x in mfd_data]
        elif name == "NY16":
            density = [np.mean([x for x in x[0] if x>0.01]) for x in mfd_data]
            flow = [np.mean(x[1]) for x in mfd_data]
        else:
            density = [np.mean(x[0]) for x in mfd_data]
            flow = [np.mean(x[1]) for x in mfd_data]
        density = [x*1000 for x in density]
        flow = [x*3600 for x in flow]
        ax.scatter(density, flow, s=2, c='r', label='Analytic', alpha=0.5)

    with open(path3, "rb") as f:
        #presslight
        mfd_data = pickle.load(f)
        mfd_data = mfd_data[init::sample_rate]

        # if name == "I":
        #     density = [np.mean(x[0]) for x in mfd_data]
        #     flow = [np.mean(x[1]) for x in mfd_data]
        # elif name == "NY196":
        if name == "NY196" or name == "Jinan" or name == "Hangzhou":
            density = [np.mean([x for x in x[0] if x>0.01]) for x in mfd_data]
            flow = [np.mean([x for x in x[1] if x>0.01]) for x in mfd_data]
        elif name == "NY16":
            density = [np.mean([x for x in x[0] if x>0.01]) for x in mfd_data]
            flow = [np.mean(x[1]) for x in mfd_data]
        else:
            density = [np.mean(x[0]) for x in mfd_data]
            flow = [np.mean(x[1]) for x in mfd_data]
            
        density = [x*1000 for x in density]
        flow = [x*3600 for x in flow]
        # else:
        #     density = [x[0] for x in mfd_data]
        #     flow = [x[1] for x in mfd_data]
        
        ax.scatter(density, flow, s=2, c='tab:purple', label='PressLight', alpha=0.5)
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 400)
        
    # ax.set(xlabel='Density (vehicles/m)')
    # ax.set(ylabel='Flow (vehicles/s)')
    ax.set_title(name)
        
# draw_tikzpicture(args)
def mfd():
    names = ['I', 'II', 'III', 'IV', 'NY16', 'NY196']
    paths1 = ['../4x4_100/mfds/4x4_100_config100_1_hybrid_load/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_2_hybrid_load/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_3_hybrid_load/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_4_hybrid_load/mfd.pickle',
              '../4x4_100/mfds/ny16_config1_hybrid_load/mfd.pickle',
              '../4x4_100/mfds/ny196_config1_hybrid_load/mfd.pickle']
    paths2 = ['../4x4_100/mfds/4x4_100_config100_1_analytical/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_2_analytical/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_3_analytical/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_4_analytical/mfd.pickle',
              '../4x4_100/mfds/ny16_config1_analytical/mfd.pickle',
              '../4x4_100/mfds/ny196_config1_analytical/mfd.pickle']
    paths3 = ['../4x4_100/mfds/4x4_100_config100_1_presslight_load/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_2_presslight_load/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_3_presslight_load/mfd.pickle',
              '../4x4_100/mfds/4x4_100_config100_4_presslight_load/mfd.pickle',
              '../4x4_100/mfds/ny16_config1_presslight_load/mfd.pickle',
              '../4x4_100/mfds/ny196_config1_presslight_load/mfd.pickle']

    fig, axs = plt.subplots(3, 2, figsize=(4,6), constrained_layout=True)
    i = 0
    j = 0
    
    for name, path1, path2, path3 in zip(names, paths1, paths2, paths3):
        draw_mfd(axs[i, j], name, path1, path2, path3)
        if i != 2:
            axs[i,j].xaxis.set_visible(False)
        if j != 0:
            axs[i,j].yaxis.set_visible(False)

        if j < 1:
            j += 1
        else:
            j = 0
            i += 1


        
    fig.supxlabel('Avg. Density (vehicles/km)')
    fig.supylabel('Avg. Flow (vehicles/h)')
    axs[2,1].legend(loc="lower right", prop={'size': 7})

    plt.show()


# mfd()


def decrease_lanes_length(args, factor=0.3):
    with open(args.roadnet, "r") as roadnet_file:
        data = json.load(roadnet_file)
        intersections = {}
        points = {}


        for intersection in data['intersections']:
            intersection['point']['x'] *= factor
            intersection['point']['y'] *= factor
                    
        for road in data['roads']:
            point1 = road['points'][0]
            point2 = road['points'][1]

            
            # diff_x = point1['x'] - point2['x']
            road['points'][0]['x'] *= factor
            road['points'][1]['x'] *= factor

            # diff_y = point1['y'] - point2['y']
            road['points'][0]['y'] *= factor
            road['points'][1]['y'] *= factor

                
    with open(args.dir + "roadnet_" + str(factor) + ".json", "w") as roadnet_file:
        json.dump(data, roadnet_file)


# decrease_lanes_length(args, factor=0.5)


draw_tikzpicture(args)
