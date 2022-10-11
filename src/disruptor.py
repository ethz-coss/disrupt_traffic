import cityflow

import argparse
import os
import random
import json
import queue

from intersection import Movement, Phase
from learning_agent import Learning_Agent

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default='../scenarios/2x2/',  type=str, help="the relative directory of the sim files")
    parser.add_argument("--sim_config", default='../scenarios/2x2/1.config',  type=str, help="the relative path to the simulation config file")
    parser.add_argument("--flow", default='../scenarios/2x2/flow.json',  type=str, help="the relative path to the simulation flow file")
    parser.add_argument("--dist_roads", default=1,  type=int, help="number of roads to be disrupted")
    parser.add_argument("--detour", default=1,  type=int, help="number of available detours")
    parser.add_argument("--sample", default=1,  type=int, help="number of samples with a given disruption")

    return parser.parse_args()



args = parse_args()
eng = cityflow.Engine(args.sim_config, thread_num=8)

agent_ids = [x for x in eng.get_intersection_ids() if not eng.is_intersection_virtual(x)]
intersections = []
inter_dict = {}

for agent_id in agent_ids:
    new_agent = Learning_Agent(eng, ID=agent_id, in_roads=eng.get_intersection_in_roads(agent_id), out_roads=eng.get_intersection_out_roads(agent_id))
    intersections.append(new_agent)
    inter_dict.update({agent_id : new_agent})

roads = set()
roads_dict = dict()
for intersect in intersections:
    for road in intersect.in_roads:
        roads.add(road)
        if road in roads_dict.keys():
            roads_dict.update({road : (roads_dict[road], intersect.ID)})
        else:
            roads_dict.update({road : intersect.ID})
    for road in intersect.out_roads:
        roads.add(road)
        if road in roads_dict.keys():
            roads_dict.update({road : (intersect.ID, roads_dict[road])})
        else:
            roads_dict.update({road : intersect.ID})
            
roads = list(roads)
removed_roads = []

for road in roads:
    if type(roads_dict[road]) is not tuple:
        removed_roads.append(road)

for removed in removed_roads:
    roads.remove(removed)
        

            
def generate_alt_route(source, goal, removed, path, disrupted_roads):
    """
    finds alternative route, returns [list of roads] 
    use bfs?
    """
    q = queue.Queue()
    q.put((source, path))
    visited = set()
    alt_routes = []
    
    while not q.empty():
        source, path = q.get()
        visited.add(source)
        # print(q.empty(), alt_routes, path)
        if alt_routes == [] or len(path) <= len(alt_routes[0]):
            for road in inter_dict[source].out_roads:
                new_path = path.copy()
                if road not in disrupted_roads and type(roads_dict[road]) == tuple and roads_dict[road][1] not in visited:
                    new_path.append(road)
                    current = roads_dict[road][1]
                    if current == goal:
                        alt_routes.append(new_path)
                    else:
                        q.put((current, new_path))
        else:
            return alt_routes
    return alt_routes



sample = 0
disruption_memory = {}

while args.sample > sample:
    disrupted_roads = []
    alt_routes_dict = {}
    while len(disrupted_roads) < args.dist_roads:
        new_road = random.sample(roads, args.dist_roads-len(disrupted_roads))

        if tuple(new_road) in disruption_memory.keys():
            continue
        disrupted_roads += new_road
        print(disrupted_roads)

        disruption_memory.update({tuple(new_road) : 1})

        with open(args.flow, "r") as flow_file:
            data = json.load(flow_file)


        for road in disrupted_roads:
            if road in alt_routes_dict.keys():
                alt_route = alt_routes_dict[road]
            else:
                alt_route = generate_alt_route(roads_dict[road][0], roads_dict[road][1], road, [], set(disrupted_roads))
                alt_routes_dict[road] = alt_route

        for road in disrupted_roads:
            if road in alt_routes_dict and len(alt_routes_dict[road]) != args.detour:
                disrupted_roads.remove(road)
                alt_routes_dict.pop(road)

    for elem in data:
        route = elem["route"]
        for road in disrupted_roads:
            if road in set(route):
                idx = route.index(road)
                route[idx:idx+1] = random.sample(alt_routes_dict[road], 1)[0]
                elem["route"] = route

    for k,v in zip(alt_routes_dict.keys(), alt_routes_dict.values()):
        print(k, v)


    with open(args.dir + "/flow_disrupted_" + str(sample) + ".json", "w") as jsonFile:
        json.dump(data, jsonFile)
    sample += 1

