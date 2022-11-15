import json
import os


def get_network(config):
    # map roads to intersections they connect to

    neighbors = {}
    flows = {}
    roads = {}
    # get the roadnet file
    with open(config, 'r') as cfg_f:
        config = json.load(cfg_f)
        roadnet_file = config['roadnetFile']
        flow_file = config['flowFile']
        scenario_dir = config['dir']
    # parse roadnet file
    with open(os.path.join(scenario_dir, roadnet_file), 'r') as roadnet_f:
        roadnet_config = json.load(roadnet_f)
        intersections = roadnet_config['intersections']
        roadnet = roadnet_config['roads']

    road_intersection_map = {}
    # map roads to connected intersections
    for intersection in intersections:
        if not intersection['virtual']:
            for road in intersection['roads']:
                intersection_set = road_intersection_map.setdefault(road, set())
                intersection_set.add(intersection['id'])

    # build intersection connections
    for road_id, intersection_set in road_intersection_map.items():
        if len(intersection_set) == 2:
            source = road_id.split('_')[1:3]
            for i, intersection_id in enumerate(intersection_set):
                if intersection_id.split('_')[1:3] == source:
                    other_intersection_id = list(intersection_set)[abs(1-i)]
                    source = neighbors.setdefault(
                        intersection_id, {'upstream': {}, 'downstream': {}})
                    # source['downstream'].add(other_intersection_id)
                    source['downstream'].setdefault(other_intersection_id, []).append(road_id)
                    sink = neighbors.setdefault(other_intersection_id, {
                                                'upstream': {}, 'downstream': {}})
                    # sink['upstream'].add(intersection_id)
                    source['upstream'].setdefault(intersection_id, []).append(road_id)

        elif len(intersection_set) > 2:  # this shouldn't happen...
            raise ValueError
            
    for road in roadnet:
        road_data = roads.setdefault(road['id'], {})
        road_data['num_lanes'] = len(road['lanes'])
        road_data['max_speed'] = road['lanes'][0]['maxSpeed']
        length = 0
        for p1, p2 in zip(road['points'][:-1], road['points'][1:]):
            length += ((p2['x']-p1['x'])**2 +
                        (p2['y']-p1['y'])**2)**0.5
        road_data['length'] = length

        
    # parse flow file
    with open(os.path.join(scenario_dir, flow_file), 'r') as flow_f:
        flow_data = json.load(flow_f)
        for flow_id, flow in enumerate(flow_data):
            flow_id = f"flow_{flow_id}"
            flows[flow_id] = {'route': flow['route']}
            flows[flow_id]['routelength'] = sum([roads[road_id]['length'] for road_id in flows[flow_id]['route']])
            flows[flow_id]['freeflow_time'] = sum([roads[road_id]['length']/roads[road_id]['max_speed'] for road_id in flows[flow_id]['route']])

    return neighbors, roads, flows



if __name__ == "__main__":
    config = '../scenarios/2x2/1.config'
    network, roads, flows = get_network(config)
