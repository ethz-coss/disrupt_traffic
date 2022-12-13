import json
import os
from shapely.geometry import Point, LineString

def parallel_point_shift(p1, p2, h):
    (x1,y1) = p1['x'], p1['y']
    (x2,y2) = p2['x'], p2['y']

    if h:
        theta = np.arctan((x1-x2)/(y1-y2+1e-9))
        if y1 < y2:
            x1 -= h*np.cos(theta)
            y1 += h*np.sin(theta)
            x2 -= h*np.cos(theta)
            y2 += h*np.sin(theta)
        else:
            x1 += h*np.cos(theta)
            y1 -= h*np.sin(theta)
            x2 += h*np.cos(theta)
            y2 -= h*np.sin(theta)

    return ((x1,y1), (x2,y2))

def get_network(config, h=0, keep_virtual=False):
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
        if not intersection['virtual'] or keep_virtual:
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

        points = road['points']
        # for each point excluding the start point and end point, shift point based on line to next point
        transformed_points = []
        # get pairs of points on the line segment
        for p1,p2 in zip(points[0:], points[1:]):
            (x1,y1), (x2,y2) = parallel_point_shift(p1,p2,h)
            transformed_points.append((x1,y1))

        transformed_points.append((x2,y2))
        points = transformed_points
        road_data['geometry'] = LineString([p for p in points])
        road_data['length'] = road_data['geometry'].length

        
    # parse flow file
    with open(os.path.join(scenario_dir, flow_file), 'r') as flow_f:
        flow_data = json.load(flow_f)
        for flow_id, flow in enumerate(flow_data):
            flow_id = f"flow_{flow_id}"
            flows[flow_id] = {'route': flow['route']}
            flows[flow_id]['routelength'] = sum([roads[road_id]['length'] for road_id in flows[flow_id]['route']])
            flows[flow_id]['freeflow_time'] = sum([roads[road_id]['length']/roads[road_id]['max_speed'] for road_id in flows[flow_id]['route']])
            
            start_time = flow['startTime']
            end_time = flow['endTime']
            interval = flow['interval']
            if end_time == -1: end_time = 3600
            if end_time-start_time==0:
                rate = False
            else:
                rate = True
            if rate:
                flows[flow_id]['demand'] = 1/interval*(end_time-start_time) # per hour
            else:
                flows[flow_id]['demand'] = interval # per hour

    return neighbors, roads, flows


if __name__ == "__main__":
    config = '../scenarios/2x2/1.config'
    network, roads, flows = get_network(config)
