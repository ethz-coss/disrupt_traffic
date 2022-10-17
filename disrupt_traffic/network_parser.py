import json
import os


def get_node_connections(config):
    # map roads to intersections they connect to
    roads = {}
    neighbors = {}

    # get the roadnet file
    with open(config, 'r') as cfg_f:
        config = json.load(cfg_f)
        roadnet_file = config['roadnetFile']
        scenario_dir = config['dir']
    # parse roadnet file
    with open(os.path.join(scenario_dir, roadnet_file), 'r') as roadnet_f:
        roadnet = json.load(roadnet_f)
        intersections = roadnet['intersections']

    # map roads to connected intersections
    for intersection in intersections:
        if not intersection['virtual']:
            for road in intersection['roads']:
                intersection_set = roads.setdefault(road, set())
                intersection_set.add(intersection['id'])

    # build intersection connections
    for road_id, intersection_set in roads.items():
        if len(intersection_set) == 2:
            source = road_id.split('_')[1:3]
            for i, intersection_id in enumerate(intersection_set):
                if intersection_id.split('_')[1:3] == source:
                    other_intersection_id = list(intersection_set)[abs(1-i)]
                    source = neighbors.setdefault(
                        intersection_id, {'upstream': set(), 'downstream': set()})
                    source['downstream'].add(other_intersection_id)
                    sink = neighbors.setdefault(other_intersection_id, {
                                                'upstream': set(), 'downstream': set()})
                    sink['upstream'].add(intersection_id)
        elif len(intersection_set) > 2:  # this shouldn't happen...
            raise ValueError

    return neighbors


if __name__ == "__main__":
    config = '../scenarios/2x2/1.config'
    neighbors = get_node_connections(config)
