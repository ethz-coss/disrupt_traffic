import operator
from agents.agent import Agent

class Demand_Agent(Agent):

    def __init__(self, eng, ID='', **kwargs):
        super().__init__(eng, ID)
        self.agents_type = 'demand'


    def choose_act(self, eng, time):
        lanes_count = eng.get_lane_vehicle_count() # quick hack
        phases_priority = {}
        for phase in self.phases.values():
            priority = 0
            for moveID in phase.movements:
                priority += self.movements[moveID].get_demand(lanes_count)

            phases_priority.update({phase.ID : priority})

        return max(phases_priority.items(), key=operator.itemgetter(1))[0]

    def observe(self, veh_distance):
        return None      
