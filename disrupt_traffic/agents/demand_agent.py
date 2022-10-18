import operator
from agents.agent import Agent

class Demand_Agent(Agent):

    def __init__(self, eng, ID='', **kwargs):
        super().__init__(eng, ID)
        self.agents_type = 'demand'


    def act(self, lanes_count):
        phases_priority = {}
        for phase in self.phases.values():
            priority = 0
            for moveID in phase.movements:
                priority += self.movements[moveID].get_demand(lanes_count)

            phases_priority.update({phase.ID : priority})

        return self.phases[max(phases_priority.items(), key=operator.itemgetter(1))[0]]
      
