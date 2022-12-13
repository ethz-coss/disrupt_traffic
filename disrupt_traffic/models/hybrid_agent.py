from models.dqn import DQN, device
import numpy as np
import torch
import random
import operator


class Hybrid(DQN):

    def act(self, state, epsilon=0, **kwargs):
        """
        generates the action to be taken by the agent
        :param net_local: the neural network used in the decision making process
        :param state: the current state of the intersection, given by observe
        :param eps: the epsilon value used in the epsilon greedy learing
        """
        if epsilon > np.random.random():
            if epsilon > np.random.random():
                action = np.random.choice(self.num_actions)
            else:
                agent = kwargs.get('agent')
                time = agent.env.time
                eng = agent.env.eng
                agent.update_clear_green_time(time, eng)
                agent.update_priority_idx(time)

                phases_priority = {}
                for phase in agent.phases.values():
                    movements = [
                        x for x in phase.movements if x not in agent.clearing_phase.movements]
                    phase_prioirty = 0
                    for moveID in movements:
                        phase_prioirty += agent.movements[moveID].priority

                    phases_priority.update({phase.ID: phase_prioirty})
                action = max(phases_priority.items(), key=operator.itemgetter(1))[0]
        else:
            state = state.unsqueeze(0)
            self.net_local.eval()
            with torch.no_grad():
                # Predict action Q-values from state
                action_probs = self.net_local(state)
                # Take best action
            self.net_local.train()
            action = action_probs.max(1)[1].item()

        return action
