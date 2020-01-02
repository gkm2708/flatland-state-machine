import collections
import numpy as np
from flatland.envs.agent_utils import RailAgentStatus



class TestBattery():
    """
    Build graph observations.
    """

    Node = collections.namedtuple('Node',
                                  'cell_position '  # Cell position (x, y)
                                  'agent_direction '  # Direction with which the agent arrived in this node
                                  'is_target')  # Whether agent's target is in this cell

    def __init__(self, env):
        self.max_prediction_depth = 0
        self.env = env

    def tests(self, observations_full, max_prediction_depth):

        observations = observations_full["preprocessed_observation"]
        self.max_prediction_depth = max_prediction_depth
        self.cells_sequence = observations_full["cells_sequence"]

        final = {}
        for a in range(0, len(self.env.agents)):
            # for every conflicting agent
            # run below tests
            #
            final1 = {}
            #
            for ca in observations[a]["conflicting_agents"]:
                #
                final2 = {}
                #
                # Which Conflicting agent ?
                #
                final2["conflict_agent_id"] = ca
                #
                # is the conflict agent faster/slower?
                # find more info
                #
                final2["speed"] = self.speed(a, ca)
                #
                # is the current agent at the entry-border of the conflict zone (CZ)(per other conflict agent)?
                # if first bit of per_agent_occupancy_in_time for agent is greater than 0
                #
                final2["A_entering_CZ"] = self.entering_cz(a, ca, observations)
                final2["B_entering_CZ"] = self.entering_cz(a, ca, observations)
                #
                # is the current agent in the CZ (per other conflict agent)?
                #
                final2["A_already_in_CZ"] = self.check_active_conflict(a, ca, observations)
                final2["B_already_in_CZ"] = self.check_active_conflict(ca, a, observations)
                #
                # is the conflict agent moving in the opposite direction?
                #
                final2["B_direction_in_CZ"] = self.check_direction(a, ca)
                #
                # Which Agent ?
                # final1["agent_id"] = a
                #
                #if observations[a]["path"][ca][0][0] != 0 or observations[a]["path"][ca][0][1] != 0:
                #    final1[ca] = final2
                final1[ca] = final2
                #
            final[a] = final1

        final3 = {}
        for a in range(0, len(self.env.agents)):
            final3[a] = self.check_target(a, observations)

        final4 = {}
        final4["main"] = final
        final4["target"] = final3

        return final4

    # Test 1 and 2 : Check if agent is entering conflict zone with another agent
    def entering_cz(self, a, b, observations):
        return 0 if observations[a]["path"][a][0][0] == 0 and observations[a]["path"][a][0][1] == 0 else \
            1 if observations[a]["per_agent_occupancy_in_time"][b][0] > 0 else 0

    def check_target(self,a, observations):
        return 1 if observations[a]["path"][a][0][0] == 0 and observations[a]["path"][a][0][1] == 0 else 0

    # Test 3 : Check if agent and conflicting agent are in opposing direction
    def check_direction(self, handle1, handle2):
        """
        Checks whether one agent is (probably) following the other one.
        :param handle1:
        :param handle2:
        :return:
        """
        agent1 = self.env.agents[handle1]
        agent2 = self.env.agents[handle2]
        virtual_position1 = agent1.initial_position if agent1.status == RailAgentStatus.READY_TO_DEPART else agent1.position
        virtual_position2 = agent2.initial_position if agent2.status == RailAgentStatus.READY_TO_DEPART else agent2.position

        if agent1.initial_position == agent2.initial_position \
                and agent1.initial_direction == agent2.initial_direction \
                and agent1.target == agent2.target \
                and (abs(virtual_position1[0] - virtual_position2[0]) <= 2 or abs(virtual_position1[1] - virtual_position2[1]) <= 2):
            return "Same"
        else:
            return "Opposite"

    # Test 4 and 5: Check if conflict of one agent with another already started
    def check_active_conflict(self, a, b, observations):

        agents = self.env.agents
        second_layer = np.zeros(self.max_prediction_depth, dtype=int) # Same size as occupancy

        if agents[a].status is RailAgentStatus.ACTIVE and agents[b].status is RailAgentStatus.ACTIVE:

            ts = [x for x, y in enumerate(self.cells_sequence[b]) if y[0] == agents[a].position[0] and y[1] == agents[a].position[1]]

            # Set to 1 conflict span which was already entered by some agent - fill left side and right side of ts
            if len(ts) > 0:
                i = ts[0] # Since the previous returns a list of ts
                while 0 <= i < self.max_prediction_depth:
                    second_layer[i] = 1 if observations[b]["occupancy_new"][i] > 0 else 0
                    i -= 1
                i = ts[0]
                while i < self.max_prediction_depth:
                    second_layer[i] = 1 if observations[b]["occupancy_new"][i] > 0 else 0
                    i += 1

        return int(second_layer[0])
        # default return from conflict

    # Test 6: Compare speed of any two agents
    def speed(self, a, b):
        agents = self.env.agents
        return "High" \
            if agents[a].speed_data["speed"] < agents[b].speed_data["speed"] \
            else "Equal" if agents[a].speed_data["speed"] == agents[b].speed_data["speed"] \
            else "Low"
