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

    def __init__(self, env, observation_builder):
        self.max_prediction_depth = 0
        self.env = env
        self.observation_builder = observation_builder
        self.memory_SC = [[[dict() for x in range(env.width)] for y in range(env.height)] for z in range(env.number_of_agents)]
        self.state_machine_actions_history = [0 for x in range(self.env.number_of_agents)]



    def reset(self):
        self.memory_SC = [[[dict() for x in range(self.env.width)] for y in range(self.env.height)] for z in range(self.env.number_of_agents)]
        self.state_machine_actions_history = [0 for x in range(self.env.number_of_agents)]
        memory = self.read_memory()
        self.memory = np.ones((self.env.number_of_agents, self.env.width, self.env.height), dtype=np.int16)

        # update SM
        for agent in self.env.agents:
            self.update_conflict_data(memory, agent.handle)



    def read_memory(self):
        return self.observation_builder.read_memory()



    def tests(self, observations_full, max_prediction_depth, state_machine_actions):

        self.memory = observations_full
        # update agents if not moving
        for item in state_machine_actions:
            if state_machine_actions[item] == 1:
                self.state_machine_actions_history[item] += 1
            else:
                self.state_machine_actions_history[item] = 0

        # if an agent is at a fork
        # call update of paths
        # if an agent is not moving from ten time steps
        for agent in self.env.agents:
            if agent.position in self.observation_builder.forks_coords or self.state_machine_actions_history[agent.handle] >= 10:
                self.observation_builder.update_path_memory_at_recalculate(agent.handle)
                self.state_machine_actions_history[agent.handle] = 0
                # read memory
                #self.memory = self.read_memory()
                # update SM
                self.update_conflict_data(self.memory, agent.handle)

        # run test battery
        final = {}
        for a in self.env.agents:
            # for every conflicting agent
            # run below tests
            #
            final1 = {}

            # check if agent is entering a conflict zone
            # find current position
            current_pos_a = a.position
            # if a valid position
            if not current_pos_a is None:
                final1["A_entering_CZ"] = 0
                final1["B_entering_CZ"] = 0
                final1["B_already_in_CZ"] = 0
                final1["ca"] = {}

                # find next position
                val_cur_pos = self.memory[0][a.handle][current_pos_a[0]][current_pos_a[1]]

                # in memory map check surrounding to find next cell
                if val_cur_pos < self.memory[0][a.handle][current_pos_a[0]-1][current_pos_a[1]]:
                    next_pos_a = (current_pos_a[0]-1, current_pos_a[1])
                elif val_cur_pos < self.memory[0][a.handle][current_pos_a[0]][current_pos_a[1]-1]:
                    next_pos_a = (current_pos_a[0], current_pos_a[1]-1)
                elif current_pos_a[0] +1 < self.env.width and current_pos_a[1] + 1 < self.env.height:
                    if val_cur_pos < self.memory[0][a.handle][current_pos_a[0]][current_pos_a[1]+1]:
                        next_pos_a = (current_pos_a[0], current_pos_a[1]+1)
                    elif val_cur_pos < self.memory[0][a.handle][current_pos_a[0] + 1][current_pos_a[1]]:
                        next_pos_a = (current_pos_a[0] + 1, current_pos_a[1])
                else:
                    next_pos_a = current_pos_a

                # check that next cell
                #   if next cell has a conflicting agent
                if len(self.memory_SC[a.handle][next_pos_a[0]][next_pos_a[1]]) > 0:

                    final_ca = {}
                    #   check if another agent is in conflicting span
                    for k in self.memory_SC[a.handle][next_pos_a[0]][next_pos_a[1]].keys():
                        final1["A_entering_CZ"] = 1

                        ca_position = self.env.agents[k].position
                        if not ca_position is None:
                            for item in self.memory_SC[a.handle][next_pos_a[0]][next_pos_a[1]][k]:
                                if item[0] == ca_position[0] and item[1] == ca_position[1]:
                                    final1["B_already_in_CZ"] = 1
                                    final_ca[k] = k
                                else:

                                    val_cur_pos_ca = self.memory[0][k][ca_position[0]][ca_position[1]]

                                    # in memory map check surrounding to find next cell
                                    if val_cur_pos_ca < self.memory[0][k][ca_position[0] - 1][ca_position[1]]:
                                        next_pos_ca = (ca_position[0] - 1, ca_position[1])
                                    elif val_cur_pos_ca < self.memory[0][k][ca_position[0]][ca_position[1] - 1]:
                                        next_pos_ca = (ca_position[0], ca_position[1]-1)
                                    elif ca_position[0] + 1 < self.env.width and ca_position[1] + 1 < self.env.height:
                                        if val_cur_pos_ca < self.memory[0][k][ca_position[0]][ca_position[1] + 1]:
                                            next_pos_ca = (ca_position[0], ca_position[1]+1)
                                        elif val_cur_pos_ca < self.memory[0][k][ca_position[0] + 1][ca_position[1]]:
                                            next_pos_ca = (ca_position[0] + 1, ca_position[1])
                                    else:
                                        next_pos_ca = ca_position

                                    # check if the next position of this agent is one in the current list of conflicting keys
                                    if item[0] == next_pos_ca[0] and item[1] == next_pos_ca[1]:
                                        final1["B_entering_CZ"] = 1
                                        final_ca[k] = k

                    final1["ca"] = final_ca

            else:
                final1["A_entering_CZ"] = 0
                final1["B_entering_CZ"] = 0
                final1["B_already_in_CZ"] = 0
                final1["ca"] = {}

            final[a.handle] = final1

        # give input for the state machine
        return final




    def update_conflict_data(self, memory, handle):
        for b in range(len(memory[0])):
            if handle != b:

                # individual occupancy map
                conflict_map_handle = np.asarray([[1 if item > 0 else 0 for item in row] for row in memory[0][handle]])
                conflict_map_b = np.asarray([[1 if item > 0 else 0 for item in row] for row in memory[0][b]])

                # conflict map comparing two agents
                conflict_map = conflict_map_handle + conflict_map_b
                conflict_map = np.asarray([[1 if item > 1 else 0 for item in row] for row in conflict_map])
                #print(handle, b,"\n", conflict_map)


                # direction
                direction_map_handle = np.multiply(memory[0][handle], conflict_map)
                direction_map_b = np.multiply(memory[0][b], conflict_map)
                #print(handle,b,"\n", direction_map_handle, direction_map_b)

                # time value for conflict
                conflict_check_id = np.argwhere(direction_map_handle > 0)

                #conflict_check_id_b = np.argwhere(direction_map_b > 0)

                #print("ID's where a conflict may happen",handle,b,"\n", conflict_check_id)
                if len(conflict_check_id) > 0:

                    val_handle_1 = memory[0][handle][conflict_check_id[0][0]][conflict_check_id[0][1]]
                    val_handle_2 = memory[0][handle][conflict_check_id[-1][0]][conflict_check_id[-1][1]]
                    val_b_1 = memory[0][b][conflict_check_id[0][0]][conflict_check_id[0][1]]
                    val_b_2 = memory[0][b][conflict_check_id[-1][0]][conflict_check_id[-1][1]]

                    if (val_handle_2 - val_handle_1 >= 0 and val_b_2 - val_b_1 <= 0) \
                                or (val_handle_2 - val_handle_1 <= 0 and val_b_2 - val_b_1 >= 0):
                        #print("same direction")
                        #pass
                        #else:
                        #print("check further for time overlap")

                        val_handle_max = np.max(direction_map_handle)
                        val_b_max = np.max(direction_map_b)
                        val_handle_min = np.min([np.nonzero(direction_map_handle)])
                        val_b_min = np.min([np.nonzero(direction_map_b)])

                        #print(val_handle_max, val_handle_min, val_b_max, val_b_min)

                        if val_handle_min > val_b_max > val_handle_max or val_b_min > val_handle_max > val_b_max\
                                or val_b_min < val_b_max < val_b_max or val_handle_min < val_b_max < val_handle_max:
                            #print("no time conflict")
                            pass
                        else:
                            #print("time conflict")
                            # write these coordinates as the conflict

                            # for both the agents
                            # for all the coordinates
                            # write

                            #conflict
                            for item in conflict_check_id:

                                dict_temp = self.memory_SC[handle][item[0]][item[1]]
                                dict_temp[b] = conflict_check_id
                                self.memory_SC[handle][item[0]][item[1]] = dict_temp

                                dict_temp = self.memory_SC[b][item[0]][item[1]]
                                dict_temp[handle] = conflict_check_id
                                self.memory_SC[b][item[0]][item[1]] = dict_temp


















    def tests1(self, observations_full, max_prediction_depth):

        obs = self.read_memory()

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
                final2["B_entering_CZ"] = self.entering_cz(ca, a, observations)
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
        return 1 if observations[a]["per_agent_occupancy_in_time"][b][0] > 0 else 0
        #return 1 if observations[a]["conflict"][0] > 0 else 0


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

        if not virtual_position1 is None and not virtual_position2 is None:
            if agent1.initial_position == agent2.initial_position \
                    and agent1.initial_direction == agent2.initial_direction \
                    and agent1.target == agent2.target \
                    and (abs(virtual_position1[0] - virtual_position2[0]) <= 2 or abs(virtual_position1[1] - virtual_position2[1]) <= 2):
                return "Same"
            else:
                return "Opposite"
        return "Same"

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
                    second_layer[i] = 1 if observations[b]["per_agent_occupancy_in_time"][a][i] > 0 else 0
                    i -= 1
                i = ts[0]
                while i < self.max_prediction_depth:
                    second_layer[i] = 1 if observations[b]["per_agent_occupancy_in_time"][a][i] > 0 else 0
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
