"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset (i.e. at the beginning of a new episode), to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""

import collections
from typing import Optional, List, Dict, Tuple
import queue
import numpy as np
from collections import defaultdict
import math

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_valid_move_actions_
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position, distance_on_rail, position_to_coordinate
from flatland.utils.ordered_set import OrderedSet


from src.priority import assign_priority


class GraphObsForRailEnv(ObservationBuilder):
    """
    Build graph observations.
    """

    Node = collections.namedtuple('Node',
                                  'cell_position '  # Cell position (x, y)
                                  'agent_direction '  # Direction with which the agent arrived in this node
                                  'is_target')  # Whether agent's target is in this cell


    def __init__(self, predictor, bfs_depth):
        super(GraphObsForRailEnv, self).__init__()
        self.predictor = predictor
        self.bfs_depth = bfs_depth
        self.max_prediction_depth = 0
        self.prediction_dict = {}  # Dict handle : list of tuples representing prediction steps
        self.predicted_pos = {}  # Dict ts : int_pos_list
        self.predicted_pos_list = {} # Dict handle : int_pos_list
        self.predicted_pos_coord = {}  # Dict ts : coord_pos_list
        self.predicted_dir = {}  # Dict ts : dir (float)
        self.num_active_agents = 0
        self.cells_sequence = None
        self.forks_coords = None


    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            # Use set_env available in PredictionBuilder (parent class)
            self.predictor.set_env(self.env)


    def reset(self):
        """
        Inherited method used for pre computations.
        :return: 
        """

        self.forks_coords = self._find_forks()


    def get_many(self, handles: Optional[List[int]] = None) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles: 
        :return: 
        """

        self.num_active_agents = 0
        for a in self.env.agents:
            if a.status == RailAgentStatus.ACTIVE:
                self.num_active_agents += 1
        self.prediction_dict = self.predictor.get()
        # Useful to check if occupancy is correctly computed
        self.cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)

        if self.prediction_dict:
            self.max_prediction_depth = self.predictor.max_depth
            for t in range(self.max_prediction_depth):
                pos_list = []
                dir_list = []
                for a in handles:
                    if self.prediction_dict[a] is None:
                        continue
                    pos_list.append(self.prediction_dict[a][t][1:3])
                    dir_list.append(self.prediction_dict[a][t][3])
                self.predicted_pos_coord.update({t: pos_list})
                self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                self.predicted_dir.update({t: dir_list})

            for a in range(len(self.env.agents)):
                pos_list = []
                for ts in range(self.max_prediction_depth):
                    pos_list.append(self.predicted_pos[ts][a])  # Use int positions
                self.predicted_pos_list.update({a: pos_list})

        observations = {}
        for a in handles:
            observations[a] = self.get(a)

        preprocessed_observation = {}
        for a in handles:
            preprocessed_observation[a] = self.preprocess_satate(observations, a)

        state_machine_triggers = self.test_battery(preprocessed_observation)

        return state_machine_triggers


    # TODO Optimize considering that I don't need obs for those agents who don't have to pick actions
    def get(self, handle: int = 0) -> {}:
        """
        Returns obs for one agent, obs are a single array of concatenated values representing:
        - occupancy of next prediction_depth cells,
        - agent priority/speed,
        - number of malfunctioning agents (encountered),
        - number of agents that are ready to depart (encountered).
        :param handle:
        :return:
        """

        agents = self.env.agents
        agent = agents[handle]

        # Occupancy
        occupancy, conflicting_agents = self._fill_occupancy(handle)
        #overlapping_paths, conflict_vector = self._fill_occupancy_1(handle)

        # Augment occupancy with another one-hot encoded layer:
        # 1 if this cell is overlapping and the conflict span was already entered by some other agent
        second_layer = np.zeros(self.max_prediction_depth, dtype=int) # Same size as occupancy
        for ca in conflicting_agents:
            if ca != handle:
                # Find ts when conflict occurred
                # Find index/ts for conflict
                ts = [x for x, y in enumerate(self.cells_sequence[handle]) if y[0] == agents[ca].position[0] and y[1] == agents[ca].position[1]]
                # Set to 1 conflict span which was already entered by some agent - fill left side and right side of ts
                if len(ts) > 0:
                    i = ts[0] # Since the previous returns a list of ts
                    while 0 <= i < self.max_prediction_depth:
                        second_layer[i] = 1 if occupancy[i] > 0 else 0
                        i -= 1
                    i = ts[0]
                    while i < self.max_prediction_depth:
                        second_layer[i] = 1 if occupancy[i] > 0 else 0
                        i += 1

        ret = {}
        path = np.zeros((len(self.predicted_pos_coord[0])
                         ,len(self.predicted_pos_coord)
                         ,self.predicted_pos_coord[0][0].shape[0]))

        for index in range(0,len(self.predicted_pos_coord[0])):
            for index1 in range(0,len(self.predicted_pos_coord)):
                path[index][index1][0] = self.predicted_pos_coord[index1][index][0]
                path[index][index1][1] = self.predicted_pos_coord[index1][index][1]


        ret["path"] = path
        ret["overlap"] = self._compute_overlapping_paths(handle)
        ret["direction"] = self.absolute_dir_dict(path)

        ret["conflicting_agents"] = conflicting_agents
        ret["occupancy_old"] = occupancy
        ret["conflict"] = second_layer


        ########################################################


        # Bifurcation points, one-hot encoded layer of predicted cells where 1 means that this cell is a fork
        # (globally - considering cell transitions not depending on agent orientation)
        forks = np.zeros(self.max_prediction_depth, dtype=int)
        # Target
        target = np.zeros(self.max_prediction_depth, dtype=int)
        for index in range(self.max_prediction_depth):
            # Fill as 1 if transitions represent a fork cell
            cell = self.cells_sequence[handle][index]
            if cell in self.forks_coords:
                forks[index] = 1
            if cell == agent.target:
                target[index] = 1

        #  Speed/priority
        is_conflict = True if len(conflicting_agents) > 0 else False
        priority = assign_priority(self.env, agent, is_conflict)
        max_prio_encountered = 0
        if is_conflict:
            conflicting_agents_priorities = [assign_priority(self.env, agents[ca], True) for ca in conflicting_agents]
            max_prio_encountered = np.min(conflicting_agents_priorities)  # Max prio is the one with lowest value

        # Malfunctioning obs
        # Counting number of agents that are currently malfunctioning (globally) - experimental
        n_agents_malfunctioning = 0  # in TreeObs they store the length of the longest malfunction encountered
        for a in agents:
            if a.malfunction_data['malfunction'] != 0:
                n_agents_malfunctioning += 1  # Considering ALL agents

        # Agents status (agents ready to depart) - it tells the agent how many will appear - encountered? or globally?
        n_agents_ready_to_depart = 0
        for a in agents:
            if a.status in [RailAgentStatus.READY_TO_DEPART]:
                n_agents_ready_to_depart += 1  # Considering ALL agents

        #ret["forks"] = forks
        #ret["target"] = target

        #ret["priority"] = priority
        #ret["max_priority"] = max_prio_encountered
        #ret["n_malfunction"] = n_agents_malfunctioning
        #ret["ready_to_depart"] = n_agents_ready_to_depart

        # With this obs the agent actually decides only if it has to move or stop
        return ret


    def preprocess_satate(self, state, handle):

        #must return the dictionary with additional values

        ret = {}

        #print(handle)
        #print(state[handle]["occupancy_old"])

        ret["occupancy_old"] = state[handle]["occupancy_old"]
        ret["conflict"] = state[handle]["conflict"]
        ret["conflicting_agents"] = state[handle]["conflicting_agents"]

        # Single values
        conflict_all = np.zeros((len(self.env.agents), self.max_prediction_depth))
        conflict_without_dir_all = np.zeros((len(self.env.agents), self.max_prediction_depth))

        for j in range(0, len(self.env.agents)):
            conflict = np.zeros(self.max_prediction_depth)
            conflict_status_vector = np.zeros(self.max_prediction_depth)

            if j != handle:

                # indices of surrounding agent j for overlap with main agent handle
                indices = np.where(state[handle]["overlap"][j]==1)

                # if there is a conflict
                if len(indices[0]) != 0:

                    # find these points in the path of main agent
                    # and reshape to remove extra one dimension
                    path_coordinates = np.asarray([state[handle]["path"][handle][i] for i in indices])
                    path_coordinates = path_coordinates.reshape((path_coordinates.shape[1], path_coordinates.shape[2]))

                    # find only unique of them
                    unique_a, idx = np.unique(path_coordinates,axis=0, return_index=True)
                    unique_a = unique_a[np.argsort(idx)]

                    # Look them up in the trajectory of other agent
                    # This gives a vector with 1 dimension for each point
                    path_coordinates_for_conflict = np.asarray([[ 1 if i[0] == item[0] and i[1] == item[1] else 0
                                                                  for item in state[handle]["path"][j]] for i in unique_a])

                    # Summarize above vector
                    # This gives the span of the same path that was conflicting with main agent
                    # But on the trajector of other agents
                    #
                    # Use it to see the overlap
                    #
                    time_overlap_vector = np.sum(path_coordinates_for_conflict, axis=0)

                    # After this 2's indicate time conflict
                    conflict = np.sum(np.concatenate([np.expand_dims(time_overlap_vector, axis=0),
                                                      np.expand_dims(state[handle]["overlap"][j],axis=0)],axis=0),axis=0)

                    # find points where conflict is happening
                    conflicting_points = [ [int(y[0]), int(y[1])] if x > 1 else [0,0] for x, y in zip(conflict, state[handle]["path"][handle])]

                    # check these points for direction
                    # for both the agents
                    conflict_status_vector = np.asarray([ 0
                                                          if state[handle]["direction"][j].get(str(int(i[0]))+","+str(int(i[1])),0)
                                                             == state[handle]["direction"][handle].get(str(int(i[0]))+","+str(int(i[1])),0) else 1
                                                          for i in conflicting_points]
                                                        )

                conflict_without_dir_all[j] = conflict
                conflict_all[j] = conflict_status_vector

        path_conflict = np.sum(conflict_all, axis=0)

        ret["per_agent_occupancy_in_time"] = conflict_without_dir_all
        ret["per_agent_occupancy_with_dir"] = conflict_all
        ret["occupancy_new"] = [1 if item > 0 else 0 for item in path_conflict]

        return ret





    ############## Test battery #################

    def test_battery(self, observations):

        final = {}
        for a in range(0, len(self.env.agents)):
            # for every conflicting agent
            # run below tests
            final1 = {}
            for ca in observations[a]["conflicting_agents"]:
                final2 = {}

                #
                # Which Conflicting agent ?
                #
                final2["conflict_agent_id"] = ca
                #
                # is the conflict agent faster/slower?
                # find more info
                #
                final2["speed"] = self.speed(a, ca, observations)

                # is the current agent at the entry-border of the conflict zone (CZ)(per other conflict agent)?
                # if first bit of per_agent_occupancy_in_time for agent is greater than 0
                final2["A_entering_CZ"] = self.a_entering_cz(a, ca, observations)
                # is the conflict agent about to enter the conflict zone (one step before the conflict zone)?
                # if first bit of per_agent_occupancy_in_time for conflicting agent is greater than 0
                final2["B_entering_CZ"] = self.ca_entering_cz(a, ca, observations)


                #
                # is the current agent in the CZ (per other conflict agent)?
                #
                final2["A_already_in_CZ"] = self.check_mutual_conflict(a, ca, observations)
                #
                # is the conflict agent in the conflict zone already?
                # if first bit is greater than 0
                #
                final2["B_already_in_CZ"] = self.check_mutual_conflict(ca, a, observations)


                #
                # is the conflict agent moving in the opposite direction?
                # check with the direction now
                #
                final2["B_direction_in_CZ"] = self.ca_direction_in_cz(a, ca, observations)

                #
                # Which Agent ?
                #
                #final1["agent_id"] = a

                final1[ca] = final2

            final[a] = final1

        return final

    # Test 1 : Done
    def a_entering_cz(self, a, ca, observations):
        return 1 if observations[a]["per_agent_occupancy_in_time"][ca][0] > 0 else 0

    # Test 2 : Done
    def ca_entering_cz(self, a, ca, observations):
        return 1 if observations[ca]["per_agent_occupancy_in_time"][a][1] > 0 else 0

    # Test 3 : Done
    def ca_direction_in_cz(self, a, ca, observations):
        return "Same" if self._is_following(a, ca) else "Opposite"
        #return 1 if observations[ca]["per_agent_occupancy_with_dir"][a][0] > 0 else 0

    # Test 4
    def check_mutual_conflict(self, a, ca, observations):

        agents = self.env.agents
        second_layer = np.zeros(self.max_prediction_depth, dtype=int) # Same size as occupancy

        if agents[a].status is RailAgentStatus.ACTIVE and agents[ca].status is RailAgentStatus.ACTIVE:

            ts = [x for x, y in enumerate(self.cells_sequence[ca]) if y[0] == agents[a].position[0] and y[1] == agents[a].position[1]]

            # Set to 1 conflict span which was already entered by some agent - fill left side and right side of ts
            if len(ts) > 0:
                i = ts[0] # Since the previous returns a list of ts
                while 0 <= i < self.max_prediction_depth:
                    second_layer[i] = 1 if observations[ca]["occupancy_new"][i] > 0 else 0
                    i -= 1
                i = ts[0]
                while i < self.max_prediction_depth:
                    second_layer[i] = 1 if observations[ca]["occupancy_new"][i] > 0 else 0
                    i += 1

        return int(second_layer[0])
        # default return from conflict

    # Test 6
    def speed(self, a, ca, observations):
        agents = self.env.agents
        return "High" \
            if agents[a].speed_data["speed"] < agents[ca].speed_data["speed"] \
            else "Equal" if agents[a].speed_data["speed"] == agents[ca].speed_data["speed"] \
            else "Low"



    #############################################

    def absolute_dir_dict(self, state):
        #dir = np.zeros((state.shape[0], state.shape[1]))
        dir = {}
        #dir = np.array((state.shape[0], state.shape[1]))
        for index in range(0, state.shape[0]):
            section_start = False
            temp_dir = {}
            #global start
            start = 0
            for index1 in range(1, state.shape[1]):
                #print(state[index][index1][0])
                if np.isnan(state[index][index1]).any():
                    pass
                else:
                    cur_x = state[index][index1][0]
                    cur_y = state[index][index1][1]
                    prev_x = state[index][index1-1][0]
                    prev_y = state[index][index1-1][1]

                    # find chunk with repetition
                    if not section_start:
                        start = index1 - 1
                        section_start = True
                    if section_start and (cur_x != prev_x or cur_y != prev_y):
                        # calculate direction when it ends
                        dir_val = 0
                        end = index1
                        section_start = False
                        if cur_x < prev_x:
                            dir_val = 12
                        elif cur_x > prev_x:
                            dir_val = 6
                        elif cur_y < prev_y:
                            dir_val = 9
                        elif cur_y > prev_y:
                            dir_val = 3

                        temp_dir[str(int(prev_x))+","+str(int(prev_y))] = dir_val
            dir[index] = temp_dir
            #with np.printoptions(threshold=np.inf):
            #    print(state[index], dir[index])
        #for temp_index in range(0, state.shape[0]):
        #    dir[temp_index][state.shape[1]-1] = dir[temp_index][state.shape[1]-2]
        return dir


    def _fill_occupancy(self, handle):
        """
        Returns encoding of agent occupancy as an array where each element is
        0: no other agent in this cell at this ts (free cell)
        >= 1: counter (probably) other agents here at the same ts, so conflict, e.g. if 1 => one possible conflict, 2 => 2 possible conflicts, etc.
        :param handle: agent id
        :return: occupancy, conflicting_agents
        """
        occupancy = np.zeros(self.max_prediction_depth, dtype=int)
        conflicting_agents = set()
        overlapping_paths = self._compute_overlapping_paths(handle)

        # cells_sequence = self.cells_sequence[handle]
        # span_cells = []
        for ts in range(self.max_prediction_depth):
            if self.env.agents[handle].status in [RailAgentStatus.READY_TO_DEPART, RailAgentStatus.ACTIVE]:
                occupancy[ts], conflicting_agents_ts = self._possible_conflict(handle, ts)
                conflicting_agents.update(conflicting_agents_ts)

        # If a conflict is predicted, then it makes sense to populate occupancy with overlapping paths
        # But only with THAT agent
        # Because I could have overlapping paths but without conflict (TODO improve)
        if len(conflicting_agents) != 0: # If there was conflict
            for ca in conflicting_agents:
                for ts in range(self.max_prediction_depth):
                    occupancy[ts] = overlapping_paths[ca, ts] if occupancy[ts] == 0 else 1

        # Occupancy is 0 for agents that are done - they don't perform actions anymore

        # the calculated occupancy is for the agents that have conflict and hence conflict occupancy

        return occupancy, conflicting_agents


    def _possible_conflict(self, handle, ts):
        """
        Function that given agent (as handle) and time step, returns a counter that represents the sum of possible conflicts with
        other agents at that time step.
        Possible conflict is computed considering time step (current, pre and stop), direction, and possibility to enter that cell
        in opposite direction (w.r.t. to current agent).
        Precondition: 0 <= ts <= self.max_prediction_depth - 1.
        Exclude READY_TO_DEPART agents from this count, namely, check conflicts only with agents that are already active.

        :param handle: agent id
        :param ts: time step
        :return occupancy_counter, conflicting_agents
        """
        occupancy_counter = 0
        cell_pos = self.predicted_pos_coord[ts][handle]
        int_pos = self.predicted_pos[ts][handle]
        pre_ts = max(0, ts - 1)
        post_ts = min(self.max_prediction_depth - 1, ts + 1)
        int_direction = int(self.predicted_dir[ts][handle])
        cell_transitions = self.env.rail.get_transitions(int(cell_pos[0]), int(cell_pos[1]), int_direction)
        conflicting_agents_ts = set()

        # Careful, int_pos, predicted_pos are not (y, x) but are given as int
        if int_pos in np.delete(self.predicted_pos[ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        elif int_pos in np.delete(self.predicted_pos[pre_ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[pre_ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[pre_ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[pre_ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        elif int_pos in np.delete(self.predicted_pos[post_ts], handle, 0):
            conflicting_agents = np.where(self.predicted_pos[post_ts] == int_pos)
            for ca in conflicting_agents[0]:
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts][handle] != self.predicted_dir[post_ts][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[post_ts][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            occupancy_counter += 1
                            conflicting_agents_ts.add(ca)

        return occupancy_counter, conflicting_agents_ts


    @staticmethod
    def _reverse_dir(direction):
        """
        Invert direction (int) of one agent.
        :param direction:
        :return:
        """
        return int((direction + 2) % 4)


    # More than overlapping paths, this function computes cells in common in the predictions
    def _compute_overlapping_paths(self, handle):
        """
        Function that checks overlapping paths, where paths take into account shortest path prediction, so time/speed, 
        but not the fact that the agent is moving or not.
        :param handle: agent id
        :return: overlapping_paths is a np.array that computes path overlapping for pairs of agents, where 1 means overlapping.
        Each layer represents overlapping with one particular agent.
        """
        overlapping_paths = np.zeros((self.env.get_num_agents(), self.max_prediction_depth), dtype=int)
        cells_sequence = self.predicted_pos_list[handle]
        for a in range(len(self.env.agents)):
            if a != handle:
                i = 0
                other_agent_cells_sequence = self.predicted_pos_list[a]
                for pos in cells_sequence:
                    if pos in other_agent_cells_sequence:
                        overlapping_paths[a, i] = 1
                    i += 1
        return overlapping_paths


    def _find_forks(self):
        """
        A fork (in the map) is either a switch or a diamond crossing.
        :return: 
        """
        forks = set() # Set of nodes as tuples/coordinates
        # Identify cells hat are nodes (have switches)
        for i in range(self.env.height):
            for j in range(self.env.width):

                is_switch = False
                is_crossing = False

                # Check if diamond crossing
                transitions_bit = bin(self.env.rail.get_full_transitions(i, j))
                if int(transitions_bit, 2) == int('1000010000100001', 2):
                    is_crossing = True

                else:
                    # Check if switch
                    for direction in (0, 1, 2, 3):  # 0:N, 1:E, 2:S, 3:W
                        possible_transitions = self.env.rail.get_transitions(i, j, direction)
                        num_transitions = np.count_nonzero(possible_transitions)
                        if num_transitions > 1:
                            is_switch = True

                if is_switch or is_crossing:
                    forks.add((i, j))

        return forks


    def _is_following(self, handle1, handle2):
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
            return True
        else:
            return False


    ##################################################################



    def _bfs_graph(self, handle: int = 0) -> {}:
        """
        Build a graph (dict) of nodes, where nodes are identified by ids, graph is directed, depends on agent direction
        (that are tuples that represent the cell position, eg (11, 23))
        :param handle: agent id
        :return: 
        """
        obs_graph = defaultdict(list)  # Dict node (as pos) : adjacent nodes
        visited_nodes = set()  # set
        bfs_queue = []
        done = False  # True if agent has reached its target

        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
            done = True
        else:
            return None

        agent_current_direction = agent.direction

        # Push root node into the queue
        root_node_obs = GraphObsForRailEnv.Node(cell_position=agent_virtual_position,
                                                agent_direction=agent_current_direction,
                                                is_target=done)
        bfs_queue.append(root_node_obs)

        # Perform BFS of depth = bfs_depth
        for i in range(1, self.bfs_depth + 1):
            # Temporary queue to store nodes that must be appended at the next pass
            tmp_queue = []
            while not len(bfs_queue) == 0:
                current_node = bfs_queue.pop(0)
                agent_position = current_node[0]

                # Init node in the obs_graph (if first time)
                if not agent_position in obs_graph.keys():
                    obs_graph[agent_position] = []

                agent_current_direction = current_node[1]
                # Get cell transitions given agent direction
                possible_transitions = self.env.rail.get_transitions(*agent_position, agent_current_direction)

                orientation = agent_current_direction
                possible_branch_directions = []
                # Build list of possible branching directions from cell
                for j, branch_direction in enumerate([(orientation + j) % 4 for j in range(-1, 3)]):
                    if possible_transitions[branch_direction]:
                        possible_branch_directions.append(branch_direction)
                for branch_direction in possible_branch_directions:
                    # Gets adjacent cell and start exploring from that for possible fork points
                    neighbour_cell = get_new_position(agent_position, branch_direction)
                    adj_node = self._explore_path(handle, neighbour_cell, branch_direction)
                    if not (*adj_node[0], adj_node[1]) in visited_nodes:
                        # For now I'm using as key the agent_position tuple
                        obs_graph[agent_position].append(adj_node)
                        visited_nodes.add((*adj_node[0], adj_node[1]))
                        tmp_queue.append(adj_node)
            # Add all the nodes of the next level to the BFS queue
            for el in tmp_queue:
                bfs_queue.append(el)

        # After the last pass add adj nodes to the obs graph wih empty lists
        for el in bfs_queue:
            if not el[0] in obs_graph.keys():
                obs_graph[el[0]] = []
                # visited_nodes.add((*el[0], el[1]))
        # For obs rendering
        # self.env.dev_obs_dict[handle] = [(node[0], node[1]) for node in visited_nodes]

        # Build graph with graph-tool library for visualization
        # g = build_graph(obs_graph, handle)

        return obs_graph

    def _explore_path(self, handle, position, direction):
        """
        Given agent handle, current position, and direction, explore that path until a new branching point is found.
        :param handle: agent id
        :param position: agent position as cell 
        :param direction: agent direction
        :return: a tuple Node with its features
        """

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        # 4 different cases to have a branching point:
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell or cycle
        last_is_target = False  # target was reached
        agent = self.env.agents[handle]
        visited = OrderedSet()

        while True:

            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            cell_transitions = self.env.rail.get_transitions(*position, direction)
            num_transitions = np.count_nonzero(cell_transitions)
            cell_transitions_bitmap = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = cell_transitions_bitmap.count("1")

            if num_transitions == 1:
                # Check if dead-end (1111111111111111), or if we can go forward along direction
                if total_transitions == 1:
                    last_is_dead_end = True
                    break

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)

            elif num_transitions > 1:
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break
        # Out of while loop - a branching point was found

        # TODO Here to save more features in a node
        node = GraphObsForRailEnv.Node(cell_position=position,
                                       agent_direction=direction,
                                       is_target=last_is_target)

        return node


    # TODO Stop when shortest_path() says that rail is disrupted
    def _get_shortest_path_action(self, handle):
        """
        Takes an agent handle and returns next action for that agent following shortest path:
        - if agent status == READY_TO_DEPART => agent moves forward;
        - if agent status == ACTIVE => pick action using shortest_path.py() fun available in prediction utils;
        - if agent status == DONE => agent does nothing.
        :param handle:
        :return:
        """

        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:

            if self.num_active_agents < 10:  # TODO
                # This could be reasonable since agents never start on switches - I guess
                action = RailEnvActions.MOVE_FORWARD
            else:
                action = RailEnvActions.DO_NOTHING


        elif agent.status == RailAgentStatus.ACTIVE:
            # This can return None when rails are disconnected or there was an error in the DistanceMap
            shortest_paths = self.predictor.get_shortest_paths()

            if shortest_paths[handle] is None:  # Railway disrupted
                action = RailEnvActions.STOP_MOVING
            else:
                step = shortest_paths[handle][0]
                next_action_element = step[2][0]  # Get next_action_element

                # Just to use the correct form/name
                if next_action_element == 1:
                    action = RailEnvActions.MOVE_LEFT
                elif next_action_element == 2:
                    action = RailEnvActions.MOVE_FORWARD
                elif next_action_element == 3:
                    action = RailEnvActions.MOVE_RIGHT

        else:  # If status == DONE or DONE_REMOVED
            action = RailEnvActions.DO_NOTHING

        return action


    def choose_railenv_action(self, handle, network_action):
        """
        Choose action to perform from RailEnvActions, namely follow shortest path or stop if DQN network said so.

        :param handle:
        :param network_action:
        :return:
        """

        if network_action == 1:
            return RailEnvActions.STOP_MOVING
        else:
            return self._get_shortest_path_action(handle)

