"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset (i.e. at the beginning of a new episode), to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""
import random

import collections
from typing import Optional, List, Dict, Tuple
import numpy as np
from collections import defaultdict

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position, distance_on_rail, position_to_coordinate
from flatland.utils.ordered_set import OrderedSet
from typing import Dict, List, Optional, NamedTuple, Tuple, Set

from src.priority import assign_priority

WalkingElement = \
    NamedTuple('WalkingElement',
               [('position', Tuple[int, int]), ('direction', int), ('next_action_element', RailEnvNextAction)])


class GraphObsForRailEnv(ObservationBuilder):
    """
    Build graph observations.
    """

    Node = collections.namedtuple('Node',
                                  'cell_position '  # Cell position (x, y)
                                  'agent_direction '  # Direction with which the agent arrived in this node
                                  'is_target '       # Whether agent's target is in this cell
                                  'depth ')          # Depth of this node from start


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

        # cell sequence changes hence calculation of direction should change
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



        path = self.path()
        direction = self.absolute_dir_dict(path)

        observations = {}
        for a in handles:
            observations[a] = self.get(path, direction, a)

        preprocessed_observation = {}
        for a in handles:
            preprocessed_observation[a] = self.preprocess_state(observations, a)

        dict_temp = {}
        dict_temp["preprocessed_observation"] = preprocessed_observation
        dict_temp["cells_sequence"] = self.cells_sequence


        #self.find_alternate(dict_temp)


        return dict_temp


    # TODO Optimize considering that I don't need obs for those agents who don't have to pick actions
    def get(self, path, direction, handle: int = 0) -> {}:
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
        occupancy, conflicting_agents, overlapping_paths = self._fill_occupancy(handle, path)

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
        """

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

        """

        ret = {}
        ret["path"] = path
        ret["overlap_new"] = self._compute_overlapping_paths1(handle, path)
        ret["overlap_old"] = overlapping_paths
        ret["direction"] = direction
        ret["occupancy_old"] = occupancy
        #ret["bypass"] = self._bypass_dict(direction, path[handle], handle)
        ret["conflicting_agents"] = conflicting_agents
        ret["conflict"] = second_layer

        # With this obs the agent actually decides only if it has to move or stop
        return ret


    def preprocess_state(self, state, handle):

        ret = state[handle]
        conflict_without_dir_all, path_conflict = self.preprocess_state_part(state, handle)

        #ret = {}
        #ret["conflicting_agents"] = state[handle]["conflicting_agents"]
        #ret["path"] = state[handle]["path"]
        #ret["overlap_old"] = state[handle]["overlap_old"]
        #ret["overlap_new"] = state[handle]["overlap_new"]
        #ret["direction"] = state[handle]["direction"]
        #ret["bypass"] = state[handle]["bypass"]
        #ret["occupancy_old"] = state[handle]["occupancy_old"]

        ret["per_agent_occupancy_in_time"] = conflict_without_dir_all
        ret["occupancy_new"] = [1 if item > 0 else 0 for item in path_conflict]

        return ret


    def preprocess_state_part(self, state, handle):

        # Single values
        conflict_all = np.zeros((len(self.env.agents), self.max_prediction_depth))
        conflict_without_dir_all = np.zeros((len(self.env.agents), self.max_prediction_depth))

        for j in range(0, len(self.env.agents)):
            conflict = np.zeros(self.max_prediction_depth)
            conflict_status_vector = np.zeros(self.max_prediction_depth)

            if j != handle:

                # indices of surrounding agent j for overlap with main agent handle
                indices = np.where(state[handle]["overlap_new"][j]==1)

                # if there is a conflict
                if len(indices[0]) != 0:

                    # find these points in the path of main agent
                    # and reshape to remove extra one dimension
                    path_coordinates = np.asarray([state[handle]["path"][handle][i] for i in indices])
                    path_coordinates = path_coordinates.reshape((path_coordinates.shape[1], path_coordinates.shape[2]))

                    # find only unique of them
                    unique_a, idx = np.unique(path_coordinates,axis=0, return_index=True)
                    unique_a = unique_a[np.argsort(idx)]

                    #
                    # Look them up in the trajectory of other agent
                    # This gives a vector with 1 dimension for each point
                    path_coordinates_for_conflict = np.asarray([[ 1 if i[0] == item[0] and i[1] == item[1] else 0
                                                                  for item in state[handle]["path"][j]] for i in unique_a])

                    #
                    # modify path and trim last values to zeros
                    # Intended to resolve the end point waiting problem
                    #
                    repetition = np.count_nonzero(path_coordinates_for_conflict[0])
                    index = np.argmax(path_coordinates_for_conflict[-1]>0)
                    path_coordinates_for_conflict[-1][:] = 0
                    path_coordinates_for_conflict[-1][index:index+repetition] = 1

                    # Summarize above vector
                    # This gives the span of the same path that was conflicting with main agent
                    # But on the trajector of other agents
                    #
                    # Use it to see the overlap
                    #
                    time_overlap_vector = np.sum(path_coordinates_for_conflict, axis=0)

                    # After this 2's indicate time conflict
                    conflict = np.sum(np.concatenate([np.expand_dims(time_overlap_vector, axis=0),
                                                      np.expand_dims(state[handle]["overlap_new"][j],axis=0)],axis=0),axis=0)

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

        return conflict_without_dir_all, path_conflict



    ############################################# BYPASS EVALUATION ####################################################

    def find_alternate(self, dict_temp):
        # ############### Evaluate alternative paths #####################
        # get per agent cost
        cost_mat = self.evaluate_cost(dict_temp)
        best_cost = np.sum(cost_mat)
        print("Original Overall conflict cost is ", best_cost)

        for i in range(0,20):
            alt_dict = self.alternate_path(cost_mat, dict_temp)
            alt_cost = self.evaluate_cost(alt_dict)
            if best_cost > np.sum(alt_cost):
                print("found better alternative")
                break
            else:
                print("next check", cost_mat, alt_cost)


    def alternate_path(self, cost_mat, dict_temp_temp):

        dict_temp = dict_temp_temp
        most_costly = random.randint(0,3)
        agent = self.env.agents[most_costly]
        speed = agent.speed_data["speed"]

        repeat = 1
        if float("{0:.2f}".format(speed)) == 1.0:
            repeat = 1
        if float("{0:.2f}".format(speed)) == 0.50:
            repeat = 2
        if float("{0:.2f}".format(speed)) == 0.33:
            repeat = 3
        if float("{0:.2f}".format(speed)) == 0.25:
            repeat = 4

        selected_bypass = random.randint(0,len(dict_temp["preprocessed_observation"][most_costly]["bypass"])-1)

        retry_counter = 0
        while True:
            # check if the bypass is the same old path
            retry_counter += 1

            unique_a, idx = np.unique(dict_temp_temp["preprocessed_observation"][most_costly]["path"][most_costly], axis=0, return_index=True)
            unique_a = unique_a[np.argsort(idx)]
            where = np.where((unique_a == (0.0, 0.0)).all(axis=1))
            unique_a = np.delete(unique_a, where, axis=0)

            unique_b, idy = np.unique(dict_temp_temp["preprocessed_observation"][most_costly]["bypass"][selected_bypass], axis=0, return_index=True)
            unique_b = unique_b[np.argsort(idy)]
            where = np.where((unique_b == (0.0, 0.0)).all(axis=1))
            unique_b = np.delete(unique_b, where, axis=0)

            # find intersection
            a = set((tuple(i) for i in unique_a))
            b = set((tuple(i) for i in unique_b))
            i_section = a.intersection(b)


            if len(i_section) > 0 or retry_counter > 5:
                break
            else:
                selected_bypass = random.randint(0,
                                                 len(dict_temp["preprocessed_observation"][most_costly]["bypass"]) - 1)


            #unique_a = np.unique(dict_temp_temp["preprocessed_observation"][most_costly]["path"][most_costly], axis=0)
            #where = np.where((unique_a == (0.0, 0.0)).all(axis=1))
            #unique_a = np.delete(unique_a, where, axis=0)
            #unique_a = [str(int(item[0])) + "," + str(int(item[1])) for item in unique_a]

            #unique_b = np.unique(dict_temp_temp["preprocessed_observation"][most_costly]["bypass"][selected_bypass], axis=0)

            #if len(np.setdiff1d(unique_a, unique_b)) > 0 or retry_counter > 5:
            #    break
            #else:
            #    selected_bypass = random.randint(0,
            #                                     len(dict_temp["preprocessed_observation"][most_costly]["bypass"]) - 1)

        local_path_bunch = dict_temp["preprocessed_observation"][most_costly]["path"]
        #print(local_path_bunch[most_costly])
        for i in range(0, len(dict_temp["preprocessed_observation"][most_costly]["bypass"][selected_bypass])-1):
            temp_val = dict_temp["preprocessed_observation"][most_costly]["bypass"][selected_bypass][i].split(",")
            local_path_bunch[most_costly][i*repeat:(i+1)*repeat] = [int(temp_val[0]), int(temp_val[1])]

            if i == len(dict_temp["preprocessed_observation"][most_costly]["bypass"][selected_bypass])-1 and\
                    i < 200:
                local_path_bunch[most_costly][(i + 1) * repeat:200] = [0,0]
        #print(local_path_bunch[most_costly])


        for i in range(0, len(dict_temp["preprocessed_observation"])):
            # change path
            dict_temp["preprocessed_observation"][i]["path"] = local_path_bunch
            # change overlap
            dict_temp["preprocessed_observation"][i]["overlap"] = self._compute_overlapping_paths1(i, dict_temp["preprocessed_observation"][i]["path"])
            # change direction : not needed for cost

        # now compute occupancy
        for i in range(0, len(dict_temp["preprocessed_observation"])):
            a, b = self.preprocess_state_part(dict_temp["preprocessed_observation"], i)
            dict_temp["preprocessed_observation"][i]["overlap"] = a
            dict_temp["preprocessed_observation"][i]["occupancy_new"] = b

        return dict_temp


    def evaluate_cost(self, dict_temp):
        temp = dict_temp["preprocessed_observation"]
        cost_mat = np.zeros(len(dict_temp["preprocessed_observation"]))
        for item in temp:
            cost = 0
            for item1 in temp[item]["per_agent_occupancy_in_time"]:
                cost += np.sum(item1, axis=0)
            cost_mat[item] = cost
        return cost_mat

    ####################################################################################################################
    ############################################## BYPASSES CALCULATION ################################################
    ####################################################################################################################

    def _bypass_dict(self, direction, path, handle):

        obs_graph = defaultdict(list)  # Dict node (as pos) : adjacent nodes
        if np.isnan(path[0][0]):
            return obs_graph

        # find only unique of them
        unique_a, idx = np.unique(path,axis=0, return_index=True)
        unique_a = unique_a[np.argsort(idx)][::-1]
        where = np.where((unique_a == (0.0, 0.0)).all(axis=1))
        unique_a = np.delete(unique_a, where, axis=0)

        #unique_a = unique_a[np.argsort(idx)]

        if len(unique_a) <= 2:
            return obs_graph

        target = str(int(unique_a[-1][0]))+","+str(int(unique_a[-1][1]))

        visited_nodes = set()  # set
        bfs_queue = []

        initial = str(int(unique_a[0][0]))+","+str(int(unique_a[0][1]))
        adjacent = str(int(unique_a[1][0]))+","+str(int(unique_a[1][1]))

        obs_graph[initial].append(adjacent)
        visited_nodes.add((initial))

        agent_virtual_position = adjacent
        bfs_queue.append(agent_virtual_position)

        # Perform BFS of depth = bfs_depth
        while True:
            # Temporary queue to store nodes that must be appended at the next pass
            tmp_queue = []
            while not len(bfs_queue) == 0:
                current_node = bfs_queue.pop(0)
                agent_position = current_node
                # Init node in the obs_graph (if first time)
                if not agent_position in obs_graph.keys():
                    obs_graph[agent_position] = []

                node = current_node.split(",")
                a1 = self.env.distance_map.distance_map[handle][int(node[0])][int(node[1])]
                a2 = np.min(a1)
                a3 = np.asarray([1 if item == a2 else 0 for item in a1])
                a5 = np.where(a3==1)

                for i in a5:
                    for j in i:
                        found = False
                        if j == 0:
                            a22 = np.min(self.env.distance_map.distance_map[handle][int(node[0])+1][int(node[1])])
                            if a22 -1 == a2:
                                adjacent = str(int(node[0])+1)+","+str(int(node[1]))
                                found = True
                        if j == 1:
                            a22 = np.min(self.env.distance_map.distance_map[handle][int(node[0])][int(node[1])-1])
                            if a22 -1 == a2:
                                adjacent = str(int(node[0]))+","+str(int(node[1])-1)
                                found = True
                        if j == 2:
                            a22 = np.min(self.env.distance_map.distance_map[handle][int(node[0])-1][int(node[1])])
                            if a22 -1 == a2:
                                adjacent = str(int(node[0])-1)+","+str(int(node[1]))
                                found = True
                        if j == 3:
                            a22 = np.min(self.env.distance_map.distance_map[handle][int(node[0])][int(node[1])+1])
                            if a22 -1 == a2:
                                adjacent = str(int(node[0]))+","+str(int(node[1])+1)
                                found = True

                        if not adjacent in visited_nodes:
                            # For now I'm using as key the agent_position tuple
                            obs_graph[agent_position].append(adjacent)
                            visited_nodes.add((adjacent))
                            tmp_queue.append(adjacent)
                        elif adjacent in visited_nodes and len(obs_graph[agent_position]) == 0 and found:
                            obs_graph[agent_position].append(adjacent)
                        #else:
                        #    for item in obs_graph[agent_position]:
                        #        if item != adjacent:
                        #            obs_graph[agent_position].append(adjacent)
            # Add all the nodes of the next level to the BFS queue
            for el in tmp_queue:
                bfs_queue.append(el)

            if len(tmp_queue) == 0 and len(bfs_queue) == 0:
                break

        dict = defaultdict(list)  # Dict node (as pos) : adjacent nodes

        for node in obs_graph:
            for item in obs_graph[node]:
                dict[item].append(node)

        # initial is target and target is initial now.
        # Inversion as the map starts from goal to start and traversal from goal to start
        #stack = self.find_all_paths(obs_graph, target, initial)
        stack = self.find_all_paths(dict, target, initial)

        return stack

    def find_all_paths(self, graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not start in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self.find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    ####################################################################################################################

    # paths with zeros substituted at the end
    def path(self):
        path = np.zeros((len(self.predicted_pos_coord[0])
                         ,len(self.predicted_pos_coord)
                         ,self.predicted_pos_coord[0][0].shape[0]))

        for index in range(0,len(self.predicted_pos_coord[0])):

            repetition = 0
            first = self.predicted_pos_coord[0][index]
            run_index = 0

            while True:
                #print(index, run_index)
                if first[0] == self.predicted_pos_coord[run_index][index][0] and first[1] == self.predicted_pos_coord[run_index][index][1]:
                    run_index += 1
                    if run_index == 4:
                        repetition = run_index
                        break
                else:
                    repetition = run_index
                    break

            temp = np.zeros((len(self.predicted_pos_coord),2))

            for item in self.predicted_pos_coord.keys():
                #print(item)
                temp[int(item)] = self.predicted_pos_coord[item][index]

            unique_a = np.unique(temp,axis=0)
            unique_count = len(unique_a)

            for index1 in range(0,len(self.predicted_pos_coord)):
                if index1 < unique_count*repetition:
                    path[index][index1][0] = self.predicted_pos_coord[index1][index][0]
                    path[index][index1][1] = self.predicted_pos_coord[index1][index][1]
                #else:
                #    path[index][index1][0] = np.nan
                #    path[index][index1][1] = np.nan

        return path


    def _compute_overlapping_paths1(self,handle, state):

        occ = np.zeros( (len(state), len(state[0])) , dtype=np.int8)


        for j in range(0, len(state)):
            if j != handle:
                # you have two arrays
                # find unique and sort them
                unique_a, idx = np.unique(state[handle],axis=0, return_index=True)
                unique_a = unique_a[np.argsort(idx)]
                where = np.where((unique_a==(0.0, 0.0)).all(axis=1))
                unique_a = np.delete(unique_a, where, axis=0)

                unique_b, idy = np.unique(state[j],axis=0, return_index=True)
                unique_b = unique_b[np.argsort(idy)]
                where = np.where((unique_b==(0.0, 0.0)).all(axis=1))
                unique_b = np.delete(unique_b, where, axis=0)

                # find intersection
                a = set((tuple(i) for i in unique_a))
                b = set((tuple(i) for i in unique_b))
                i_section = a.intersection(b)

                masked_b = np.sum([[1 if x[0] == item[0] and x[1] == item[1] else 0 for x in state[handle]] for item in i_section ], axis=0)

                occ[j] = masked_b
        return occ


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


    def absolute_dir_dict(self, state):
        direction = {}
        for item in self.env.dev_pred_dict:
            dir_local = {}
            for item1 in self.env.dev_pred_dict[item]:
                #dir_local[]
                #print(item1[0], item1[1], item1[2])
                dir_local[str(item1[0])+","+str(item1[1])] = item1[2]
            direction[item] = dir_local
        return direction

    #################################################################################################

    def _fill_occupancy(self, handle, path):
        """
        Returns encoding of agent occupancy as an array where each element is
        0: no other agent in this cell at this ts (free cell)
        >= 1: counter (probably) other agents here at the same ts, so conflict, e.g. if 1 => one possible conflict, 2 => 2 possible conflicts, etc.
        :param handle: agent id
        :return: occupancy, conflicting_agents
        """
        occupancy = np.zeros(self.max_prediction_depth, dtype=int)
        conflicting_agents = set()
        #overlapping_paths = self._compute_overlapping_paths(handle)
        overlapping_paths = self._compute_overlapping_paths1(handle, path)

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

        return occupancy, conflicting_agents, overlapping_paths


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

    """
    # More than overlapping paths, this function computes cells in common in the predictions
    def _compute_overlapping_paths(self, handle):
        #Function that checks overlapping paths, where paths take into account shortest path prediction, so time/speed, 
        #but not the fact that the agent is moving or not.
        #:param handle: agent id
        #:return: overlapping_paths is a np.array that computes path overlapping for pairs of agents, where 1 means overlapping.
        #Each layer represents overlapping with one particular agent.
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
    """

    def _find_forks(self):
        """
        A fork (in the map) is either a switch or a diamond crossing.
        :return: 
        """
        forks = set() # Set of nodes as tuples/coordinates
        # Identify cells that are nodes (have switches)
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


    # TODO Stop when shortest_path() says that rail is disrupted
    def _get_shortest_path_action(self, handle):
        #Takes an agent handle and returns next action for that agent following shortest path:
        #- if agent status == READY_TO_DEPART => agent moves forward;
        #- if agent status == ACTIVE => pick action using shortest_path.py() fun available in prediction utils;
        #- if agent status == DONE => agent does nothing.
        #:param handle:
        #:return:

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

    #################################################################################################
