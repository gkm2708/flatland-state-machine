"""
ObservationBuilder objects are objects that can be passed to environments designed for customizability.
The ObservationBuilder-derived custom classes implement 2 functions, reset() and get() or get(handle).

+ `reset()` is called after each environment reset (i.e. at the beginning of a new episode), to allow for pre-computing relevant data.

+ `get()` is called whenever an observation has to be computed, potentially for each agent independently in case of \
multi-agent environments.

"""
import random

import collections
import itertools
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


class GraphObsForRailEnv(ObservationBuilder):
    """
    Build graph observations.
    """
    ####################################################################################################################
    ################################################## Initializers ####################################################
    ####################################################################################################################


    Node = collections.namedtuple('Node',
                                  'cell_position '  # Cell position (x, y)
                                  'agent_direction '  # Direction with which the agent arrived in this node
                                  'is_target '       # Whether agent's target is in this cell
                                  'depth ')          # Depth of this node from start


    def __init__(self, predictor, bfs_depth):
        super(GraphObsForRailEnv, self).__init__()
        self.predictor = predictor
        self.bfs_depth = bfs_depth
        #self.max_prediction_depth = 0
        self.prediction_dict = {}  # Dict handle : list of tuples representing prediction steps
        self.predicted_pos = {}  # Dict ts : int_pos_list
        self.predicted_pos_list = {} # Dict handle : int_pos_list
        self.predicted_pos_coord = {}  # Dict ts : coord_pos_list
        self.predicted_dir = {}  # Dict ts : dir (float)
        self.num_active_agents = 0
        self.cells_sequence = None
        self.forks_coords = None
        self.observations = {}
        self.preprocessed_observation = {}
        self.first_step = True
        self.max_prediction_depth = self.predictor.max_depth

        self.conflicting_agents = {} # Dict handle : set of predicted conflicting agents


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
        self.first_step = True
        self.forks_coords = self._find_forks()


        self.prediction_dict = {}  # Dict handle : list of tuples representing prediction steps
        self.predicted_pos = {}  # Dict ts : int_pos_list
        self.predicted_pos_list = {} # Dict handle : int_pos_list
        self.predicted_pos_coord = {}  # Dict ts : coord_pos_list
        self.predicted_dir = {}  # Dict ts : dir (float)
        self.num_active_agents = 0
        self.cells_sequence = None
        self.max_prediction_depth = self.predictor.max_depth

        self.conflicting_agents = {} # Dict handle : set of predicted conflicting agents

        self.observations = {}
        self.preprocessed_observation = {}
        self.direction = {}
        for a in self.env.agents:
           self.prediction_dict[a.handle] = {}

        self.path_to_dest = np.zeros((len(self.env.agents), self.env.height, self.env.width), dtype=np.int16)
        self.conflict_in_time = np.zeros((len(self.env.agents), self.max_prediction_depth), dtype=np.int8)
        self.conflict_in_cells = np.zeros((len(self.env.agents), self.env.height, self.env.width), dtype=np.int8)
        self.conflict_bitmap = np.zeros((len(self.env.agents), len(self.env.agents)), dtype=np.int8)
        self.update_mask = np.ones((len(self.env.agents), self.env.height, self.env.width), dtype=np.int8)
        self.path_all = np.zeros((self.env.number_of_agents,self.max_prediction_depth,2), dtype=np.int8)


    ####################################################################################################################
    ##################################################### Cockpit ######################################################
    ####################################################################################################################

    # only return memory map every time step
    # also populate memory map by calling get_many_parts at step 0
    def get_many(self, handles: Optional[List[int]] = None) -> {}:
        """
        Compute observations for all agents in the env.
        :param handles: 
        :return: 
        """

        if self.first_step:
            self.get_many_parts(handles)
            self.first_step = False
        #else:
        #    self.update_memory_every_step()

        return (self.path_to_dest, self.conflict_bitmap)






    def get_many_parts(self, handles):

        self.num_active_agents = 0

        for a in handles:
            agent = self.env.agents[a]

            if agent.status == RailAgentStatus.ACTIVE:
                self.num_active_agents += 1

            self.prediction_dict[a] = self.predictor.get(a)

            # cell sequence changes hence calculation of direction should change
            if self.prediction_dict and len(self.prediction_dict) == self.env.number_of_agents:

                self.cells_sequence = self.predictor.compute_cells_sequence(self.prediction_dict)
                # Useful to check if occupancy is correctly computed

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

                for b in range(len(self.env.agents)):
                    pos_list = []
                    for ts in range(self.max_prediction_depth):
                        pos_list.append(self.predicted_pos[ts][b])  # Use int positions
                    self.predicted_pos_list.update({b: pos_list})

                for b in handles:
                    self.conflicting_agents[b] = set()

                self.conflicting_agents = self._predicted_conflicts()

                if self.first_step:
                    for b in handles:
                        # change this also to work on a handle only
                        self.path_all[b] = self.path(b)
                        # change this also to work on a handle only
                        self.direction[b] = self.absolute_dir_dict(b)
                        # maintain historical observations
                        self.observations[b] = self.get(self.path_all, self.direction, b)
                        self.preprocessed_observation[b] = self.preprocess_state(self.observations, b)
                        self.update_path_memory_at_recalculate(b)
                else:
                    self.path_all[a] = self.path(a)
                    # change this also to work on a handle only
                    self.direction[a] = self.absolute_dir_dict(a)
                    # maintain historical observations
                    self.observations[a] = self.get(self.path_all, self.direction, a)
                    self.preprocessed_observation[a] = self.preprocess_state(self.observations, a)
                    self.update_path_memory_at_recalculate(a)


    def update_path_memory_at_recalculate(self, handle):
        speed = self.env.agents[handle].speed_data["speed"]
        repeat = 1
        if float("{0:.2f}".format(speed)) == 1.0:
            repeat = 1
        if float("{0:.2f}".format(speed)) == 0.50:
            repeat = 2
        if float("{0:.2f}".format(speed)) == 0.33:
            repeat = 3
        if float("{0:.2f}".format(speed)) == 0.25:
            repeat = 4

        # populate path memory for an agent
        mul = 1
        for item in self.preprocessed_observation[handle]["path"][handle]:
            if item[0] != 0 and item[1] != 0:
                self.path_to_dest[handle][item[0]][item[1]] = repeat*mul
                mul += 1

        # populate time overlap memory for an agent
        #print(np.sum(self.preprocessed_observation[handle]["per_agent_occupancy_in_time"], axis=0))
        #self.conflict_in_time[handle] = np.sum(self.preprocessed_observation[handle]["per_agent_occupancy_in_time"], axis=0)

        # populate cell overlap memory for an agent

        # populate conflicting agents bitmap
        #for a in self.conflicting_agents:
        #    for item in self.conflicting_agents[a]:
        #        self.conflict_bitmap[item[0]][item[0]] = 1


    def update_memory_every_step(self):
        #print("updating")
        for a in self.env.agents:
            for i in range(0, self.max_prediction_depth - 1):
                self.conflict_in_time[a.handle][i] = self.conflict_in_time[a.handle][i+1]

        self.path_to_dest = self.path_to_dest - self.update_mask


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

        ret = {}
        # Just Copy
        ret["path"] = path
        # this will happen for an agent in comparision of all teh agents
        ret["overlap_new"] = self._compute_overlapping_paths(handle, path)
        # Just Copy
        ret["direction"] = direction
        #ret["bypass"] = self._bypass_dict(path[handle], handle)
        # Done for a said agent
        ret["conflicting_agents"] = self.conflicting_agents[handle]

        # With this obs the agent actually decides only if it has to move or stop
        return ret


    def preprocess_state(self, state, handle):

        ret = state[handle]
        ret["per_agent_occupancy_in_time"] = self.preprocess_state_part(state, handle)

        return ret


    def preprocess_state_part(self, state, handle):

        # Single values
        #conflict_all = np.zeros((len(self.env.agents), self.max_prediction_depth))
        conflict_without_dir_all = np.zeros((len(self.env.agents), self.max_prediction_depth))

        for j in range(0, len(self.env.agents)):
            conflict = np.zeros(self.max_prediction_depth)
            #conflict_status_vector = np.zeros(self.max_prediction_depth)

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
                    #conflicting_points = [ [int(y[0]), int(y[1])] if x > 1 else [0,0] for x, y in zip(conflict, state[handle]["path"][handle])]

                    # check these points for direction
                    # for both the agents
                    #conflict_status_vector = np.asarray([ 0
                    #                                      if state[handle]["direction"][j].get(str(int(i[0]))+","+str(int(i[1])),0)
                    #                                         == state[handle]["direction"][handle].get(str(int(i[0]))+","+str(int(i[1])),0) else 1
                    #                                      for i in conflicting_points]
                    #                                    )

                conflict_without_dir_all[j] = conflict
                #conflict_all[j] = conflict_status_vector

        #path_conflict = np.sum(conflict_all, axis=0)

        #return conflict_without_dir_all, path_conflict
        return conflict_without_dir_all


    def read_memory(self):
        return (self.path_to_dest, self.conflict_bitmap)

    ####################################################################################################################
    ############################################# BYPASS EVALUATION    #################################################
    ############################################# BYPASSES CALCULATION #################################################
    ####################################################################################################################

    def use_bypass(self, dict_temp):

        """
                        # Replan
                    if use_bypass:
                        best_cost = np.sum(self.evaluate_cost(dict_temp))
                        self.alternate_observation_dict = self.find_alternate(dict_temp)
                        current_cost = np.sum(self.evaluate_cost(self.alternate_observation_dict))
                        # if better path
                        # replace
                        if current_cost < best_cost:
                            print("new path", current_cost, best_cost)
                            dict_temp["preprocessed_observation"] = self.alternate_observation_dict[
                                "preprocessed_observation"].copy()
                            # also change cell sequence
                            cell_sequence = self.cells_sequence.copy()
                            for a in range(len(self.env.agents)):
                                for i in range(0, len(cell_sequence[a]) - 1):
                                    cell_sequence[a][i] = (dict_temp["preprocessed_observation"][a]["path"][a][i][0],
                                                           dict_temp["preprocessed_observation"][a]["path"][a][i][1])
                            cell_sequence[a][i + 1] = (dict_temp["preprocessed_observation"][a]["path"][a][-1][0],
                                                       dict_temp["preprocessed_observation"][a]["path"][a][-1][1])
                            dict_temp["cells_sequence"] = cell_sequence
                        else:
                            dict_temp["cells_sequence"] = self.cells_sequence
                    else:
                        dict_temp["cells_sequence"] = self.cells_sequence
                        self.alternate_observation_dict = dict_temp



        """

        return dict_temp


    def find_alternate(self, dict_temp):
        # Evaluate current path costs
        # get per agent cost (overall conflict for every agent)
        cost_mat = self.evaluate_cost(dict_temp)
        # sum to make a cost (overall conflict for all the agents)
        best_cost = np.sum(cost_mat)
        best_cost_perm = best_cost
        #print("Original Overall conflict cost is ", best_cost, cost_mat)

        # ############### Evaluate alternative paths #####################

        for a in dict_temp["preprocessed_observation"]:
            #print("loop1",self.evaluate_cost(dict_temp))
            bypass_len = len(dict_temp["preprocessed_observation"][a]["bypass"])
            if bypass_len > 1:
                alt_dict, alt_cost = self.alternate_path(best_cost, dict_temp, a, bypass_len)
                if best_cost > alt_cost:
                    #print("found better alternative", best_cost, alt_cost)
                    #with np.printoptions(threshold=np.inf):
                    #    print(dict_temp["preprocessed_observation"][0]["path"], alt_dict["preprocessed_observation"][0]["path"])
                    dict_temp = alt_dict
                    best_cost = alt_cost
            #print("loop1-",self.evaluate_cost(dict_temp))

        if best_cost_perm > best_cost:
            # here recalculate per_agent_occupancy_in_time
            for a in dict_temp["preprocessed_observation"]:
                x = self.preprocess_state_part(dict_temp["preprocessed_observation"], a)
                dict_temp["preprocessed_observation"][a]["per_agent_occupancy_in_time"] = x
                # heer change cell sequence

        return dict_temp


    def alternate_path(self, best_cost, dict_temp_temp, handle, option_size):

        # make a copy of dictionary
        dict_temp = dict_temp_temp.copy()

        agent = self.env.agents[handle]

        # Get agent speed to find the multiplier for poppoing up new path
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

        for option in range(0, 5 if option_size > 5 else option_size):
            if option_size > 5:
                # select a bypass randomly to evaluate
                selected_bypass = random.randint(0,len(dict_temp["preprocessed_observation"][handle]["bypass"])-1)
            else:
                selected_bypass = option


            unique_a = [str(int(item[0]))+","+str(int(item[1])) for item in dict_temp["preprocessed_observation"][handle]["path"][handle]]
            unique_a, idx = np.unique(unique_a, axis=0, return_index=True)
            unique_a = unique_a[np.argsort(idx)]
            where = np.where(unique_a == '0,0')
            unique_a = np.delete(unique_a, where, axis=0)

            unique_b, idy = np.unique(dict_temp["preprocessed_observation"][handle]["bypass"][selected_bypass], axis=0, return_index=True)
            unique_b = unique_b[np.argsort(idy)]
            where = np.where(unique_b == '0,0')
            unique_b = np.delete(unique_b, where, axis=0)

            i_section = [0 if x == y else 1 for x,y in zip(unique_a, unique_b)]

            # proceed if the path is different from bypass
            if np.sum(i_section) > 0:
                local_path_bunch = dict_temp["preprocessed_observation"][handle]["path"].copy()
                # copy baypass as new path
                for i in range(0, len(dict_temp["preprocessed_observation"][handle]["bypass"][selected_bypass])):

                    temp_val = dict_temp["preprocessed_observation"][handle]["bypass"][selected_bypass][i].split(",")
                    local_path_bunch[handle][i*repeat:(i+1)*repeat] = [int(temp_val[0]), int(temp_val[1])]

                    if i == len(dict_temp["preprocessed_observation"][handle]["bypass"][selected_bypass])-1 and i < 200:
                        local_path_bunch[handle][(i + 1) * repeat:200] = [0,0]

                # now compute  insert this new set of path
                # as path set for all the agents
                # and compute overlap
                # replace the overlap that is already there
                for i in range(0, len(dict_temp["preprocessed_observation"])):
                    # change path
                    dict_temp["preprocessed_observation"][i]["path"] = local_path_bunch.copy()
                    # change overlap
                    dict_temp["preprocessed_observation"][i]["overlap_new"] = self._compute_overlapping_paths1(i, dict_temp["preprocessed_observation"][i]["path"])


                # Once all overlaps are calculated
                # find the cost of new set of data
                alt_cost_dict = self.evaluate_cost(dict_temp)

                if np.sum(alt_cost_dict) < best_cost:
                    #print(" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Better Path found", best_cost, np.sum(alt_cost_dict), alt_cost_dict)
                    return dict_temp, np.sum(alt_cost_dict)

        return dict_temp_temp, best_cost


    def evaluate_cost(self, dict_temp):
        temp = dict_temp["preprocessed_observation"]
        cost_mat = np.zeros(len(dict_temp["preprocessed_observation"]))
        for item in temp:
            cost = 0
            for item1 in temp[item]["overlap_new"]:
                cost += np.sum(item1, axis=0)
            cost_mat[item] = cost
        return cost_mat


    def _bypass_dict(self, path, handle):

        obs_graph = defaultdict(list)  # Dict node (as pos) : adjacent nodes
        if np.isnan(path[0][0]):
            return obs_graph

        # find only unique of them
        unique_a, idx = np.unique(path,axis=0, return_index=True)
        unique_a = unique_a[np.argsort(idx)][::-1]
        where = np.where((unique_a == (0.0, 0.0)).all(axis=1))
        unique_a = np.delete(unique_a, where, axis=0)

        if len(unique_a) <= 2:
            return obs_graph

        target = str(int(unique_a[-1][0]))+","+str(int(unique_a[-1][1]))

        visited_nodes = set()  # set
        bfs_queue = []

        initial = str(int(unique_a[0][0]))+","+str(int(unique_a[0][1]))
        adjacent = str(int(unique_a[1][0]))+","+str(int(unique_a[1][1]))

        #initial = str(int(unique_a[0][0]))+","+str(int(unique_a[0][1]))
        #adjacent = str(int(unique_a[0][0]))+","+str(int(unique_a[0][1]))
        #adjacent = str(int(unique_a[1][0]))+","+str(int(unique_a[1][1]))

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
    ######################################## Fundamental Data for Test Battery #########################################
    ####################################################################################################################

    def _compute_overlapping_paths(self,handle, state):

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


    ############################################### Needs Prediction ###################################################

    # paths with zeros substituted at the end
    # needs prediction
    def path(self, handle):
        path = np.zeros((len(self.predicted_pos_coord)
                         ,self.predicted_pos_coord[0][0].shape[0]), dtype=np.int8)

        agent = self.env.agents[handle]

        # Get agent speed to find the multiplier for poppoing up new path
        speed = agent.speed_data["speed"]
        repeat = 1
        if float("{0:.2f}".format(speed)) == 1.0:
            repeat = 1
        elif float("{0:.2f}".format(speed)) == 0.50:
            repeat = 2
        elif float("{0:.2f}".format(speed)) == 0.33:
            repeat = 3
        elif float("{0:.2f}".format(speed)) == 0.25:
            repeat = 4

        temp = np.zeros((len(self.predicted_pos_coord),2))

        for item in self.predicted_pos_coord.keys():
            temp[int(item)] = self.predicted_pos_coord[item][handle]

        unique_a = np.unique(temp,axis=0)
        unique_count = len(unique_a)

        for index1 in range(0,len(self.predicted_pos_coord)):
            if index1 < unique_count*repeat:
                path[index1][0] = int(self.predicted_pos_coord[index1][handle][0])
                path[index1][1] = int(self.predicted_pos_coord[index1][handle][1])

        return path

    # needs prediction
    def absolute_dir_dict(self, handle):
        direction = {}
        for item1 in self.env.dev_pred_dict[handle]:
            direction[str(item1[0])+","+str(item1[1])] = item1[2]
        return direction

    # needs prediction
    def _predicted_conflicts(self):
        """
        Predict conflicts for all agents
        :return:
        """
        ca_ids = []
        conflicting_agents = {a: set() for a in range(self.env.get_num_agents())}
        for ts in range(self.max_prediction_depth - 1):  # prediction_depth times
            # Check for a conflict between ALL agents comparing current time step, previous and following
            pre_ts = max(0, ts - 1)
            post_ts = min(self.max_prediction_depth - 1, ts + 1)

            # Find conflicting agent ids
            # Find intersection of cells between two prediction (cell x == cell y) excluding conflicts with same agent id and there are conditions for conflict (is_conflict())
            ts_ca_ids = [(i, j) for i, x in enumerate(self.predicted_pos[ts]) for j, y in
                         enumerate(self.predicted_pos[ts]) if x == y and i != j and self.is_conflict(ts, ts, i, j)]
            pre_ts_ca_ids = [(i, j) for i, x in enumerate(self.predicted_pos[ts]) for j, y in
                             enumerate(self.predicted_pos[pre_ts]) if
                             x == y and i != j and self.is_conflict(ts, pre_ts, i, j)]
            post_ts_ca_ids = [(i, j) for i, x in enumerate(self.predicted_pos[ts]) for j, y in
                              enumerate(self.predicted_pos[post_ts]) if
                              x == y and i != j and self.is_conflict(ts, post_ts, i, j)]

            ca_ids = list(itertools.chain(ca_ids, ts_ca_ids, pre_ts_ca_ids, post_ts_ca_ids))
        # Unique
        ca_ids = list(set(ca_ids))
        for ca_pair in ca_ids:
            conflicting_agents[ca_pair[0]].add(ca_pair[1])

        return conflicting_agents

    # needs prediction
    def _predicted_conflicts_2(self):
        """

        :return:
        """
        conflicting_agents = {a: set() for a in range(self.env.get_num_agents())}
        ca_ids = []
        np_predicted_pos = np.zeros((self.max_prediction_depth + 1, self.env.get_num_agents()))
        for ts in range(self.max_prediction_depth):
            np_predicted_pos[ts] = np.array(self.predicted_pos[ts])

        for ts in range(self.max_prediction_depth - 1):
            pre_ts = max(0, ts - 1)
            post_ts = min(self.max_prediction_depth - 1, ts + 1)
            '''
            ts_pos_array = np.array(self.predicted_pos[ts])
            pre_ts_pos_array = np.array(self.predicted_pos[pre_ts])
            post_ts_pos_array = np.array(self.predicted_pos[post_ts])
            '''

            el, ind1, ind2 = np.intersect1d(np_predicted_pos[ts], np_predicted_pos[ts], return_indices=True)
            ts_ids = [(ind1[e], ind2[e]) for e in range(len(el)) if
                      ind1[e] != ind2[e] and self.is_conflict(ts, ts, ind1[e], ind2[e])]
            el, ind1, ind2 = np.intersect1d(np_predicted_pos[ts], np_predicted_pos[pre_ts], return_indices=True)
            pre_ts_ids = [(ind1[e], ind2[e]) for e in range(len(el)) if
                          ind1[e] != ind2[e] and self.is_conflict(ts, pre_ts, ind1[e], ind2[e])]
            el, ind1, ind2 = np.intersect1d(np_predicted_pos[ts], np_predicted_pos[post_ts], return_indices=True)
            post_ts_ids = [(ind1[e], ind2[e]) for e in range(len(el)) if
                           ind1[e] != ind2[e] and self.is_conflict(ts, post_ts, ind1[e], ind2[e])]
            ca_ids = list(itertools.chain(ca_ids, ts_ids, pre_ts_ids, post_ts_ids))
        # Unique
        ca_ids = list(set(ca_ids))
        for ca_pair in ca_ids:
            conflicting_agents[ca_pair[0]].add(ca_pair[1])
        return conflicting_agents

    # needs prediction
    def is_conflict(self, ts1, ts2, handle, ca):

        if self.env.agents[handle].status in [RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART]:
            cell_pos = self.predicted_pos_coord[ts1][handle]
            int_direction = int(self.predicted_dir[ts1][handle])
            cell_transitions = self.env.rail.get_transitions(int(cell_pos[0]), int(cell_pos[1]), int_direction)

            if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                if self.predicted_dir[ts1][handle] != self.predicted_dir[ts2][ca] and cell_transitions[
                    self._reverse_dir(self.predicted_dir[ts2][ca])] == 1:
                    if not (self._is_following(ca, handle)):
                        return True

        return False



    ####################################################################################################################
    #################################### Generate Actions in Usable Format #############################################
    ####################################################################################################################

    """
    def choose_railenv_action(self, handle, network_action):
        #Choose action to perform from RailEnvActions, namely follow shortest path or stop if DQN network said so.

        #:param handle:
        #:param network_action:
        #:return:

        if network_action == 1:
            return RailEnvActions.STOP_MOVING
        else:
            return self._get_shortest_path_action(handle)


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

                agent_dir = agent.direction

                agent_pos = agent.position
                agent_pos_x = agent_pos[0]
                agent_pos_y = agent_pos[1]

                next_pos = self.alternate_observation_dict["preprocessed_observation"][handle]["path"][handle][0]

                # check agents current direction
                # check agents next point

                # if current dir is 0 then
                #   and y next is less than y current - left
                #   and y next is greater than y current - right
                #   and x next is less than x current - forward
                if agent_dir == 0:
                    if next_pos[1] < agent_pos_y:
                        action = RailEnvActions.MOVE_LEFT
                    elif next_pos[1] > agent_pos_y:
                        action = RailEnvActions.MOVE_RIGHT
                    elif next_pos[0] < agent_pos_x:
                        action = RailEnvActions.MOVE_FORWARD
                    else:
                        action = RailEnvActions.DO_NOTHING
                        print(handle, agent_dir, agent_pos, next_pos)



                # if current dir is 1 then
                #   and x next is less than x current - left
                #   and x next is greater than x current - right
                #   and y next is greater than y current - forward
                elif agent_dir == 1:
                    if next_pos[0] < agent_pos_x:
                        action = RailEnvActions.MOVE_LEFT
                    elif next_pos[0] > agent_pos_x:
                        action = RailEnvActions.MOVE_RIGHT
                    elif next_pos[1] > agent_pos_y:
                        action = RailEnvActions.MOVE_FORWARD
                    else:
                        action = RailEnvActions.DO_NOTHING
                        print(handle, agent_dir, agent_pos, next_pos)



                # if current dir is 2 then
                #   and y next is greater than y current - left
                #   and y next is less than y current - right
                #   and x next is greater than x current - forward
                elif agent_dir == 2:
                    if next_pos[1] > agent_pos_y:
                        action = RailEnvActions.MOVE_LEFT
                    elif next_pos[1] < agent_pos_y:
                        action = RailEnvActions.MOVE_RIGHT
                    elif next_pos[0] > agent_pos_x:
                        action = RailEnvActions.MOVE_FORWARD
                    else:
                        action = RailEnvActions.DO_NOTHING
                        print(handle, agent_dir, agent_pos, next_pos)



                # if current dir is 3 then
                #   and x next is greater than x current - left
                #   and x next is less than x current - right
                #   and y next is less than y current - forward
                elif agent_dir == 3:
                    if next_pos[0] > agent_pos_x:
                        action = RailEnvActions.MOVE_LEFT
                    elif next_pos[0] < agent_pos_x:
                        action = RailEnvActions.MOVE_RIGHT
                    elif next_pos[1] < agent_pos_y:
                        action = RailEnvActions.MOVE_FORWARD
                    else:
                        action = RailEnvActions.DO_NOTHING
                        print(handle, agent_dir, agent_pos, next_pos)

                else:
                    print(handle, agent_dir, agent_pos, next_pos)

        else:  # If status == DONE or DONE_REMOVED
            action = RailEnvActions.DO_NOTHING

        return action
    """


    def _get_shortest_path_action1(self, handle):
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

            #shortest_paths = self.predictor.get_shortest_paths()

            #if shortest_paths[handle] is None:  # Railway disrupted
            #    action = RailEnvActions.STOP_MOVING

            #else:

            agent_dir = agent.direction

            agent_pos = agent.position
            agent_pos_x = agent_pos[0]
            agent_pos_y = agent_pos[1]

            val_cur_pos = self.path_to_dest[handle][agent_pos_x][agent_pos_y]

            next_pos = agent_pos
            # in memory map check surrounding to find next cell
            if val_cur_pos < self.path_to_dest[handle][agent_pos_x-1][agent_pos_y]:
                next_pos = (agent_pos_x-1, agent_pos_y)
            elif val_cur_pos < self.path_to_dest[handle][agent_pos_x][agent_pos_y-1]:
                next_pos = (agent_pos_x, agent_pos_y-1)
            elif agent_pos_x + 1 < self.env.width and agent_pos_y + 1 < self.env.height:
                if val_cur_pos < self.path_to_dest[handle][agent_pos_x][agent_pos_y+1]:
                    next_pos = (agent_pos_x, agent_pos_y+1)
                elif val_cur_pos < self.path_to_dest[handle][agent_pos_x+1][agent_pos_y]:
                    next_pos = (agent_pos_x+1, agent_pos_y)

            #next_pos = self.alternate_observation_dict["preprocessed_observation"][handle]["path"][handle][0]

            # check agents current direction
            # check agents next point

            # if current dir is 0 then
            #   and y next is less than y current - left
            #   and y next is greater than y current - right
            #   and x next is less than x current - forward
            if agent_dir == 0:
                if next_pos[1] < agent_pos_y:
                    action = RailEnvActions.MOVE_LEFT
                elif next_pos[1] > agent_pos_y:
                    action = RailEnvActions.MOVE_RIGHT
                elif next_pos[0] < agent_pos_x:
                    action = RailEnvActions.MOVE_FORWARD
                else:
                    action = RailEnvActions.DO_NOTHING
                    #print(handle, agent_dir, agent_pos, next_pos)



            # if current dir is 1 then
            #   and x next is less than x current - left
            #   and x next is greater than x current - right
            #   and y next is greater than y current - forward
            elif agent_dir == 1:
                if next_pos[0] < agent_pos_x:
                    action = RailEnvActions.MOVE_LEFT
                elif next_pos[0] > agent_pos_x:
                    action = RailEnvActions.MOVE_RIGHT
                elif next_pos[1] > agent_pos_y:
                    action = RailEnvActions.MOVE_FORWARD
                else:
                    action = RailEnvActions.DO_NOTHING
                    #print(handle, agent_dir, agent_pos, next_pos)



            # if current dir is 2 then
            #   and y next is greater than y current - left
            #   and y next is less than y current - right
            #   and x next is greater than x current - forward
            elif agent_dir == 2:
                if next_pos[1] > agent_pos_y:
                    action = RailEnvActions.MOVE_LEFT
                elif next_pos[1] < agent_pos_y:
                    action = RailEnvActions.MOVE_RIGHT
                elif next_pos[0] > agent_pos_x:
                    action = RailEnvActions.MOVE_FORWARD
                else:
                    action = RailEnvActions.DO_NOTHING
                    #print(handle, agent_dir, agent_pos, next_pos)



            # if current dir is 3 then
            #   and x next is greater than x current - left
            #   and x next is less than x current - right
            #   and y next is less than y current - forward
            elif agent_dir == 3:
                if next_pos[0] > agent_pos_x:
                    action = RailEnvActions.MOVE_LEFT
                elif next_pos[0] < agent_pos_x:
                    action = RailEnvActions.MOVE_RIGHT
                elif next_pos[1] < agent_pos_y:
                    action = RailEnvActions.MOVE_FORWARD
                else:
                    action = RailEnvActions.DO_NOTHING
                    #print(handle, agent_dir, agent_pos, next_pos)

                #else:
                #    print(handle, agent_dir, agent_pos, next_pos)

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

            ####################################################################################################################


    ####################################### Auxilary Functions and Utility #############################################
    ####################################################################################################################

    @staticmethod
    def _reverse_dir(direction):
        """
        Invert direction (int) of one agent.
        :param direction:
        :return:
        """
        return int((direction + 2) % 4)


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

