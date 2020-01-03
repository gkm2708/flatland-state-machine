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
        # Init
        for a in handles:
            self.conflicting_agents[a] = set()
        
        # Compute conflicting agents for all handles at once, saved in self.conflicting_agents
        self.conflicting_agents = self._predicted_conflicts()
        
        observations = {}
        for a in handles:
            observations[a] = self.get(a)
        
        # Preprocess observation
        preprocessed_observation = {}
        for a in handles:
            preprocessed_observation[a] = self.preprocess_state(observations, a)

        dict_temp = {}
        dict_temp["preprocessed_observation"] = preprocessed_observation
        dict_temp["cells_sequence"] = self.cells_sequence


        #self.find_alternate(dict_temp)


        return dict_temp


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
        #occupancy, conflicting_agents = self._fill_occupancy(handle)
        #conflicting_agents = self._fill_occupancy(handle)

        # Augment occupancy with another one-hot encoded layer:
        # 1 if this cell is overlapping and the conflict span was already entered by some other agent
        '''
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
        '''
        # Bifurcation points, one-hot encoded layer of predicted cells where 1 means that this cell is a fork
        # (globally - considering cell transitions not depending on agent orientation)
        '''
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
        '''

        """

        ret = {}
        path = np.zeros((len(self.predicted_pos_coord[0]), len(self.predicted_pos_coord),self.predicted_pos_coord[0][0].shape[0]))

        for index in range(0,len(self.predicted_pos_coord[0])):
            for index1 in range(0,len(self.predicted_pos_coord)):
                path[index][index1][0] = self.predicted_pos_coord[index1][index][0]
                path[index][index1][1] = self.predicted_pos_coord[index1][index][1]


        ret["path"] = path
        ret["overlap_new"] = self._compute_overlapping_paths1(handle, path)
        ret["overlap_old"] = overlapping_paths
        ret["direction"] = direction

        #ret["bypass"] = self._bypass_dict_1(direction, path[handle], handle)

        #print(ret["bypass"])

        ret["conflicting_agents"] = self.conflicting_agents[handle]
        #ret["forks"] = forks
        #ret["target"] = target

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

                        dict[str(i)+","+str(j)] = np.min(self.env.distance_map.distance_map[handle][i][j])


        unique_a = unique_a[::-1]
        #print("Reverse Sequence",unique_a)

        dict1 = {}
        for item in unique_a:

            #check all the surrounding points and push the one that is lower with its distance
            #print("next node")
            for i in range(int(item[0])-1, int(item[0])+2):
                for j in range(int(item[1])-1, int(item[1])+2):

                    if np.min(self.env.distance_map.distance_map[handle][i][j]) == np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])])+1:
                        #print(i,j,np.min(self.env.distance_map.distance_map[handle][i][j]))

                        dict1[str(i)+","+str(j)] = np.min(self.env.distance_map.distance_map[handle][i][j])

        dict_main = {}
        for item in unique_a:

            #check all the surrounding points and push the one that is lower with its distance
            #print("next node")
            #print(i,j, " for ", int(item[0]), int(item[1]) )


            # find minimum of center point
            # find count of minimum value (repetition is bifurcation)
            # find position of minimum value
            # if position is
            # 0 - south
            # 1 - west
            # 2 - north
            # 3 - east

            # add to dictionary in the same way together with parent
            #

            a1 = self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])]
            print(a1)
            a2 = np.min(a1)
            print(a2)
            a3 = np.asarray([1 if item == a2 else 0 for item in a1])
            print(a3)
            a4 = np.count_nonzero(a3)
            print(a4)
            a5 = np.where(a3==1)
            print(a5)


            dict = {}
            for i in a5:
                for j in i:
                    if j == 0:
                        print("check south")
                        if np.min(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])]) - 1 == a2:
                            dict[str(int(item[0]))+","+str(int(item[1]+1))] = np.min(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])])
                    if j == 1:
                        print("check west")
                        if np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])-1]) - 1 == a2:
                            dict[str(int(item[0])-1)+","+str(int(item[1]))] = np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])-1])
                    if j == 2:
                        print("check north")
                        if np.min(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1])]) - 1 == a2:
                            dict[str(int(item[0]))+","+str(int(item[1])-1)] = np.min(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1])])
                    if j == 3:
                        print("check east")
                        if np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])+1]) - 1 == a2:
                            dict[str(int(item[0])+1)+","+str(int(item[1]))] = np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])+1])

            dict_main[str(int(item[0]))+","+str(int(item[1]))] = dict






        print(dict,"\n", dict1)

            # take first point and push in a dictionary



            #print(item, item[0], item[1])
            #print(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])])
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])]))

            # find values in adjacent cells
            #print("\n\n",item)
            #print(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1]-1)])
            #print(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])-1])
            #print(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])-1])
            #print(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1])])
            #print(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])])
            #print(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])])
            #print(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1])+1])
            #print(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])+1])
            #print(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])+1])

            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1]-1)]))
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])+1]))
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])]))
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1])-1]))
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])+1]))
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])-1][int(item[1])]))
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])][int(item[1])-1]))
            #print(np.min(self.env.distance_map.distance_map[handle][int(item[0])+1][int(item[1])+1]))


        # for all the points in the path
        # check neighbouring cells
        # if the lowest number in the cell or one above is found in a neighbouring cell
        # it is a potential fork


    def preprocess_state(self, state, handle):

        ret = {}

        ret["conflicting_agents"] = state[handle]["conflicting_agents"]
        #ret["bypass"] = state[handle]["bypass"]

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
                        break
                else:
                    repetition = run_index
                    break

            temp = np.zeros((len(self.predicted_pos_coord),2))

            for item in self.predicted_pos_coord.keys():
                #print(item)
                temp[int(item)] = self.predicted_pos_coord[item][index]

    # I need only conflicting agents from here
    def _fill_occupancy(self, handle):
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
        # overlapping_paths = self._compute_overlapping_paths(handle)

        # cells_sequence = self.cells_sequence[handle]
        # span_cells = []
        if self.env.agents[handle].status in [RailAgentStatus.READY_TO_DEPART, RailAgentStatus.ACTIVE]:
            for ts in range(self.max_prediction_depth):
                    occupancy[ts], conflicting_agents_ts = self._possible_conflict(handle, ts)
                    conflicting_agents.update(conflicting_agents_ts)

        # If a conflict is predicted, then it makes sense to populate occupancy with overlapping paths
        # But only with THAT agent
        # Because I could have overlapping paths but without conflict
        '''
        if len(conflicting_agents) != 0: # If there was conflict
            for ca in conflicting_agents:
                for ts in range(self.max_prediction_depth):
                    occupancy[ts] = overlapping_paths[ca, ts] if occupancy[ts] == 0 else 1
        '''

        # Occupancy is 0 for agents that are done - they don't perform actions anymore

        # the calculated occupancy is for the agents that have conflict and hence conflict occupancy

        # return occupancy, conflicting_agents
        return conflicting_agents

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
            conflicting_agents = np.where(self.predicted_pos[ts] == int_pos)  # x200
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

    # With lists
    def _predicted_conflicts(self):
        """
        Predict conflicts for all agents
        :return: 
        """
        ca_ids = []
        conflicting_agents = {a:set() for a in range(self.env.get_num_agents())}

        for ts in range(self.max_prediction_depth - 1): # prediction_depth times
            # Check for a conflict between ALL agents comparing current time step, previous and following 
            pre_ts = max(0, ts - 1)
            post_ts = min(self.max_prediction_depth - 1, ts + 1)
            
            # Find conflicting agent ids
            # Find intersection of cells between two prediction (cell x == cell y) excluding conflicts with same agent id and there are conditions for conflict (is_conflict())
            ts_ca_ids = [(i, j) for i, x in enumerate(self.predicted_pos[ts]) for j, y in enumerate(self.predicted_pos[ts]) if x == y and i != j and self.is_conflict(ts, ts, i, j)]
            pre_ts_ca_ids = [(i, j) for i, x in enumerate(self.predicted_pos[ts]) for j, y in enumerate(self.predicted_pos[pre_ts]) if x == y and i != j and self.is_conflict(ts, pre_ts, i, j)]
            post_ts_ca_ids =[(i, j) for i, x in enumerate(self.predicted_pos[ts]) for j, y in enumerate(self.predicted_pos[post_ts]) if x == y and i != j and self.is_conflict(ts, post_ts, i, j)]
            
            ca_ids = list(itertools.chain(ca_ids, ts_ca_ids, pre_ts_ca_ids, post_ts_ca_ids))
        # Unique
        ca_ids = list(set(ca_ids))
        for ca_pair in ca_ids:
            conflicting_agents[ca_pair[0]].add(ca_pair[1])
        
        return conflicting_agents
            
    # With numpy arrays
    def _predicted_conflicts_2(self):
        """
        
        :return: 
        """
        conflicting_agents = {a:set() for a in range(self.env.get_num_agents())}

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
            ts_ids = [(ind1[e], ind2[e]) for e in range(len(el)) if ind1[e] != ind2[e] and self.is_conflict(ts, ts, ind1[e], ind2[e])]
            el, ind1, ind2 = np.intersect1d(np_predicted_pos[ts], np_predicted_pos[pre_ts], return_indices=True)
            pre_ts_ids = [(ind1[e], ind2[e]) for e in range(len(el)) if ind1[e] != ind2[e] and self.is_conflict(ts, pre_ts, ind1[e], ind2[e])]
            el, ind1, ind2 = np.intersect1d(np_predicted_pos[ts], np_predicted_pos[post_ts], return_indices=True)
            post_ts_ids = [(ind1[e], ind2[e]) for e in range(len(el)) if ind1[e] != ind2[e] and self.is_conflict(ts, post_ts, ind1[e], ind2[e])]

            ca_ids = list(itertools.chain(ca_ids, ts_ids, pre_ts_ids, post_ts_ids))
        # Unique
        ca_ids = list(set(ca_ids))
        for ca_pair in ca_ids:
            conflicting_agents[ca_pair[0]].add(ca_pair[1])
        return conflicting_agents

    def is_conflict(self, ts1, ts2, handle, ca):
            
            if self.env.agents[handle].status in [RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART]:
                cell_pos = self.predicted_pos_coord[ts1][handle]
                int_direction = int(self.predicted_dir[ts1][handle])
                cell_transitions = self.env.rail.get_transitions(int(cell_pos[0]), int(cell_pos[1]), int_direction)
        
                if self.env.agents[ca].status == RailAgentStatus.ACTIVE:
                    if self.predicted_dir[ts1][handle] != self.predicted_dir[ts2][ca] and cell_transitions[self._reverse_dir(self.predicted_dir[ts2][ca])] == 1:
                        if not (self._is_following(ca, handle)):
                            return True
                
            return False
    """
    def _bfs_graph(self, handle: int = 0) -> {}:
        #Build a graph (dict) of nodes, where nodes are identified by ids, graph is directed, depends on agent direction
        #(that are tuples that represent the cell position, eg (11, 23))
        #:param handle: agent id
        #:return: 
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
        #Given agent handle, current position, and direction, explore that path until a new branching point is found.
        #:param handle: agent id
        #:param position: agent position as cell 
        #:param direction: agent direction
        #:return: a tuple Node with its features

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

    """

