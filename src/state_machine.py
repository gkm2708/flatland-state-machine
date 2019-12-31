import numpy as np
import time

import random
from typing import Optional, List, Dict, Tuple
from flatland.core.env import Environment
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4_utils import get_new_position, get_direction, Grid4TransitionsEnum
from flatland.core.transition_map import GridTransitionMap

class stateMachine():

	def __init__(self):
		print("Init")

	def act(self, triggers):
		"""
		:param prediction_depth:
		:param state: observation of one agent
		:return: 0 to follow shortest path, 1 to stop
		"""

		#check all conflicting agents with a
		decision_dict = {0:1, 1:1, 2:1, 3:1}
		for a in triggers:
			local_decision_vector = np.zeros(len(triggers))
			for ca in triggers[a]:
				decision = 1
				if decision_dict[ca] == 1:
					if triggers[a][ca]["A_entering_CZ"] == 0 or  triggers[a][ca]["A_already_in_CZ"] == 1:
						decision = 0
					elif triggers[a][ca]["A_entering_CZ"] == 1:
						if triggers[a][ca]["B_already_in_CZ"] == 1:
							if triggers[a][ca]["B_direction_in_CZ"] == "Same":
								decision = 0
							else:
								decision = 1
						elif triggers[a][ca]["B_entering_CZ"] == 1:
							if triggers[a][ca]["B_direction_in_CZ"] == "Same":
								decision = 0
							else:
								if triggers[a][ca]["speed"] == "Equal" and decision_dict[ca] == 1:
									decision = random.randint(0,1)
								else:
									decision = 0
						else:
							decision = 0
				local_decision_vector[ca] = decision
			decision_dict[a] = np.max(local_decision_vector)

		#print(decision_dict)


		return decision_dict


###############################################################################


class Trigger():

	def __init__(self):
		print("Init")

	def setGlobals(self, prediction_depth, agent_count):
		self.prediction_depth = prediction_depth
		self.agent_count = agent_count

	def check_and_trigger(self, state):
		print("Checking Triggers")
		triggers = {}
		for a in range(0, self.agent_count):
			print("Checking Triggers")
			triggers[a] = self._per_agent_trigger(state[a], a)
		return triggers

	def _per_agent_trigger(self, state, handle):

		# Single values

		conflict_all = np.zeros((self.agent_count, self.prediction_depth))

		for j in range(0, self.agent_count):
			conflict = np.zeros(self.prediction_depth)

			if j != handle:
				conflict_status_vector = np.zeros(self.prediction_depth)

				# indices of surrounding agent j for overlap with main agent handle
				indices = np.where(state["overlap"][j]==1)

				# if there is a conflict
				if len(indices[0]) != 0:

					# find these points in the path of main agent
					# and reshape to remove extra one dimension
					path_coordinates = np.asarray([state["path"][handle][i] for i in indices])
					path_coordinates = path_coordinates.reshape((path_coordinates.shape[1], path_coordinates.shape[2]))

					# find only unique of them
					unique_a, idx = np.unique(path_coordinates,axis=0, return_index=True)
					unique_a = unique_a[np.argsort(idx)]

					# Look them up in the trajectory of other agent
					# This gives a vector with 1 dimension for each point
					path_coordinates_for_conflict = np.asarray([[ 1 if i[0] == item[0] and i[1] == item[1] else 0
															  for item in state["path"][j]] for i in unique_a])
					#print("path_coordinates_for_conflict", path_coordinates_for_conflict)

					# Summarize above vector
					# This gives the span of the same path that was conflicting with main agent
					# But on the trajector of other agents
					#
					# Use it to see the overlap
					#
					time_overlap_vector = np.sum(path_coordinates_for_conflict, axis=0)
					#print("time_overlap_vector",time_overlap_vector)

					# After this 2's indicate time conflict
					conflict = np.sum(np.concatenate([np.expand_dims(time_overlap_vector, axis=0),
						 np.expand_dims(state["overlap"][j],axis=0)],axis=0),axis=0)

					#print("conflict",conflict)
					# find points where conflict is happening
					conflicting_points = [ [int(y[0]), int(y[1])] if x > 1 else [0,0] for x, y in zip(conflict, state["path"][handle])]
					#print("conflicting_points",conflicting_points)
					# check these points for direction
					# for both the agents
					conflict_status_vector = np.asarray([ 0
										if state["direction"][j].get(str(int(i[0]))+","+str(int(i[1])),0)
										   == state["direction"][handle].get(str(int(i[0]))+","+str(int(i[1])),0) else 1
									  for i in conflicting_points]
									 )

					#print("conflict_status_vector",conflict_status_vector)
					#if np.sum(conflict_status_vector, axis=0) == 0:
					# No directional conflict even if theer was a conflict
					conflict = conflict_status_vector

				conflict_all[j] = conflict

		path_conflict = np.sum(conflict_all, axis=0)
