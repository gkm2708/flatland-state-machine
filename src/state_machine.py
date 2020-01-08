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


		decision_dict = {}
		for i in range(len(triggers)):
			decision_dict[i] = 1


		for a in triggers:
			local_decision_vector = np.zeros(len(triggers))
			if len(triggers[a]["ca"]) > 0:

				for ca in triggers[a]["ca"]:
					decision = 1
					if decision_dict[ca] == 1:
						if triggers[a]["A_entering_CZ"] == 0 or triggers[a]["A_already_in_CZ"] == 1:
							decision = 0
						elif triggers[a]["A_entering_CZ"] == 1:
							if triggers[a]["B_already_in_CZ"] == 1:
								decision = 1
							elif triggers[a]["B_entering_CZ"] == 1:
								#decision = random.randint(0,1)
								decision = 0
							else:
								decision = 0
					local_decision_vector[ca] = decision
			else:
				local_decision_vector[a] = 0
			decision_dict[a] = np.max(local_decision_vector)

		return decision_dict


	def act2(self, triggers):
		#print("Test")
		decision_dict = {}
		for i in range(len(triggers)):
			decision_dict[i] = 0


		decision = 0
		for a in triggers:
			local_decision_vector = np.zeros(len(triggers))
			#if len(triggers[a]["ca"]) > 0:
			for ca in triggers[a]["ca"]:
				decision = 1
				if decision_dict[ca] == 1:
					if triggers[a]["A_already_in_CZ"] == 1:
						decision = 0
					elif triggers[a]["B_already_in_CZ"] == 1:
						decision = 1
					elif triggers[a]["A_entering_CZ"] == 0:
						decision = 0
					else:
						if triggers[a]["A_entering_CZ"] == 1:
							if triggers[a]["B_entering_CZ"] == 1:
								decision = random.randint(0, 1)
							else:
								decision = 0
						else:
							decision = 0
			local_decision_vector[a] = decision
			#else:
			#	decision = 0
			#	local_decision_vector[a] = decision
			decision_dict[a] = np.max(local_decision_vector)

		return decision_dict


	def act1(self, triggers1):
		"""
		:param prediction_depth:
		:param state: observation of one agent
		:return: 0 to follow shortest path, 1 to stop
		"""

		triggers = triggers1["main"]

		decision_dict = {}
		for i in range(len(triggers)):
			decision_dict[i] = 1



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

		triggers2 = triggers1["target"]
		for a in triggers2:
			if triggers2[a] == 1:
				decision_dict[a] = 0

		return decision_dict
