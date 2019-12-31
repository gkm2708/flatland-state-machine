# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import sys
import numpy as np

#import torch
#from tqdm import trange
#from knockknock import telegram_sender
#from pathlib import Path

base_dir = "/home/gaurav/flatland-state-machine"
sys.path.append(base_dir)

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_file, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv

#from src.state_machine import act, act_new
from src.state_machine import stateMachine, Trigger


def log(s):
	with np.printoptions(threshold=np.inf):
		print(s)
		f = open("log.txt","a")
		f.write(s)
		f.close()


def main(args, dir):
	'''
	
	:param args: 
	:return: 
	Episodes to debug (set breakpoint in episodes loop to debug):
	- ep = 3, agent 1 spawns in front of 3, blocking its path; 0 and 2 are in a deadlock since they have same priority
	- ep = 4, agents stop because of wrong priorities even though the conflict zone wasn't entered,
	- ep = 14, 
	'''
	rail_generator = sparse_rail_generator(max_num_cities=args.max_num_cities,
	                                       seed=args.seed,
	                                       grid_mode=args.grid_mode,
	                                       max_rails_between_cities=args.max_rails_between_cities,
	                                       max_rails_in_city=args.max_rails_in_city,
	                                       )
	
	# Maps speeds to % of appearance in the env
	speed_ration_map = {1.: 0.25,  # Fast passenger train
	                    1. / 2.: 0.25,  # Fast freight train
	                    1. / 3.: 0.25,  # Slow commuter train
	                    1. / 4.: 0.25}  # Slow freight train

	observation_builder = GraphObsForRailEnv(predictor=ShortestPathPredictorForRailEnv(max_depth=args.prediction_depth),
	                                         bfs_depth=4)

	env = RailEnv(width=args.width,
	              height=args.height,
	              rail_generator= rail_generator,
	              schedule_generator=sparse_schedule_generator(speed_ration_map),
	              number_of_agents=args.num_agents,
	              obs_builder_object=observation_builder,
	              malfunction_generator_and_process_data=malfunction_from_params(
		              parameters={
		              'malfunction_rate': args.malfunction_rate,  # Rate of malfunction occurrence
		              'min_duration': args.min_duration,  # Minimal duration of malfunction
		              'max_duration': args.max_duration  # Max duration of malfunction
	              }))

	env_renderer = RenderTool(
		env,
		agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
		show_debug=True)

	sm = stateMachine()
	trigger = Trigger()

	state_machine_action_dict = {}
	railenv_action_dict = {}
	# max_time_steps = env.compute_max_episode_steps(args.width, args.height)
	max_time_steps = 200
	T_rewards = []  # List of episodes rewards
	T_Qs = []  # List of q values
	T_num_done_agents = []  # List of number of done agents for each episode
	T_all_done = []  # If all agents completed in each episode

	if args.save_image and not os.path.isdir("image_dump"):
		os.makedirs("image_dump")

	step_taken = 0
	total_step_taken = 0
	total_episodes = 0

	for ep in range(args.num_episodes):
		# Reset info at the beginning of an episode

		if args.generate_baseline:
			if not os.path.isdir("image_dump/"+str(dir)) and args.save_image:
				os.makedirs("image_dump/"+str(dir))
		else:
			if not os.path.isdir("image_dump/"+str(ep)) and args.save_image:
				os.makedirs("image_dump/"+str(ep))

		state, info = env.reset()
		env_renderer.reset()
		reward_sum, all_done = 0, False  # reward_sum contains the cumulative reward obtained as sum during the steps
		num_done_agents = 0

		trigger.setGlobals(args.prediction_depth, env.get_num_agents())

		for step in range(max_time_steps):

			if step % 10 == 0:
				print(step)

			state_machine_action = sm.act(state) # State machine picks action
			for a in range(env.get_num_agents()):
				railenv_action = observation_builder.choose_railenv_action(a, state_machine_action[a])
				state_machine_action_dict.update({a: state_machine_action})
				railenv_action_dict.update({a: railenv_action})

			state, reward, done, info = env.step(railenv_action_dict)  # Env step

			if args.generate_baseline:
				env_renderer.render_env(show=True, show_observations=False, show_predictions=True)
			else:
				env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

			if args.generate_baseline:
				if args.save_image:
					env_renderer.save_image("image_dump/"+str(dir)+"/image_"+str(step)+"_.png")
			else:
				if args.save_image:
					env_renderer.save_image("image_dump/"+str(ep)+"/image_"+str(step)+"_.png")


			if args.debug:
				for a in range(env.get_num_agents()):
					log('\n\n#########################################')
					log('\nInfo for agent {}'.format(a))
					#log('\npath : {}'.format(state[a]["path"]))
					log('\noverlap : {}'.format(state[a]["overlap"]))
					log('\ndirection : {}'.format(state[a]["direction"]))
					log('\nOccupancy, first layer: {}'.format(state[a]["occupancy"]))
					log('\nOccupancy, second layer: {}'.format(state[a]["conflict"]))
					log('\nForks: {}'.format(state[a]["forks"]))
					log('\nTarget: {}'.format(state[a]["target"]))
					log('\nPriority: {}'.format(state[a]["priority"]))
					log('\nMax priority encountered: {}'.format(state[a]["max_priority"]))
					log('\nNum malfunctioning agents (globally): {}'.format(state[a]["n_malfunction"]))
					log('\nNum agents ready to depart (globally): {}'.format(state[a]["ready_to_depart"]))
					log('\nStatus: {}'.format(info['status'][a]))
					log('\nPosition: {}'.format(env.agents[a].position))
					log('\nTarget: {}'.format(env.agents[a].target))
					log('\nMoving? {} at speed: {}'.format(env.agents[a].moving, info['speed'][a]))
					log('\nAction required? {}'.format(info['action_required'][a]))
					log('\nState machine action: {}'.format(state_machine_action_dict[a]))
					log('\nRailenv action: {}'.format(railenv_action_dict[a]))
					log('\nRewards: {}'.format(reward[a]))
					log('\n\n#########################################')

			reward_sum += sum(reward[a] for a in range(env.get_num_agents()))

			step_taken = step

			if done['__all__']:
				all_done = True
				break

		total_step_taken += step_taken
		total_episodes = ep

		# No need to close the renderer since env parameter sizes stay the same
		T_rewards.append(reward_sum)
		# Compute num of agents that reached their target
		for a in range(env.get_num_agents()):
			if done[a]:
				num_done_agents += 1
		percentage_done_agents = num_done_agents / env.get_num_agents()
		log("\nDone agents in episode: {}".format(percentage_done_agents))
		T_num_done_agents.append(percentage_done_agents)  # In proportion to total
		T_all_done.append(all_done)
		
	# Average number of agents that reached their target
	avg_done_agents = sum(T_num_done_agents) / len(T_num_done_agents)  if len(T_num_done_agents) > 0 else 0
	avg_reward = sum(T_rewards) / len(T_rewards) if len(T_rewards) > 0 else 0
	avg_norm_reward = avg_reward / (max_time_steps / env.get_num_agents())

	if total_episodes == 0:
		total_episodes = 1

	log("\nSeed: " + str(args.seed) \
			+ "\tAvg_done_agents: " + str(avg_done_agents)\
			+ "\tAvg_reward: " + str(avg_reward)\
			+ "\tAvg_norm_reward: " + str(avg_norm_reward)\
			+ "\tMax_time_steps: " + str(max_time_steps)\
			+ "\tAvg_time_steps: " + str(total_step_taken/total_episodes))

	
if __name__ == '__main__':

	log("\n\n\n ########################## FRESH RUN ##########################\n\n\n")

	parser = argparse.ArgumentParser(description='State machine')
	# Env parameters
	parser.add_argument('--network-action-space', type=int, default=2, help='Number of actions allowed in the environment')
	parser.add_argument('--width', type=int, default=20, help='Environment width')
	parser.add_argument('--height', type=int, default=20, help='Environment height')
	parser.add_argument('--num-agents', type=int, default=4, help='Number of agents in the environment')
	parser.add_argument('--max-num-cities', type=int, default=2, help='Maximum number of cities where agents can start or end')
	parser.add_argument('--seed', type=int, default=98, help='Seed used to generate grid environment randomly')
	parser.add_argument('--grid-mode', type=bool, default=True, help='Type of city distribution, if False cities are randomly placed')
	parser.add_argument('--max-rails-between-cities', type=int, default=3, help='Max number of tracks allowed between cities, these count as entry points to a city')
	parser.add_argument('--max-rails-in-city', type=int, default=3, help='Max number of parallel tracks within a city allowed')
	parser.add_argument('--malfunction-rate', type=int, default=2000, help='Rate of malfunction occurrence of single agent')
	parser.add_argument('--min-duration', type=int, default=0, help='Min duration of malfunction')
	parser.add_argument('--max-duration', type=int, default=0, help='Max duration of malfunction')
	parser.add_argument('--observation-builder', type=str, default='GraphObsForRailEnv', help='Class to use to build observation for agent')
	parser.add_argument('--predictor', type=str, default='ShortestPathPredictorForRailEnv', help='Class used to predict agent paths and help observation building')
	parser.add_argument('--prediction-depth', type=int, default=200, help='Prediction depth for shortest path strategy, i.e. length of a path')
	parser.add_argument('--num-episodes', type=int, default=1, help='Number of episodes to run')
	parser.add_argument('--debug', action='store_true', default=False, help='Print debug info')
	parser.add_argument('--generate-baseline', type=str, default='', help='--generate-baseline 6,12,13,14')
	parser.add_argument('--save-image', type=int, default=1, help='Save image')

	args = parser.parse_args()

	if len(args.generate_baseline) > 0:
		parser.set_defaults(num_episodes=1)
		"""
		for i in range(1,100):
			parser.set_defaults(seed=i)
			args = parser.parse_args()
			main(args, i)
		"""
		for i in args.generate_baseline.split(','):
			i = int(i)
			parser.set_defaults(seed=i)
			args = parser.parse_args()
			main(args, i)

	else:
		args = parser.parse_args()
		main(args, 0)
