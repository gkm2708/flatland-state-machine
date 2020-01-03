import numpy as np
import time
import sys
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent
print(str(base_dir))
sys.path.append(str(base_dir))

from flatland.evaluators.client import FlatlandRemoteClient
from src.graph_observations import GraphObsForRailEnv
from src.predictions import ShortestPathPredictorForRailEnv
from src.state_machine import stateMachine
from src.test_battery import TestBattery

prediction_depth = 20

#####################################################################
# Instantiate a Remote Client
#####################################################################
print('starting remote_client')
remote_client = FlatlandRemoteClient()
observation_builder = GraphObsForRailEnv(predictor=ShortestPathPredictorForRailEnv(max_depth=prediction_depth),bfs_depth=4)

#####################################################################
# Main evaluation loop
#
# This iterates over an arbitrary number of env evaluations
#####################################################################
evaluation_number = 0
railenv_action_dict = {}
T_rewards = []
T_num_done_agents = []
T_all_done = []

while True:

    evaluation_number += 1
    # Switch to a new evaluation environment
    # 
    # a remote_client.env_create is similar to instantiating a 
    # RailEnv and then doing a env.reset()
    # hence it returns the first observation from the 
    # env.reset()
    # 
    # You can also pass your custom observation_builder object
    # to allow you to have as much control as you wish 
    # over the observation of your choice.
    time_start = time.time()
    state, info = remote_client.env_create(
                    obs_builder_object=observation_builder
                )
    env_creation_time = time.time() - time_start
    if not state:
        #
        # If the remote_client returns False on a `env_create` call,
        # then it basically means that your agent has already been 
        # evaluated on all the required evaluation environments,
        # and hence its safe to break out of the main evaluation loop
        break
    # print("######### Start new evaluation #########")
    print("Evaluation Number : {}".format(evaluation_number), 'env_creation_time =', env_creation_time, 'number_of_agents =', len(remote_client.env.agents))

    #####################################################################
    # Access to a local copy of the environment
    # 
    #####################################################################
    # Note: You can access a local copy of the environment 
    # by using : 
    #       remote_client.env 
    # 
    # But please ensure to not make any changes (or perform any action) on 
    # the local copy of the env, as then it will diverge from 
    # the state of the remote copy of the env, and the observations and 
    # rewards, etc will behave unexpectedly
    # 
    # You can however probe the local_env instance to get any information
    # you need from the environment. It is a valid RailEnv instance.
    local_env = remote_client.env
    number_of_agents = len(local_env.agents)

    sm = stateMachine()
    tb = TestBattery(local_env)

    # Now we enter into another infinite loop where we 
    # compute the actions for all the individual steps in this episode
    # until the episode is `done`
    # 
    # An episode is considered done when either all the agents have 
    # reached their target destination
    # or when the number of time steps has exceed max_time_steps, which 
    # is defined by : 
    #
    # max_time_steps = int(4 * 2 * (env.width + env.height + 20))
    #
    max_time_steps = int(4 * 2 * (local_env.width + local_env.height + 20)) - 10 # '-10' works as an epsilon
    time_taken_by_controller = []
    time_taken_per_step = []
    steps = 0
    reward_sum = 0
    num_done_agents = 0
    all_done = False
    
    while True:

        #####################################################################
        # Evaluation of a single episode
        #
        #####################################################################
        # Compute the action for this step by using the previously 
        # defined controller
        time_start = time.time()

        # Test battery
        # see test_battery.py
        triggers = tb.tests(state, prediction_depth)
        # state machine based on triggers of test battery
        # see state_machine.py
        state_machine_action = sm.act(triggers) # State machine picks action

        for a in range(number_of_agents):
            #state_machine_action = act(prediction_depth, state[a])  # State machine picks action
            railenv_action = observation_builder.choose_railenv_action(a, state_machine_action)
            # state_machine_action_dict.update({a: state_machine_action})
            railenv_action_dict.update({a: railenv_action})
        time_taken = time.time() - time_start
        time_taken_by_controller.append(time_taken)
        # Perform the chosen action on the environment.
        # The action gets applied to both the local and the remote copy 
        # of the environment instance, and the observation is what is 
        # returned by the local copy of the env, and the rewards, and done and info
        # are returned by the remote copy of the env
        time_start = time.time()
        state, reward, done, info = remote_client.env_step(railenv_action_dict)
        steps += 1
        time_taken = time.time() - time_start
        time_taken_per_step.append(time_taken)
        reward_sum += sum(list(reward.values()))

        if steps % 1 == 0:
           print("Step / Max Steps: {}/{}".format(steps, max_time_steps), 'time_taken_by_controller', round(time_taken_by_controller[-1],3), 'time_taken_per_step', round(time_taken_per_step[-1],1), 'reward_step', round(sum(list(reward.values())),1), 'reward_sum', round(reward_sum))

        if steps > max_time_steps: # To avoid that all dones are set to 0 after reaching max_time_steps
            break
        if done['__all__']:
            # print("Reward : ", sum(list(reward.values())))
            #
            # When done['__all__'] == True, then the evaluation of this 
            # particular Env instantiation is complete, and we can break out 
            # of this loop, and move onto the next Env evaluation
            all_done = True
            break

    T_rewards.append(reward_sum)
    # Compute num of agents that reached their target  
    for a in range(number_of_agents):
        if done[a]:
            num_done_agents += 1
    percentage_done_agents = num_done_agents / number_of_agents
    T_num_done_agents.append(percentage_done_agents)
    T_all_done.append(all_done)

    np_time_taken_by_controller = np.array(time_taken_by_controller)
    np_time_taken_per_step = np.array(time_taken_per_step)
    print("="*100)
    print("="*100)
    print("Evaluation Number : ", evaluation_number)
    print("Current Env Path : ", remote_client.current_env_path)
    print("Env Creation Time : ", env_creation_time)
    print("Number of Steps : ", steps)
    print("Mean/Std of Time taken by Controller : ", np_time_taken_by_controller.mean(), np_time_taken_by_controller.std())
    print("Mean/Std of Time per Step : ", np_time_taken_per_step.mean(), np_time_taken_per_step.std())
    print("="*100)
    print('Done agents: {}'.format(percentage_done_agents))
    print('Reward: {}'.format(reward_sum))

## When all evaluations have ended
# Average number of agents that reached their target
avg_done_agents = sum(T_num_done_agents) / len(T_num_done_agents) if len(T_num_done_agents) > 0 else 0
avg_reward = sum(T_rewards) / len(T_rewards) if len(T_rewards) > 0 else 0
print("Avg. done agents: {}".format(avg_done_agents))
print("Avg. reward: {}".format(avg_reward))
print("Evaluation of all environments complete...")
########################################################################
# Submit your Results
# 
# Please do not forget to include this call, as this triggers the 
# final computation of the score statistics, video generation, etc
# and is necessary to have your submission marked as successfully evaluated
########################################################################
print(remote_client.submit())
