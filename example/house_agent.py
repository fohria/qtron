# example: getting out of the house
# the house has 10 rooms, 0-9, arranged as a circular corridor
# so from room0 you can go to room1 or 9
# room5 is the goal, and only there will you receive a reward
# a state is the current configuration of the house;
# if the agent is in room1, the house array[1] == 1, and empty rooms are 0.1

import logging
import numpy as np
import time

import os,sys,inspect  # needed for relative import of Qtron
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from qtron import Qtron


def select_optimal_action(actions):
    """ finds and returns the optimal action based on current Qtron values """

    action_value_dict = { action: qtron.value for action, qtron in actions.iteritems() }

    action_values = np.asarray(action_value_dict.values())
    action_keys = np.asarray(action_value_dict.keys())

    optimal_action = action_keys[np.argmax(action_values)]

    return optimal_action

def take_action(action, house):
    """ executes 'action' """

    agent_position = np.argmax(house)

    if action == 'up':
        if agent_position == len(house) - 1:
            house[len(house) - 1] = 0.1
            house[0] = 1
        else:
            house[agent_position] = 0.1
            house[agent_position + 1] = 1
    elif action == 'down':
        if agent_position == 0:
            house[0] = 0.1
            house[len(house) - 1] = 1
        else:
            house[agent_position] = 0.1
            house[agent_position - 1] = 1
    else:
        logging.error("that action doesn't exist. action: %s" % (action))


# logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%H:%M:%S')
#logging.disable(logging.DEBUG) # uncomment to block debug log messages

# learning parameters
parameters = {
    'alpha' : 0.1, # learning rate (wikipedia page for q-learning states this may be good..)
    'epsilon' : 0.5, # policy (exploration/exploitation): higher value, more exploration
    'gamma' : 0.9, # discount factor: close to zero = immediate rewards, close to one more longsighted
}

# how big the house should be
house_size = 40

# create actions, we can go either 'up' (0->1->2) or 'down' (0->9->8)
actions = {
    'up': Qtron(house_size, parameters),
    'down': Qtron(house_size, parameters)
}

# each training episode runs until we've received a reward
# and we need x rounds to complete training
nbr_of_rounds = 1000

step_count_collection = []

for training_round in xrange(0, nbr_of_rounds):

    logging.debug("NEW ROUND!")

    # init empty house
    house = [0.1 for x in xrange(0,house_size)]

    # init states
    states = []

    # start position is always in room 0
    house[0] = 1

    # observe initial state, states[0] and save it
    states.append(house)

    # init reward and action to nothing
    reward = 0
    action = None

    # init this rounds step count
    step_count = 0

    # continue until we've reached the goal
    while house[5] == 0.1:

        if np.random.uniform(0,1) < parameters['epsilon']:

            logging.debug("EXPLORE!")
            action = np.random.choice(actions.keys())
        else:

            logging.debug("EXPLOIT!")
            action = select_optimal_action(actions)

        take_action(action, house)
        step_count += 1

        # observe new state, states[1]
        states.append(house)

        # give reward if we've reached goal, room5
        if house[5] == 1:
            logging.debug("GOOOOOOAAAAAAAAAAAL!")
            reward = 10

        actions[action].update(states, reward, actions)

        # new state will become old state in the next action step
        del states[0]

        # print current q-values and state
        logging.debug("current q value for up: %s" % (actions['up'].value))
        logging.debug("current q value for down: %s" % (actions['down'].value))
        #logging.debug("current house state: %s" % (house))

        # reset action before next move
        action = None

    #decrease explore/exploit after each round
    if parameters['epsilon'] >= 0.2 and training_round % 100 == 0:
        logging.debug("UPDATING EPSILON! current value: %s" % (parameters['epsilon']))
        parameters['epsilon'] -= 0.1
        logging.debug("new epsilon value: %s" % (parameters['epsilon']))

    step_count_collection.append(step_count)
    #onto next round


# print step count collection to file when done
filename = "stepcount_%s.csv" % (time.time())
with open(filename, 'wb') as f:
    for step in step_count_collection:
        f.write("%s," % (step))

