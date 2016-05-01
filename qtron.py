import numpy as np

""" The Qtron represents one action """
class Qtron(object):

    def __init__(self, size, alpha, gamma):
        """ size is how big the input pixel array is, so with a 20x20 RGB pixel
            frame the size is 20*20*3
            alpha is learning rate and gamma is discount of future reward """

        self.weights = self.init_weights(size)
        self.alpha = alpha
        self.gamma = gamma
        self.value = 0.0

    def init_weights(self, size):
        """ creates random weights between -0.5 and 0.5 for all inputs and
            the bias added in forward_pass() """

        return np.random.uniform(-0.5, 0.5, size + 1)

    def get_max_q(self, actions, next_state):
        """ iterates over the qtrons in the 'actions' dictionary to find the
            one with highest q-value """

        action_values = [ qtron.forward_pass(next_state) for qtron in actions.values() ]
        return max(action_values)

    def update(self, current_state, next_state, reward, actions):
        """ actions is a dictionary of all qtrons
            max value for next state is calculated and used to update weights
            with back_propagate. new value for this qtron is then calculated """

        maxQ = self.get_max_q(actions, next_state)
        self.back_propagate(reward, maxQ)
        self.value = self.forward_pass(current_state)

    def sigmoid(self, x):
        """ returns the sigmoid value of input 'x' """

        return 1.0 / ( 1.0 + np.exp(-x) )

    def forward_pass(self, state):
        """ calculates q-value for input RGB pixel array 'state' """

        biased_state = np.append(state, 1) # adds bias input

        # dot product returns scalar when inputs are 1d arrays
        return self.sigmoid( np.dot(self.weights, biased_state) )

    def back_propagate(self, reward, maxQ):
        """ computes error and backpropagates it to update weights """

        error = self.alpha * (reward + self.gamma*maxQ - self.value)

        # sigmoid derivate is sigmoid(x) * (1 - sigmoid(x) )
        dsig = self.value * (1 - self.value)

        gradient = error * dsig

        # gradient is multiplied element-wise with weights, result is then added
        self.weights = np.add( self.weights, np.multiply(gradient, self.weights) )
