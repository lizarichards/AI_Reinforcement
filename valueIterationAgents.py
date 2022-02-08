# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        next_values = self.values.copy()
        for iter_nums in range(self.iterations):
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                next_values[state] = max([self.getQValue(state, action)
                                          for action in self.mdp.getPossibleActions(state)])
            # we need to update the global value dict in time,
            # computeQValue will use the newest values of states
            self.values = next_values.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        action_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0
        for next_state, prob in action_prob_pairs:
            reward = self.mdp.getReward(state, action, next_state)
            total += prob * (reward + self.discount * self.values[next_state])
        return total

        # q_value = 0.0
        # for next_state_and_prob in self.mdp.getTransitionStatesAndProbs(state, action):
        #     s_prime = next_state_and_prob[0]
        #     trans_prob = next_state_and_prob[1]
        #     reward = self.mdp.getReward(state, action, s_prime)
        #     value_prime = self.values[s_prime]  # Use prev_value to enable batch update
        #     q_value += trans_prob * (reward + self.discount * value_prime)
        # return q_value


        # action_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action)
        # total = 0
        # for next_state, prob in action_prob_pairs:
        #     reward = self.mdp.getReward(state, action, next_state)
        #     total += prob * (reward + self.discount * self.values[next_state])
        # return total

        # value = 0
        # transitionFunction = self.mdp.getTransitionStatesAndProbs(state, action)
        # for nextState, probability in transitionFunction:
        #     value += probability * (self.mdp.getReward(state, action, nextState)
        #                             + (self.discount * self.values[nextState]))
        #
        # return value

        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        best_action = None
        max_val = float("-inf")
        for action in self.mdp.getPossibleActions(state):
          q_value = self.computeQValueFromValues(state, action)
          if q_value > max_val:
            max_val = q_value
            best_action = action
        return best_action

        # max_q_value = -1
        # policy = None
        # for action in self.mdp.getPossibleActions(state):
        #     q_value = self.computeQValueFromValues(state, action)
        #     if max_q_value == -1 or max_q_value < q_value:
        #         max_q_value = q_value
        #         policy = action
        # return policy

        # possibleActions = self.mdp.getPossibleActions(state)
        #
        # if len(possibleActions) == 0:
        #     return None
        #
        # value = None
        # result = None
        # for action in possibleActions:
        #     temp = self.computeQValueFromValues(state, action)
        #     if value == None or temp > value:
        #         value = temp
        #         result = action
        #
        # return result

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

