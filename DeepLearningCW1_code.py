import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self):
        # initialise board and obstacles
        self.num_rows = 5
        self.num_cols = 5
        self.grid = np.zeros((self.num_rows, self.num_cols))
        self.grid[2, 2:] = -1
        self.grid[3, 2] = -1

        # set initial state and winning terminal state
        self.initial_state = (1, 0)
        self.winning_state = (4, 4)

        # label states for jump
        self.jump_state = (1, 3)
        self.land_state = (3, 3)

    def give_reward(self, current_pos, nxt_pos):

        # reward for reaching terminal state
        if nxt_pos == self.winning_state:
            reward = 10

        # reward for performing jump
        elif current_pos == self.jump_state and nxt_pos == self.land_state:
            reward = 5

        # reward for any other state action pair
        else:
            reward = -1

        return reward

class QLearningAgent:
    def __init__(self, environment, alpha, epsilon, gamma):

        # initialise gridworld and agents location in environment
        self.env = environment
        self.position = self.env.initial_state

        # basic attributes
        self.allowed_actions = ["North", "South", "East", "West"]
        self.state_history = []     # container for states

        # hyper parameters
        self.learn_rate = alpha  # learning rate
        self.exp_rate = epsilon  # exploration/exploitation trade off
        self.decay_rate = gamma  # short term/long term reward trade off

        # initialise Q values
        self.Q_values = {}
        for i in range(self.env.num_rows):
            for j in range(self.env.num_cols):
                self.Q_values[(i, j)] = {}
                for a in self.allowed_actions:
                    self.Q_values[(i, j)][a] = 0.0

    def choose_action(self):

        # exploration
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.allowed_actions)

        # exploitation
        else:
            possible_actions = self.Q_values[self.position]
            max_key = max(possible_actions, key=possible_actions.get)
            action = max_key

        return action

    def position_update(self, action):

        if action == "North":
            new_agent_pos = (self.position[0] - 1, self.position[1])
        elif action == "East":
            new_agent_pos = (self.position[0], self.position[1] + 1)
        elif action == "West":
            new_agent_pos = (self.position[0], self.position[1] - 1)

        # if action == "South"
        else:
            # allows for jump from cell [2,4] to [4,4]
            if self.position == (1, 3):
                new_agent_pos = (self.position[0] + 2, self.position[1])
            else:
                new_agent_pos = (self.position[0] + 1, self.position[1])

        # check if new state is valid
        if (new_agent_pos[0] >= 0) and (new_agent_pos[0] <= self.env.num_rows - 1):
            if (new_agent_pos[1] >= 0) and (new_agent_pos[1] <= self.env.num_cols - 1):
                if self.env.grid[new_agent_pos[0], new_agent_pos[1]] != -1:
                    return new_agent_pos

        return self.position

    def train_model(self, num_episodes, num_steps, verbose):

        total_ep_reward = []

        for i in range(num_episodes):

            # initialise cumulative reward for each episode
            cum_reward = 0.0

            # check if reward condition reached to end training
            if i >= 30:
                if stat.mean(total_ep_reward[i-30:i]) > 10.0:
                    break

            for j in range(num_steps):

                self.state_history.append(self.position)

                # if terminal state reached, end episode and reset state history and position
                if self.position == self.env.winning_state:
                    self.state_history = []
                    self.position = self.env.initial_state
                    break

                # choose action
                action = self.choose_action()

                # calculate next agent position
                nxt_agent_pos = self.position_update(action)

                # get reward for moving to next state and update cumulative reward
                reward = self.env.give_reward(self.position, nxt_agent_pos)
                cum_reward += reward

                # Q update algorithm
                nxt_max_q = max(self.Q_values[nxt_agent_pos].values())
                # self.Q_values[self.position][action] = reward + (self.decay_rate * nxt_max_q)
                q_k = self.Q_values[self.position][action]  # current Q value
                alpha_comp = reward + (self.decay_rate * nxt_max_q) - q_k
                self.Q_values[self.position][action] = q_k + (self.learn_rate * alpha_comp)

                # print to screen if verbose == true
                if verbose:
                    print("current position {} action {}".format(self.position, action))
                    print("nxt state", nxt_agent_pos)
                    if nxt_agent_pos == (4, 4):
                        print(cum_reward, "\n")

                # update agent position
                self.position = nxt_agent_pos

            # record cumulative reward for training episode
            total_ep_reward.append(cum_reward)

        return total_ep_reward

def task_c():
    """
    Function to complete task C in CW1 specification
    alpha set to 1 and Q-learning algorithm run for 1 episode with 100 steps

    :return:
    state and action taken by agent at each step printed to terminal
    Q-table
    """
    alpha = 1.0
    epsilon = 0.3
    gamma = 1.0
    environment = GridWorld()

    agent = QLearningAgent(environment, alpha, epsilon, gamma)
    agent.train_model(num_episodes=1, num_steps=100, verbose=True)
    q_table = pd.DataFrame.from_dict(agent.Q_values, orient='index')
    print(q_table)

def task_e():
    """
    Function to complete task E in CW1 specification
    Q-learning algorithm run for 100 episodes each with 50 steps
    Algorithm stops when average cumulative reward is greater than 10 for 30 episodes

    :return:
    state and action taken by agent at each step printed to terminal
    Q  table
    number of episodes completed
    plot pf cumulative rewards against episode number
    """

    alpha = 0.3
    epsilon = 0.15
    gamma = 0.9
    environment = GridWorld()

    agent = QLearningAgent(environment, alpha, epsilon, gamma)
    x = agent.train_model(num_episodes=100, num_steps=50, verbose=True)
    q_table = pd.DataFrame.from_dict(agent.Q_values, orient='index')
    print(q_table)
    print(len(x))

    plt.plot(x)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.show()

def epsilon_test():
    """
    Function to graph cumulative reward vs episode for 2 Q-learning agents
    that are initialised with different epsilon values
    """
    alpha = 0.7
    epsilon = 0.7
    gamma = 1.0
    environment = GridWorld()

    agent = QLearningAgent(environment, alpha, epsilon, gamma)
    rewards1 = agent.train_model(num_episodes=100, num_steps=50, verbose=False)

    alpha = 0.7
    epsilon = 0.15
    gamma = 1.0
    environment = GridWorld()

    agent = QLearningAgent(environment, alpha, epsilon, gamma)
    rewards2 = agent.train_model(num_episodes=100, num_steps=50, verbose=False)

    plt.plot(rewards1, label='epsilon=0.7')
    plt.plot(rewards2, label='epsilon=0.15')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.show()

"""
Below shows methods to run
ONLY RUN ONE METHOD AT ONE - comment 2 of these out so only one runs at a time
"""
task_c()
task_e()
epsilon_test()
