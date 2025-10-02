"""
A simple Q-learning agent that learns to navigate a grid world.
The agent starts at a random position and tries to reach the goal.
"""
import random
import numpy as np

# Define the grid world size
GRID_SIZE = 5
GOAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)
ACTIONS = ['up', 'down', 'left', 'right']

class GridWorld:
    def __init__(self, size, goal_pos):
        self.size = size
        self.goal_pos = goal_pos
        self.reset()

    def reset(self):
        # Start at a random position except the goal
        while True:
            self.agent_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.agent_pos != self.goal_pos:
                break
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.size - 1, y + 1)
        self.agent_pos = (x, y)
        # Reward is 1 if goal is reached, else 0
        reward = 1 if self.agent_pos == self.goal_pos else 0
        done = self.agent_pos == self.goal_pos
        return self.agent_pos, reward, done

class QLearningAgent:
    def __init__(self, grid_size, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((grid_size, grid_size, len(actions)))
        self.actions = actions
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        x, y = state
        action_idx = np.argmax(self.q_table[x, y])
        return self.actions[action_idx]

    def learn(self, state, action, reward, next_state):
        x, y = state
        a = self.actions.index(action)
        nx, ny = next_state
        predict = self.q_table[x, y, a]
        target = reward + self.gamma * np.max(self.q_table[nx, ny])
        self.q_table[x, y, a] += self.alpha * (target - predict)

if __name__ == "__main__":
    env = GridWorld(GRID_SIZE, GOAL_POS)
    agent = QLearningAgent(GRID_SIZE, ACTIONS)
    episodes = 100
    for ep in range(episodes):
        state = env.reset()
        steps = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1
            if done:
                print(f"Episode {ep+1}: reached goal in {steps} steps.")
                break
    print("Training complete. Example Q-table:")
    print(agent.q_table)
