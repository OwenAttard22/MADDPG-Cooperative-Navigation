import gym
import numpy as np
import pygame
import random
from gym import spaces
from render_utils import *
from env_utils import *
from brain import *

class Agent:
    def __init__(self, id, position, target):
        self.id = id
        self.position = position
        self.neighbours = []
        self.target = target
        self.reward = 0
        self.flag = 0  # Test flag forcing stopping on target

class CooperativeNavigationEnv(gym.Env):
    def __init__(self, n_neighbours=2):
        super().__init__()
        self.map_size = 10
        self.cell_size = 70
        self.num_agents = 6
        self.window_size = self.map_size * self.cell_size
        self.agent_colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (75, 0, 130)]
        self.endpoints = {
            "Alpha": (3, 8),
            "Bravo": (6, 8),
            "Charlie": (4, 5),
            "Delta": (6, 2),
            "Echo": (9, 2),
            "Foxtrot": (2, 0)
        }
        self.n_neighbours = n_neighbours
        self.agent_positions = []
        
        # Action and observation space setup
        self.action_space = gym.spaces.Discrete(5)  # 0: up, 1: down, 2: left, 3: right, 4: stay
        obs_space_size = 2 + 2 + (self.n_neighbours * 2)
        self.observation_space = spaces.Box(
            low=-self.map_size,
            high=self.map_size,
            shape=(obs_space_size,),
            dtype=np.float32
        )
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 100))
        pygame.display.set_caption("Cooperative Navigation")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()

        # Initialize agents and neighbors
        self.agents = []
        self.init_agents()
        self.assign_neighbours()

    def init_agents(self):
        """Randomly initialize agents with unique positions."""
        for i, key in enumerate(self.endpoints):
            while True:
                pos = (random.randint(0, 9), random.randint(0, 9))
                if pos not in self.agent_positions:
                    agent = Agent(i, pos, self.endpoints[key])
                    self.agents.append(agent)
                    self.agent_positions.append(pos)
                    break

    def assign_neighbours(self):
        """Assign nearest neighbors to each agent."""
        for agent in self.agents:
            # Calculate distance to every other agent
            distances = [
                (other_agent, manhattan_distance(agent.position, other_agent.position))
                for other_agent in self.agents if other_agent != agent
            ]
            # Sort agents by distance and pick the nearest n_neighbours
            distances.sort(key=lambda x: x[1])
            agent.neighbours = [neighbour[0] for neighbour in distances[:self.n_neighbours]]

    def reset(self):
        self.agents.clear()
        self.init_agents()
        self.assign_neighbours()
        return get_observation(self)

    def step(self, actions):
        rewards = []
        for i, agent in enumerate(self.agents):
            action = actions[i]
            apply_action(self, agent, action)
            reward = calculate_reward(self, agent)
            agent.reward += reward
            rewards.append(reward)
        
        # Update observations after all actions
        new_observations = get_observation(self)
        
        # Check if the episode is done (e.g., all agents reach their targets)
        done = sum(manhattan_distance(agent.position, agent.target) == 0 for agent in self.agents) >= (self.num_agents - 1)
        
        return new_observations, np.array(rewards), done, {}

    def render(self, epoch, episode_length, total_reward):
        agent_rewards = [agent.reward for agent in self.agents]
        render_screen(
            screen=self.screen,
            font=self.font,
            map_size=self.map_size,
            cell_size=self.cell_size,
            agents=self.agents,
            agent_colors=self.agent_colors,
            endpoints=self.endpoints,
            epoch=epoch,
            episode_length=episode_length,
            total_reward=total_reward,
            window_size=self.window_size,
            agent_rewards=agent_rewards
        )
        self.clock.tick(30)

    def close(self):
        pygame.quit()
        
env = CooperativeNavigationEnv()