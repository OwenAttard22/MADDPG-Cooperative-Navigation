import gym
import numpy as np
import pygame
import random
from gym import spaces
from render_utils import render_screen

class Agent:
    def __init__(self, position, id):
        self.id = id
        self.position = position
        self.neighbours = []  # List to hold references to neighboring agents

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
        self.action_space = gym.spaces.Discrete(5)
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
        for i in range(self.num_agents):
            while True:
                pos = (random.randint(0, 9), random.randint(0, 9))
                if pos not in self.agent_positions:
                    agent = Agent(pos, i)
                    self.agents.append(agent)
                    self.agent_positions.append(pos)
                    break

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def assign_neighbours(self):
        """Assign nearest neighbors to each agent."""
        for agent in self.agents:
            # Calculate distance to every other agent
            distances = [
                (other_agent, self.manhattan_distance(agent.position, other_agent.position))
                for other_agent in self.agents if other_agent != agent
            ]
            # Sort agents by distance and pick the nearest n_neighbours
            distances.sort(key=lambda x: x[1])
            agent.neighbours = [neighbour[0] for neighbour in distances[:self.n_neighbours]]

    def reset(self):
        # Reset agent positions and assign target locations
        self.agent_positions.clear()
        self.init_agents()
        self.assign_neighbours()
        return self._get_observation()

    def _get_observation(self):
        # Observation includes each agent's position and nearest neighbors' positions
        observations = []
        for agent in self.agents:
            obs = list(agent.position)  # Agent's own position
            for neighbour in agent.neighbours:
                obs.extend(neighbour.position)  # Add each neighbor's position
            observations.append(obs[:self.observation_space.shape[0]])
        return np.array(observations)

    def step(self, actions):
        # Placeholder for updating agent positions based on actions
        new_observations = self._get_observation()
        rewards = np.zeros(self.num_agents)  # Placeholder for rewards
        done = False  # Placeholder for episode completion
        return new_observations, rewards, done, {}

    def render(self, epoch, episode_length, total_reward):
        render_screen(
            screen=self.screen,
            font=self.font,
            map_size=self.map_size,
            cell_size=self.cell_size,
            agent_positions=[agent.position for agent in self.agents],
            agent_colors=self.agent_colors,
            endpoints=self.endpoints,
            epoch=epoch,
            episode_length=episode_length,
            total_reward=total_reward,
            window_size=self.window_size
        )
        self.clock.tick(30)  # Limit to 30 FPS

    def close(self):
        pygame.quit()

# Initialize environment
env = CooperativeNavigationEnv()
