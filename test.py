import pygame
import numpy as np
from main import CooperativeNavigationEnv

# Initialize environment instance
env = CooperativeNavigationEnv()

# Test parameters
epoch = 1
episode_length = 0
total_reward = 0

# Main test loop to visualize render
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Randomly select actions for each agent
    actions = [env.action_space.sample() for _ in env.agents]  # Generate random actions

    # Take a step in the environment with the random actions
    observations, rewards, done, _ = env.step(actions)

    # Accumulate rewards for visualization purposes
    total_reward += np.sum(rewards)

    # Render the environment
    env.render(epoch, episode_length, total_reward)

    # Increment test metrics
    episode_length += 1

    # Optional: Reset environment if done for continuous testing
    if done:
        env.reset()
        episode_length = 0
        total_reward = 0

# Close Pygame window after loop
env.close()
