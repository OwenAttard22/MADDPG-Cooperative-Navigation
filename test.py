import random
import pygame
from main import CooperativeNavigationEnv

# Initialize environment instance
env = CooperativeNavigationEnv()

# Test parameters
epoch = 1
episode_length = 0
total_reward = 0

# Assign random starting positions to agents for testing
env.agent_positions = [(random.randint(0, 9), random.randint(0, 9)) for _ in range(env.num_agents)]

# Main test loop to visualize render
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Randomly move agents for visual testing
    new_positions = []
    for pos in env.agent_positions:
        x, y = pos
        action = random.choice(['up', 'down', 'left', 'right', 'stay'])
        if action == 'up' and y > 0:
            y -= 1
        elif action == 'down' and y < env.map_size - 1:
            y += 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < env.map_size - 1:
            x += 1
        # Append new position to list
        new_positions.append((x, y))

    # Update agent positions
    env.agent_positions = new_positions

    # Render the environment
    env.render(epoch, episode_length, total_reward)

    # Increment test metrics
    episode_length += 1
    total_reward += 0.1  # Increment reward for demonstration

# Close Pygame window after loop
env.close()
