import numpy as np

def manhattan_distance(pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_observation(self):
        # Observation includes each agent's position and nearest neighbors' positions
        observations = []
        for agent in self.agents:
            obs = list(agent.position)  # Agent's own position
            for neighbour in agent.neighbours:
                obs.extend(neighbour.position)  # Add each neighbor's position
            observations.append(obs[:self.observation_space.shape[0]])
        return np.array(observations)
    
def calculate_reward(self, agent):
    current_pos = agent.position
    target_pos = agent.target
    distance = manhattan_distance(current_pos, target_pos)
    reward = 0.0
    
    if distance == 0:
        reward += 1.0  # Reward for reaching the target
    else:
        reward -= 0.01 # Time step penalty for not reaching the target
        
    for neighbour in agent.neighbours:
        neighbour_pos = neighbour.position
        neighbour_distance = manhattan_distance(current_pos, neighbour_pos)
        if neighbour_distance == 0:
            reward -= 1  # Penalty for colliding with a neighbor
    
    return reward

def apply_action(self, agent, action):
    # 0 = up, 1 = down, 2 = left, 3 = right, 4 = stay
    if agent.flag == 1:
        return
    
    x, y = agent.position
    dx, dy = 0, 0

    if action == 1: # Move Up
        dy = -1
    elif action == 2: # Move Down
        dy = 1
    elif action == 3: # Move Left
        dx = -1
    elif action == 4: # Move Right
        dx = 1
        
    x += dx
    y += dy
    
    # Check if the new position is within the map boundaries
    if 0 <= x < self.map_size and 0 <= y < self.map_size:
        agent.position = (x, y)
        
    if manhattan_distance(agent.position, agent.target) == 0:
        agent.flag = 1
    