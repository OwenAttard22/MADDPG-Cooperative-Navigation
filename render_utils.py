import pygame

def render_screen(screen, font, map_size, cell_size, agents, agent_colors, endpoints, epoch, episode_length, total_reward, window_size, agent_rewards):
    screen.fill((255, 255, 255))  # White background

    # Draw the 10x10 board with a chessboard pattern
    for x in range(map_size):
        for y in range(map_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            color = (200, 200, 200) if (x + y) % 2 == 0 else (100, 100, 100)  # Light and dark squares
            pygame.draw.rect(screen, color, rect)

    # Display endpoints with labels
    for name, pos in endpoints.items():
        x, y = pos
        text = font.render(name, True, (0, 0, 0))
        screen.blit(text, (x * cell_size + 10, y * cell_size + 10))
        pygame.draw.circle(screen, (0, 0, 0), (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), 10)

    # Draw agents and their neighbors' connections
    for i, agent in enumerate(agents):
        agent_x, agent_y = agent.position
        color = agent_colors[agent.id]

        # Draw lines to neighbors
        for neighbour in agent.neighbours:
            neighbour_x, neighbour_y = neighbour.position
            start_pos = (agent_x * cell_size + cell_size // 2, agent_y * cell_size + cell_size // 2)
            end_pos = (neighbour_x * cell_size + cell_size // 2, neighbour_y * cell_size + cell_size // 2)
            pygame.draw.line(screen, color, start_pos, end_pos, 2)  # Draw line to neighbor

        # Draw the agent
        pygame.draw.circle(screen, color, (agent_x * cell_size + cell_size // 2, agent_y * cell_size + cell_size // 2), 15)

        # Display agent's reward near the agent
        reward_text = font.render(f"R: {agent_rewards[i]:.2f}", True, (0, 0, 0))
        screen.blit(reward_text, (agent_x * cell_size, agent_y * cell_size + cell_size // 2 + 10))

    # Display global metrics
    metrics_text = f"Epoch: {epoch} | Episode Length: {episode_length} | Total Reward: {total_reward}"
    metrics_surface = font.render(metrics_text, True, (0, 0, 0))
    screen.blit(metrics_surface, (10, window_size + 20))

    # Update the display
    pygame.display.flip()
