import pygame

def render_screen(screen, font, map_size, cell_size, agent_positions, agent_colors, endpoints, epoch, episode_length, total_reward, window_size):
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

    # Draw agents in different colors
    for i, (agent_pos, color) in enumerate(zip(agent_positions, agent_colors)):
        x, y = agent_pos
        pygame.draw.circle(screen, color, (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2), 15)

    # Display metrics
    metrics_text = f"Epoch: {epoch} | Episode Length: {episode_length} | Total Reward: {total_reward}"
    metrics_surface = font.render(metrics_text, True, (0, 0, 0))
    screen.blit(metrics_surface, (10, window_size + 20))

    # Update the display
    pygame.display.flip()
