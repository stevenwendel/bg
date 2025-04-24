import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
FOV = math.pi / 3  # 60 degrees
HALF_FOV = FOV / 2
NUM_RAYS = 120
MAX_DEPTH = 20
CELL_SIZE = 64

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)

# Simple map (1 = wall, 0 = empty space)
MAP = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

class Demo:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Raycasting Demo")
        self.clock = pygame.time.Clock()
        self.running = True
        self.player_x = CELL_SIZE * 1.5
        self.player_y = CELL_SIZE * 1.5
        self.player_angle = 0

    def cast_ray(self, angle):
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        
        # Vertical check
        x, dx = (self.player_x // CELL_SIZE) * CELL_SIZE, CELL_SIZE
        if cos_a > 0:
            x += CELL_SIZE
            dx = -CELL_SIZE
        
        depth_v = float('inf')
        for i in range(MAX_DEPTH):
            depth_v = (x - self.player_x) / cos_a
            y = self.player_y + depth_v * sin_a
            
            map_x = int(x / CELL_SIZE)
            map_y = int(y / CELL_SIZE)
            
            if 0 <= map_x < len(MAP[0]) and 0 <= map_y < len(MAP):
                if MAP[map_y][map_x] == 1:
                    break
            x += dx
        
        # Horizontal check
        y, dy = (self.player_y // CELL_SIZE) * CELL_SIZE, CELL_SIZE
        if sin_a > 0:
            y += CELL_SIZE
            dy = -CELL_SIZE
        
        depth_h = float('inf')
        if abs(sin_a) > 0.001:  # Prevent division by zero
            for i in range(MAX_DEPTH):
                depth_h = (y - self.player_y) / sin_a
                x = self.player_x + depth_h * cos_a
                
                map_x = int(x / CELL_SIZE)
                map_y = int(y / CELL_SIZE)
                
                if 0 <= map_x < len(MAP[0]) and 0 <= map_y < len(MAP):
                    if MAP[map_y][map_x] == 1:
                        if depth_h < depth_v:
                            return depth_h
                        break
                y += dy
        
        return min(depth_v, depth_h)

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw 3D view
        for ray in range(NUM_RAYS):
            angle = self.player_angle - HALF_FOV + (FOV / NUM_RAYS) * ray
            depth = self.cast_ray(angle)
            
            # Fix fisheye effect
            depth *= math.cos(self.player_angle - angle)
            
            # Calculate wall height
            wall_height = min(int(CELL_SIZE * SCREEN_HEIGHT / depth), SCREEN_HEIGHT)
            
            # Draw wall slice
            wall_color = max(0, min(255, int(255 * (1 - depth / MAX_DEPTH))))
            color = (wall_color, wall_color, wall_color)
            
            pygame.draw.rect(self.screen, color, (
                ray * (SCREEN_WIDTH // NUM_RAYS),
                (SCREEN_HEIGHT - wall_height) // 2,
                SCREEN_WIDTH // NUM_RAYS + 1,
                wall_height
            ))

        # Draw top-down view
        top_down_size = 200
        top_down_x = SCREEN_WIDTH - top_down_size - 20
        top_down_y = 20
        
        # Draw map
        for y in range(len(MAP)):
            for x in range(len(MAP[0])):
                if MAP[y][x] == 1:
                    pygame.draw.rect(self.screen, WHITE, (
                        top_down_x + x * (top_down_size // len(MAP[0])),
                        top_down_y + y * (top_down_size // len(MAP)),
                        top_down_size // len(MAP[0]),
                        top_down_size // len(MAP)
                    ))
        
        # Draw player
        pygame.draw.circle(self.screen, (255, 0, 0), (
            int(top_down_x + (self.player_x / CELL_SIZE) * (top_down_size // len(MAP[0]))),
            int(top_down_y + (self.player_y / CELL_SIZE) * (top_down_size // len(MAP)))
        ), 5)
        
        # Draw FOV lines
        for angle in [self.player_angle - HALF_FOV, self.player_angle + HALF_FOV]:
            end_x = self.player_x + math.cos(angle) * CELL_SIZE * 2
            end_y = self.player_y + math.sin(angle) * CELL_SIZE * 2
            pygame.draw.line(self.screen, (0, 255, 0), (
                top_down_x + (self.player_x / CELL_SIZE) * (top_down_size // len(MAP[0])),
                top_down_y + (self.player_y / CELL_SIZE) * (top_down_size // len(MAP))
            ), (
                top_down_x + (end_x / CELL_SIZE) * (top_down_size // len(MAP[0])),
                top_down_y + (end_y / CELL_SIZE) * (top_down_size // len(MAP))
            ), 1)

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player_angle -= 0.1
                elif event.key == pygame.K_RIGHT:
                    self.player_angle += 0.1
                elif event.key == pygame.K_w:
                    self.player_x += math.cos(self.player_angle) * 10
                    self.player_y += math.sin(self.player_angle) * 10
                elif event.key == pygame.K_s:
                    self.player_x -= math.cos(self.player_angle) * 10
                    self.player_y -= math.sin(self.player_angle) * 10

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.draw()

if __name__ == "__main__":
    demo = Demo()
    demo.run()
    pygame.quit()
    sys.exit() 