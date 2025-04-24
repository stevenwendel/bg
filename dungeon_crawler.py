import pygame
import math
import random

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
PLAYER_SPEED = 3
ROTATION_SPEED = 0.05

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
BACKGROUND = (30, 30, 30)  # Dark gray background

# Map (1 = wall, 0 = empty space)
MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

class Player:
    def __init__(self):
        self.x = CELL_SIZE * 1.5
        self.y = CELL_SIZE * 1.5
        self.angle = 0

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check if new position is in a wall
        cell_x = int(new_x / CELL_SIZE)
        cell_y = int(new_y / CELL_SIZE)
        
        if 0 <= cell_x < len(MAP[0]) and 0 <= cell_y < len(MAP):
            if MAP[cell_y][cell_x] == 0:
                self.x = new_x
                self.y = new_y

class Skeleton:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 20
        self.color = WHITE  # Changed back to WHITE

    def draw(self, screen, player):
        # Convert 3D position to 2D screen position
        dx = self.x - player.x
        dy = self.y - player.y
        
        # Rotate point relative to player's angle
        rotated_x = dx * math.cos(-player.angle) - dy * math.sin(-player.angle)
        rotated_y = dx * math.sin(-player.angle) + dy * math.cos(-player.angle)
        
        if rotated_y > 0:  # Only draw if in front of player
            # Project onto screen
            screen_x = (rotated_x / rotated_y) * (SCREEN_WIDTH / 2) + (SCREEN_WIDTH / 2)
            screen_y = SCREEN_HEIGHT / 2
            
            # Scale size based on distance
            scale = 1 / rotated_y
            size = int(self.size * scale)
            
            if 0 <= screen_x < SCREEN_WIDTH:
                pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), size)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dungeon Crawler")
        self.clock = pygame.time.Clock()
        self.running = True
        self.player = Player()
        self.skeletons = [
            Skeleton(CELL_SIZE * 3.5, CELL_SIZE * 3.5),
            Skeleton(CELL_SIZE * 6.5, CELL_SIZE * 6.5)
        ]

    def cast_ray(self, angle):
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        
        # Vertical check
        x, dx = (self.player.x // CELL_SIZE) * CELL_SIZE, CELL_SIZE
        if cos_a > 0:
            x += CELL_SIZE
            dx = -CELL_SIZE
        
        depth_v = float('inf')
        for i in range(MAX_DEPTH):
            depth_v = (x - self.player.x) / cos_a
            y = self.player.y + depth_v * sin_a
            
            map_x = int(x / CELL_SIZE)
            map_y = int(y / CELL_SIZE)
            
            if 0 <= map_x < len(MAP[0]) and 0 <= map_y < len(MAP):
                if MAP[map_y][map_x] == 1:
                    break
            x += dx
        
        # Horizontal check
        y, dy = (self.player.y // CELL_SIZE) * CELL_SIZE, CELL_SIZE
        if sin_a > 0:
            y += CELL_SIZE
            dy = -CELL_SIZE
        
        depth_h = float('inf')
        if abs(sin_a) > 0.001:  # Prevent division by zero
            for i in range(MAX_DEPTH):
                depth_h = (y - self.player.y) / sin_a
                x = self.player.x + depth_h * cos_a
                
                map_x = int(x / CELL_SIZE)
                map_y = int(y / CELL_SIZE)
                
                if 0 <= map_x < len(MAP[0]) and 0 <= map_y < len(MAP):
                    if MAP[map_y][map_x] == 1:
                        if depth_h < depth_v:
                            return depth_h
                        break
                y += dy
        
        return min(depth_v, depth_h)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        keys = pygame.key.get_pressed()
        
        # Rotation
        if keys[pygame.K_LEFT]:
            self.player.angle -= ROTATION_SPEED
        if keys[pygame.K_RIGHT]:
            self.player.angle += ROTATION_SPEED
            
        # Movement
        sin_a = math.sin(self.player.angle)
        cos_a = math.cos(self.player.angle)
        
        if keys[pygame.K_w]:
            self.player.move(cos_a * PLAYER_SPEED, sin_a * PLAYER_SPEED)
        if keys[pygame.K_s]:
            self.player.move(-cos_a * PLAYER_SPEED, -sin_a * PLAYER_SPEED)
        if keys[pygame.K_a]:
            self.player.move(sin_a * PLAYER_SPEED, -cos_a * PLAYER_SPEED)
        if keys[pygame.K_d]:
            self.player.move(-sin_a * PLAYER_SPEED, cos_a * PLAYER_SPEED)

    def draw(self):
        self.screen.fill(BLACK)  # Changed back to BLACK
        
        # Draw 3D view
        for ray in range(NUM_RAYS):
            angle = self.player.angle - HALF_FOV + (FOV / NUM_RAYS) * ray
            depth = self.cast_ray(angle)
            
            # Fix fisheye effect
            depth *= math.cos(self.player.angle - angle)
            
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
        
        # Draw skeletons
        for skeleton in self.skeletons:
            skeleton.draw(self.screen, self.player)
        
        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit() 