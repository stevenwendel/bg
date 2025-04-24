import pygame
import random
import math
import json
import os

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Game states
MENU = 0
PLAYING = 1
GAME_OVER = 2

# High score file
HIGH_SCORE_FILE = "high_score.json"

class Explosion(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.images = []
        self.frame = 0
        self.frame_count = 8  # Number of frames in explosion animation
        
        # Create explosion animation frames
        for i in range(self.frame_count):
            size = 30 + i * 5  # Explosion grows over time
            image = pygame.Surface((size, size), pygame.SRCALPHA)
            # Create a circular explosion
            pygame.draw.circle(image, YELLOW, (size//2, size//2), size//2)
            pygame.draw.circle(image, ORANGE, (size//2, size//2), size//2 - 5)
            pygame.draw.circle(image, RED, (size//2, size//2), size//2 - 10)
            self.images.append(image)
        
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = 50  # milliseconds between frames

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frame_rate:
            self.last_update = now
            self.frame += 1
            if self.frame == self.frame_count:
                self.kill()
            else:
                self.image = self.images[self.frame]
                self.rect = self.image.get_rect(center=self.rect.center)

# Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 30))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT - 10
        self.speed = 5
        self.lives = 3

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed

# Enemy class
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, SCREEN_WIDTH - self.rect.width)
        self.rect.y = random.randint(-100, -40)
        self.speed = random.randint(1, 3)
        self.health = 1

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > SCREEN_HEIGHT:
            self.rect.x = random.randint(0, SCREEN_WIDTH - self.rect.width)
            self.rect.y = random.randint(-100, -40)

    def explode(self, game):
        explosion = Explosion(self.rect.centerx, self.rect.centery)
        game.all_sprites.add(explosion)

# Bullet class
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.bottom = y
        self.speed = 7

    def update(self):
        self.rect.y -= self.speed
        if self.rect.bottom < 0:
            self.kill()

# Game class
class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Galaga Clone")
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = MENU
        self.score = 0
        self.high_score = self.load_high_score()
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 72)
        self.auto_play = False
        self.last_shot = 0
        self.shot_delay = 200  # milliseconds between auto shots

        # Sprite groups
        self.all_sprites = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()

        # Create player
        self.player = Player()
        self.all_sprites.add(self.player)

        # Create enemies
        for i in range(10):
            enemy = Enemy()
            self.all_sprites.add(enemy)
            self.enemies.add(enemy)

    def load_high_score(self):
        try:
            if os.path.exists(HIGH_SCORE_FILE):
                with open(HIGH_SCORE_FILE, 'r') as f:
                    return json.load(f)
            return 0
        except:
            return 0

    def save_high_score(self):
        try:
            with open(HIGH_SCORE_FILE, 'w') as f:
                json.dump(self.high_score, f)
        except:
            pass

    def reset_game(self):
        self.score = 0
        self.player.lives = 3
        self.all_sprites.empty()
        self.enemies.empty()
        self.bullets.empty()
        
        # Recreate player
        self.player = Player()
        self.all_sprites.add(self.player)
        
        # Recreate enemies
        for i in range(10):
            enemy = Enemy()
            self.all_sprites.add(enemy)
            self.enemies.add(enemy)

    def draw_menu(self):
        self.screen.fill(BLACK)
        
        # Draw title
        title = self.title_font.render("GALAGA CLONE", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//3))
        self.screen.blit(title, title_rect)
        
        # Draw instructions
        start_text = self.font.render("Press SPACE to Start", True, WHITE)
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        self.screen.blit(start_text, start_rect)
        
        # Draw high score
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, GREEN)
        high_score_rect = high_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
        self.screen.blit(high_score_text, high_score_rect)
        
        pygame.display.flip()

    def draw_game_over(self):
        self.screen.fill(BLACK)
        
        # Draw game over text
        game_over = self.title_font.render("GAME OVER", True, RED)
        game_over_rect = game_over.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//3))
        self.screen.blit(game_over, game_over_rect)
        
        # Draw final score
        score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        self.screen.blit(score_text, score_rect)
        
        # Draw high score
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, GREEN)
        high_score_rect = high_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
        self.screen.blit(high_score_text, high_score_rect)
        
        # Draw restart instructions
        restart_text = self.font.render("Press SPACE to Play Again", True, WHITE)
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 100))
        self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()

    def auto_play_update(self):
        if not self.auto_play:
            return

        # Find closest enemy
        closest_enemy = None
        min_distance = float('inf')
        for enemy in self.enemies:
            distance = abs(enemy.rect.centerx - self.player.rect.centerx)
            if distance < min_distance:
                min_distance = distance
                closest_enemy = enemy

        # Move towards closest enemy
        if closest_enemy:
            if closest_enemy.rect.centerx < self.player.rect.centerx:
                self.player.rect.x -= self.player.speed
            elif closest_enemy.rect.centerx > self.player.rect.centerx:
                self.player.rect.x += self.player.speed

        # Shoot at regular intervals
        now = pygame.time.get_ticks()
        if now - self.last_shot > self.shot_delay:
            self.last_shot = now
            bullet = Bullet(self.player.rect.centerx, self.player.rect.top)
            self.all_sprites.add(bullet)
            self.bullets.add(bullet)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.state == MENU:
                        self.state = PLAYING
                    elif self.state == GAME_OVER:
                        self.state = MENU
                        self.reset_game()
                elif event.key == pygame.K_ESCAPE:
                    if self.state == PLAYING:
                        self.state = MENU
                        self.reset_game()
                elif event.key == pygame.K_a:  # Toggle auto-play
                    if self.state == PLAYING:
                        self.auto_play = not self.auto_play

        if self.state == PLAYING and not self.auto_play:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                bullet = Bullet(self.player.rect.centerx, self.player.rect.top)
                self.all_sprites.add(bullet)
                self.bullets.add(bullet)

    def update(self):
        if self.state != PLAYING:
            return

        self.auto_play_update()
        self.all_sprites.update()

        # Check for collisions between bullets and enemies
        hits = pygame.sprite.groupcollide(self.enemies, self.bullets, False, True)
        for enemy, bullets in hits.items():
            enemy.health -= len(bullets)
            if enemy.health <= 0:
                enemy.explode(self)
                enemy.kill()
                self.score += 10
                if self.score > self.high_score:
                    self.high_score = self.score
                    self.save_high_score()
                new_enemy = Enemy()
                self.all_sprites.add(new_enemy)
                self.enemies.add(new_enemy)

        # Check for collisions between player and enemies
        hits = pygame.sprite.spritecollide(self.player, self.enemies, False)
        if hits:
            self.player.lives -= 1
            if self.player.lives <= 0:
                self.state = GAME_OVER

    def draw(self):
        if self.state == MENU:
            self.draw_menu()
        elif self.state == GAME_OVER:
            self.draw_game_over()
        else:  # PLAYING state
            self.screen.fill(BLACK)
            self.all_sprites.draw(self.screen)
            
            # Draw score and lives
            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            lives_text = self.font.render(f"Lives: {self.player.lives}", True, WHITE)
            auto_text = self.font.render("Auto-play: ON" if self.auto_play else "Auto-play: OFF", True, GREEN)
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(lives_text, (SCREEN_WIDTH - 150, 10))
            self.screen.blit(auto_text, (SCREEN_WIDTH//2 - 50, 10))
            
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