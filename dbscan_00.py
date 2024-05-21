import pygame
import sys
from sklearn.cluster import DBSCAN
import numpy as np

pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("DBSCAN")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)  
COLORS = [GREEN, YELLOW, RED] 

points = []
cluster_colors = []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            points.append(pygame.mouse.get_pos())
            cluster_colors.append(GRAY)  
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and points:
                # и наконец-то сам дбскан
                clustering = DBSCAN(eps=30, min_samples=5).fit(points)
                # максимальное расстояние между двумя точками в пикселях
                # min samples число точек чтобы считаться соседями
                labels = clustering.labels_

                cluster_colors = []
                for label in labels:
                    if label == -1:
                        cluster_colors.append(WHITE)
                    else:
                        cluster_colors.append(COLORS[label % len(COLORS)]) 
    screen.fill(BLACK)

    for idx, point in enumerate(points):
        pygame.draw.circle(screen, cluster_colors[idx], point, 5)

    pygame.display.flip()

pygame.quit()
sys.exit()
