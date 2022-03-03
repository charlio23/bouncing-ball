# Python imports
import random
from typing import List

# Library imports
import pygame
import cv2

# pymunk imports
import pymunk
import pymunk.pygame_util

from datasets.util import to_convex_contour, calculate_midpoint
import numpy as np

from pymunk import Vec2d
import os
import matplotlib.pyplot as plt

os.environ["SDL_VIDEODRIVER"] = "dummy"

class BouncingBall2D(object):
    """
    This class implements a simple scene in which there is a static closure (made up of a convex polygon). 
    A ball bounces wihthin the structure.
    """

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()

        # Physics
        # Time step
        self._dt = 1.0 / 30.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((256, 256))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self.static_lines = None
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls: List[pymunk.Circle] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10


    def run(self):
        """
        The main loop of the game.
        """
        # Main loop
        positions = []
        image_seq = []
        self._create_ball()
        for i in range(100):
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)
            if self._balls[0].body.position[0] < 0 or self._balls[0].body.position[0] > 256:
                return None, None
            self._process_events()
            self._clear_screen()
            self._draw_objects()
            # Delay fixed time between frames
            #self._clock.tick(30)
            ball_pos = [self._balls[0].body.position[0],self._balls[0].body.position[1], self._balls[0].body.velocity[0], self._balls[0].body.velocity[1]]
            positions.append(ball_pos)
            string_image = pygame.image.tostring(self._screen, 'RGB')
            temp_surf = pygame.image.fromstring(string_image,(256, 256),'RGB' )
            tmp_arr = pygame.surfarray.array3d(temp_surf)
            image = cv2.resize(tmp_arr, dsize=(64, 64))
            image_seq.append(image)
        positions = np.array(positions)
        image_seq = np.array(image_seq)
        return positions, image_seq

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        N = random.randint(4,10)
        vertices = to_convex_contour(N)
        midpoint = calculate_midpoint(vertices)
        vertices = [[p[0] - midpoint[0] + 0.5, p[1] - midpoint[1] + 0.5] for p in vertices]
        #print(vertices)
        #print(midpoint)
        line_width = 2
        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body,
                         (point1[0]*256, point1[1]*256), 
                         (point2[0]*256, point2[1]*256),  line_width)
                         for point1, point2 in zip(vertices[:-1], vertices[1:])
        ]
        static_lines.append(pymunk.Segment(static_body,
                         (vertices[-1][0]*256, vertices[-1][1]*256), 
                         (vertices[0][0]*256, vertices[0][1]*256),  line_width))
        for line in static_lines:
            line.elasticity = 1
            line.color = (100, 100, 100, 255)
        self._space.add(*static_lines)
        self.static_lines = static_lines

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")


    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        radius = 15
        body = pymunk.Body()
        x = random.randint(100, 200)
        body.position = x, 128
        vx = random.uniform(-5,5)*50
        vy = random.randint(-5,5)*50
        body.velocity = vx, vy
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 1
        shape.density = 1
        shape.color = (100, 100, 255, 255)
        self._space.add(body, shape)
        self._balls.append(shape)

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self.draw()

    def draw(self):
            self._screen.fill(pygame.Color(150, 150, 150))
            pygame.draw.circle(self._screen, pygame.Color(100,200,100),self._balls[0].body.position, 15)
            # Draw the static lines.
            for line in self.static_lines:
                pygame.draw.lines(self._screen, pygame.Color('black'), False, (line.a,line.b), 8)



if __name__ == "__main__":
    game = BouncingBall2D()
    positions, images = game.run()
    if positions is None:
        print("Failed")
    else:
        print(positions.shape)
        print(images.shape)
        np.save('positions.npy', positions)
        plt.imshow(images[0])
        plt.show()
