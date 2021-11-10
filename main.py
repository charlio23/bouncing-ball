"""This example spawns (bouncing) balls randomly on a L-shape constructed of 
two segment shapes. Not interactive.
"""

__version__ = "$Id:$"
__docformat__ = "reStructuredText"

# Python imports
import random
from typing import List

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util

from util import to_convex_contour
import numpy as np

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
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
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
        self._create_ball()
        for i in range(20*30):
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._process_events()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(30)
            ball_pos = [self._balls[0].body.position[0],self._balls[0].body.position[1]]
            positions.append(ball_pos)
            pygame.display.set_caption("FPS: " + str(self._clock.get_fps())+ " Frame: " + str(i))
        positions = np.array(positions)
        return positions
    def _add_static_scenery(self) -> None:
        """
        Create the static bodies.
        :return: None
        """
        N = random.randint(3,10)
        vertices = to_convex_contour(N)
        print(vertices)
        
        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body,
                         (point1[0]*500, point1[1]*500), 
                         (point2[0]*500, point2[1]*500),  0.0)
                         for point1, point2 in zip(vertices[:-1], vertices[1:])
        ]
        static_lines.append(pymunk.Segment(static_body,
                         (vertices[-1][0]*500, vertices[-1][1]*500), 
                         (vertices[0][0]*500, vertices[0][1]*500),  0.0))
        for line in static_lines:
            line.elasticity = 1
        self._space.add(*static_lines)

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
        radius = 25
        body = pymunk.Body()
        x = random.randint(115, 350)
        body.position = x, 200
        vx = random.uniform(-5,5)*100
        vy = random.randint(-5,5)*100
        body.velocity = vx, vy
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 1
        shape.density = 1
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
        self._space.debug_draw(self._draw_options)


if __name__ == "__main__":
    game = BouncingBall2D()
    positions = game.run()
    print(positions)