__author__ = 'arolin'

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from models.one_d import OneD
from models.MassSpring import MassSpring



class Viz():
    """
    This clas runs the window and the model for display
    """
    def __init__(self):
        """
        Setup the visualization window paramaters
        :return: None
        """
        self.display_size = (800,600)

    def init(self, model):
        """
        Open a pygame window
        :return:
        """
        pygame.init()
        self.screen = pygame.display.set_mode(self.display_size, DOUBLEBUF|OPENGL)

        self.model = model
        self.model.init_state()


    def process_event_queue(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True


    def animate(self):

        while 1:
            if self.process_event_queue():
                return None
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            gluPerspective(45, (self.display_size[0]/self.display_size[1]), 0.1, 50.0)

            glTranslatef(0.0,0.0, -25)

            self.model.step_state(.01)
            self.model.render_state()


            pygame.display.flip()

            pygame.time.wait(10)

if __name__ == '__main__':
    model = OneD()
    model = MassSpring()
    viz = Viz()
    viz.init(model)
    viz.animate()
