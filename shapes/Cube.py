__author__ = 'arolin'

from OpenGL.GL import *
from OpenGL.GLU import *

import logging
logger = logging.getLogger('shapes')


class GL_Shape():
    def __init__(self, scale=1, transform=None, name=''):
        self.scale = scale
        transform = transform
        self.name = name


class GL_Polygon(GL_Shape):
    def __init__(self, edges=None, vertecies=None, scale=1, transform=None, name=''):
        super().__init__(scale, transform, name)
        if edges:
            self.edges = edges
        else:
            self.edges = []
        if vertecies:
            self.vertecies = vertecies
        else:
            self.vertecies = []


    def render(self):
        logger.debug('render ' + str(type(self)) + ' ' +self.name)
        glBegin(GL_LINES)
        for edge in self.edges:
            for vertex in edge:
                glVertex3fv(self.vertecies[vertex])
        glEnd()


class Cube(GL_Polygon):
    vertecies = (( 1, -1, -1), # Right Bottom Back
                 ( 1,  1, -1), # Right Top    Back
                 (-1,  1, -1), # Left  Top    Back 
                 (-1, -1, -1), # Left  Bottom Back
                 ( 1, -1,  1), # Right Bottom Front
                 ( 1,  1,  1), # Right Top    Front
                 (-1, -1,  1), # Left  Bottom Front
                 (-1,  1,  1)) # Left  Top    Front
    edges = ((0,1), 
             (0,3), 
             (0,4),
             (2,1),
             (2,3),
             (2,7),
             (6,3),
             (6,4),
             (6,7),
             (5,1),
             (5,4),
             (5,7))

    def __init__(self, scale=1, transform=None, name=''):
        super(self.__class__, self).__init__( edges=self.edges, vertecies=self.vertecies, scale=1, transform=None, name=name)



if __name__ == '__main__':
    cube = Cube()




