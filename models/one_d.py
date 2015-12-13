__author__ = 'arolin'

from numpy import random, linspace, zeros, arange
from numpy import sin, cos, pi
from scipy.integrate import odeint
from shapes.Cube import Cube

from OpenGL.GL import *
from OpenGL.GLU import *


class OneD(object):

    def __init__(self, parameters=None):
        self.cube = Cube()
        if not parameters:
            self.paramters = {'A' : 15000,
                              'B' : 10000,
                              'C' : 1,
                              'D' : -.01}
        else:
            self.paramters = parameters

    # def dy(self, y, x):
    #     return x*100

    def dy(self, y , t, coeff):
        A = coeff['A']
        B = coeff['B']
        C = coeff['C']
        D = coeff['D']
        return A*t**3 + B*t**2 + C*t + D

    def init_state(self, state = None):
        if not state:
            self.state = 2
        else:
            self.state = state


    def step_state(self, dt):
        x = [0, dt]
        y = odeint(self.dy, self.state, x, args=(self.paramters,))
        self.state = y[1]

    def render_state(self):
        glTranslatef(self.state,0.0,0.0)
        self.cube.render()




#
# y0 = 2.0
# t = linspace(-5.0, 3.0, 1000)
#
#
# y = odeint(dy, y0, t, args=(sys,))
#
#
#
#
# one_d = OneD()
# state = one_d.init_state()
#
