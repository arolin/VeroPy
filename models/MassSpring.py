__author__ = 'arolin'

from sympy import *

__author__ = 'arolin'

from numpy import random, linspace, zeros, arange
from numpy import sin, cos, pi
from scipy.integrate import odeint
from shapes.Cube import Cube

from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, inertia, RigidBody


from sympy.physics.vector import init_vprinting
init_vprinting(use_latex='mathjax')


from OpenGL.GL import *
from OpenGL.GLU import *


class MassSpring(object):


    def __init__(self, parameters=None):
        """
        define the local reference coordinates and geometry


        :param parameters: m - mass, k - spring constant
        :return:
        """
        self.bodies = [Cube()]
        self.A = ReferenceFrame('A')
        self.cog = Point('cog')

        if not parameters:
            self.parameters = {'g': 9.8,
                               'm': 1,
                               'K': 1,
                               'I': 1}
        else:
            self.parameters = parameters

        self.define_model()

    def define_model(self):
        self.gravity, self.mass, self.spring_const, self.Ix, self.Iy, self.Iz,  self.time = symbols('g, m, K, Ix,Iy,Iz, t')

        self.inertia_dyaic = inertia(self.A, self.Ix, self.Iy, self.Iz)

        self.spring_force = self.A.x* self.spring_const

        I = ReferenceFrame('I')                # Inertial reference frame
        O = Point('O')                         # Origin point
        O.set_vel(I, 0)                        # Origin's velocity is zero




    def dy(self, y , t, coeff):
        m = coeff['m']
        K = coeff['K']

        return y[0]*-K*t


    def init_state(self, state = None):
        if not state:
            self.state = 2
        else:
            self.state = state

    def render_state(self):
        glTranslatef(self.state,0.0,0.0)
        for body in self.bodies:
            body.render()

    def apply_force(self, force):
        self.forces.appnd(force)

    def step_state(self, dt):
        """
        perfoms a single time step of the diferentila equation solver
        :param dt:
        :return:
        """
        x = [0, dt]

        y = odeint(self.dy, self.state, x, args=(self.parameters,))
        self.state = y[1]

massSpring = MassSpring()
massSpring.define_model()