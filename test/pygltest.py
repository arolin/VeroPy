__author__ = 'arolin'
import pygame
from pygame.locals import *
from pydy.viz.scene import Scene
from OpenGL.GL import *
from OpenGL.GLU import *


import logging

from shapes.Cube import  *

from time import sleep

logger = logging.getLogger('pydygl')
logging.basicConfig(level='INFO')

# We'll start by generating the equations of motion for the system
# with SymPy mechanics. The functionality that mechanics provides is
# much more in depth than Mathematica's functionality. In the
# Mathematica example, Lagrangian mechanics were implemented manually
# with Mathematica's symbolic functionality. mechanics provides an
# assortment of functions and classes to derive the equations of
# motion for arbitrarily complex (i.e. configuration constraints,
# nonholonomic motion constraints, etc) multibody systems in a very
# natural way. First we import the necessary functionality from SymPy.
from sympy import symbols
from sympy.physics.mechanics import *

n = 1

# mechanics will need the generalized coordinates, generalized speeds,
# and the input force which are all time dependent variables and the
# bob masses, link lengths, and acceleration due to gravity which are
# all constants. Time, t, is also made available because we will need
# to differentiate with respect to time.

q = dynamicsymbols('q:' + str(n + 1))  # Generalized coordinates
u = dynamicsymbols('u:' + str(n + 1))  # Generalized speeds
f = dynamicsymbols('f')                # Force applied to the cart

m = symbols('m:' + str(n + 1))         # Mass of each bob
l = symbols('l:' + str(n))             # Length of each link
g, t = symbols('g t')                  # Gravity and time

#Now we can create and inertial reference frame I and define the point, O, as the origin.
I = ReferenceFrame('I')                # Inertial reference frame
O = Point('O')                         # Origin point
O.set_vel(I, 0)                        # Origin's velocity is zero

#Secondly, we define the define the first point of the pendulum as a
#particle which has mass. This point can only move laterally and
#represents the motion of the "cart".
P0 = Point('P0')                       # Hinge point of top link
P0.set_pos(O, q[0] * I.x)              # Set the position of P0    
P0.set_vel(I, u[0] * I.x)              # Set the velocity of P0
Pa0 = Particle('Pa0', P0, m[0])        # Define a particle at P0

###Now we can define the n reference frames, particles, gravitational
###forces, and kinematical differential equations for each of the
###pendulum links. This is easily done with a loop.
frames = [I]                              # List to hold the n + 1 frames
points = [P0]                             # List to hold the n + 1 points
particles = [Pa0]                         # List to hold the n + 1 particles
forces = [(P0, f * I.x - m[0] * g * I.y)] # List to hold the n + 1 applied forces, including the input force, f
kindiffs = [q[0].diff(t) - u[0]]          # List to hold kinematic ODE's



for i in range(n):
    Bi = I.orientnew('B' + str(i), 'Axis', [q[i + 1], I.z])   # Create a new frame
    Bi.set_ang_vel(I, u[i + 1] * I.z)                         # Set angular velocity
    frames.append(Bi)                                         # Add it to the frames list

    Pi = points[-1].locatenew('P' + str(i + 1), l[i] * Bi.x)  # Create a new point
    Pi.v2pt_theory(points[-1], I, Bi)                         # Set the velocity
    points.append(Pi)                                         # Add it to the points list
    
    Pai = Particle('Pa' + str(i + 1), Pi, m[i + 1])           # Create a new particle
    particles.append(Pai)                                     # Add it to the particles list

    forces.append((Pi, -m[i + 1] * g * I.y))                  # Set the force applied at the point
    
    kindiffs.append(q[i + 1].diff(t) - u[i + 1])              # Define the kinematic ODE:  dq_i / dt - u_i = 0

# With all of the necessary point velocities and particle masses
# defined, the KanesMethod class can be used to derive the equations
# of motion of the system automatically
kane = KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs) # Initialize the object
fr, frstar = kane.kanes_equations(forces, particles)     # Generate EoM's fr + frstar = 0


# The equations of motion are quite long as can been seen below. This
# is the general nature of most non-simple mutlibody problems. That is
# why a SymPy is so useful; no more mistakes in algegra,
# differentiation, or copying in hand written equations.
fr

# Now that the symbolic equations of motion are available we can
# simulate the pendulum's motion. We will need some more SymPy
# functionality and several NumPy functions, and most importantly the
# integration function from SciPy, odeint.


from sympy import Dummy, lambdify
from numpy import array, hstack, zeros, ones, linspace, pi
from numpy.linalg import solve
from scipy.integrate import odeint


# First, define some numeric values for all of the constant parameters
# in the problem.

arm_length = 10. / n                          # The maximum length of the pendulum is 1 meter
bob_mass = 0.01 / n                          # The maximum mass of the bobs is 10 grams
parameters = [g, m[0]]                       # Parameter definitions starting with gravity and the first bob
parameter_vals = [9.81, 0.01 / n]            # Numerical values for the first two
for i in range(n):                           # Then each mass and length
    parameters += [l[i], m[i + 1]]            
    parameter_vals += [arm_length, bob_mass]

# Mathematica has a really nice NDSolve function for quickly
# integrating their symbolic differential equations. We have plans to
# develop something similar for SymPy but haven't found the
# development time yet to do it properly. So the next bit isn't as
# clean as we'd like but you can make use of SymPy's lambdify function
# to create functions that will evaluate the mass matrix, M, and

# make use of dummy symbols to replace the time varying functions in
# the SymPy equations a simple dummy symbol.
dynamic = q + u                                                # Make a list of the states
dynamic.append(f)                                              # Add the input force
dummy_symbols = [Dummy() for i in dynamic]                     # Create a dummy symbol for each variable
dummy_dict = dict(zip(dynamic, dummy_symbols))                 
kindiff_dict = kane.kindiffdict()                              # Get the solved kinematical differential equations
M = kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)  # Substitute into the mass matrix 
F = kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)      # Substitute into the forcing vector
M_func = lambdify(dummy_symbols + parameters, M)               # Create a callable function to evaluate the mass matrix 
F_func = lambdify(dummy_symbols + parameters, F)               # Create a callable function to evaluate the forcing vector



# To integrate the ODE's we need to define a function that returns the
# derivatives of the states given the current state and time.
def right_hand_side(x, t, args):
    """Returns the derivatives of the states.

    Parameters
    ----------
    x : ndarray, shape(2 * (n + 1))
        The current state vector.
    t : float
        The current time.
    args : ndarray
        The constants.

    Returns
    -------
    dx : ndarray, shape(2 * (n + 1))
        The derivative of the state.
    
    """
    u = 0.0                              # The input force is always zero     
    arguments = hstack((x, u, args))     # States, input, and parameters
    dx = array(solve(M_func(*arguments), # Solving for the derivatives
                     F_func(*arguments))).T[0]
    
    return dx


x0 = hstack(( 0, ones(len(q) - 1) , 1e-3 * ones(len(u)) )) # Initial conditions, q and u
t = [0, .01]                                        # Time vector

state = x0

def step_model(state):
    y = odeint(right_hand_side, state, t, args=(parameter_vals,))         # Actual integration
    return y[1]

class Scene():
    pass

scene = Scene()
scene.cube = Cube()

def render_scene(scene):
    for shape in scene['shapes']:
        shape.render()


def animate():
    screen = pygame.display.set_mode((600,400), 0, 32)


def render_state(state):
    logger.debug(state[0])
    glTranslatef(state[0],0.0, 1)
    scene.cube.render()

    glRotatef(state[1]*53, 0, 0, 1)
    glTranslate(parameter_vals[2],0,0,1)
    scene.cube.render()

    # glRotatef(state[2]*53, 0, 0, 1)
    # glTranslate(0,parameter_vals[4],0,1)
    # scene.cube.render()


class Vis():

    def init(self):
        self.run = True
        pygame.init()
        self.display = (800,600)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF|OPENGL)



    def process_event_queue(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True


    def animate(self):
        global state
        while self.run:
            if self.process_event_queue():
                return None
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            gluPerspective(45, (self.display[0]/self.display[1]), 0.1, 50.0)

            glTranslatef(0.0,0.0, -25)

            state = step_model(state)

            render_state(state)

            pygame.display.flip()

            pygame.time.wait(10)



vis = Vis()
vis.init()
vis.animate()
    
def update():
    glRotatef(1, 3, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    Cube()
    render_scene(scene)
    pygame.display.flip()




def run_event_loop():
    while process_event_queue():
        sleep(.1)
        print('a')



def start():
    init()
    update()
    run_event_loop()

#quit()

#imp.reload(pygltest); from pygltest import *
