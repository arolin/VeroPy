__author__ = 'arolin'

from models.Model import Model

import control

from numpy import matrix
from numpy import dot, rank
from numpy.linalg import matrix_rank

from OpenGL.GL import *
from OpenGL.GLU import *


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

from sympy import Dummy, lambdify
from numpy import array, hstack, zeros, ones, linspace, pi
from numpy.linalg import solve
from scipy.integrate import odeint

from shapes.Cube import *

cube = Cube()

class Pendulum(Model):
    def __init__(self):
        self.control = True
        self.n = 1
        # First, define some numeric values for all of the constant parameters
        # in the problem.

        #Now we can create and inertial reference frame I and define the point, O, as the origin.
        self.I = ReferenceFrame('I')                # Inertial reference frame
        self.O = Point('O')                         # Origin point
        self.O.set_vel(self.I, 0)                        # Origin's velocity is zero

        self.m = symbols('m:' + str(self.n + 1))         # Mass of each bob
        self.l = symbols('l:' + str(self.n))             # Length of each link
        self.g, self.t = symbols('g t')                  # Gravity and time


        self.arm_length = 3. / self.n                          # The maximum length of the pendulum is 1 meter
        self.bob_mass = 0.1 / self.n                          # The maximum mass of the bobs is 10 grams
        self.parameters = [self.g, self.m[0]]                       # Parameter definitions starting with gravity and the first bob
        self.parameter_vals = [9.81, 0.01 / self.n]
        for i in range(self.n):                           # Then each mass and length
            self.parameters += [self.l[i], self.m[i + 1]]
            self.parameter_vals += [self.arm_length, self.bob_mass]



        self.q = dynamicsymbols('q:' + str(self.n + 1))  # Generalized coordinates
        self.u = dynamicsymbols('u:' + str(self.n + 1))  # Generalized speeds
        self.f = dynamicsymbols('f')                # Force applied to the cart

        self.derive_model()

    def derive_model(self):
        # Secondly, we define the define the first point of the pendulum as a
        # particle which has mass. This point can only move laterally and
        # represents the motion of the "cart".
        self.P0 = Point('P0')                       # Hinge point of top link
        self.P0.set_pos(self.O, self.q[0] * self.I.x)              # Set the position of P0
        self.P0.set_vel(self.I, self.u[0] * self.I.x)              # Set the velocity of P0
        self.Pa0 = Particle('Pa0', self.P0, self.m[0])        # Define a particle at P0

        # Now we can define the n reference frames, particles, gravitational
        # forces, and kinematical differential equations for each of the
        # pendulum links. This is easily done with a loop.
        self.frames = [self.I]                              # List to hold the n + 1 frames
        self.points = [self.P0]                             # List to hold the n + 1 points
        self.particles = [self.Pa0]                         # List to hold the n + 1 particles
        self.forces = [(self.P0, self.f * self.I.x - self.m[0] * self.g * self.I.y)] # List to hold the n + 1 applied forces, including the input force, f
        self.kindiffs = [self.q[0].diff(self.t) - self.u[0]]          # List to hold kinematic ODE's


        for i in range(self.n):
            Bi = self.I.orientnew('B' + str(i), 'Axis', [self.q[i + 1], self.I.z])   # Create a new frame
            Bi.set_ang_vel(self.I, self.u[i + 1] * self.I.z)                         # Set angular velocity
            self.frames.append(Bi)                                                   # Add it to the frames list

            Pi = self.points[-1].locatenew('P' + str(i + 1), self.l[i] * Bi.x)  # Create a new point
            Pi.v2pt_theory(self.points[-1], self.I, Bi)                         # Set the velocity
            self.points.append(Pi)                                              # Add it to the points list

            Pai = Particle('Pa' + str(i + 1), Pi, self.m[i + 1])           # Create a new particle
            self.particles.append(Pai)                                     # Add it to the particles list

            self.forces.append((Pi, -self.m[i + 1] * self.g * self.I.y))                  # Set the force applied at the point

            self.kindiffs.append(self.q[i + 1].diff(self.t) - self.u[i + 1])              # Define the kinematic ODE:  dq_i / dt - u_i = 0

            # With all of the necessary point velocities and particle masses
            # defined, the KanesMethod class can be used to derive the equations
            # of motion of the system automatically
            self.kane = KanesMethod(self.I, q_ind=self.q, u_ind=self.u, kd_eqs=self.kindiffs) # Initialize the object
            self.fr, self.frstar = self.kane.kanes_equations(self.forces, self.particles)     # Generate EoM's fr + frstar = 0

            # Now that the symbolic equations of motion are available we can
            # simulate the pendulum's motion.

            # make use of dummy symbols to replace the time varying functions in
            # the SymPy equations a simple dummy symbol.
            dynamic = self.q + self.u   # Make a list of the states
            dynamic.append(self.f)      # Add the input force
            dummy_symbols = [Dummy() for i in dynamic]   # Create a dummy symbol for each variable
            dummy_dict = dict(zip(dynamic, dummy_symbols))
            kindiff_dict = self.kane.kindiffdict()  # Get the solved kinematical differential equations
            self.M = self.kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)  # Substitute into the mass matrix
            self.F = self.kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)      # Substitute into the forcing vector
            self.M_func = lambdify(dummy_symbols + self.parameters, self.M) # Create a callable function to evaluate the mass matrix
            self.F_func = lambdify(dummy_symbols + self.parameters, self.F) # Create a callable function to evaluate the forcing vector



            # To integrate the ODE's we need to define a function that returns the
            # derivatives of the states given the current state and time.
    def right_hand_side(self, x, t, args):
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
        if self.control:
            # self.control_eq()
            # u = dot(self.K, self.equilibrium_point - x)    # The controller
            if -pi/2 <= x[1] <= 0:
                if x[3] < 0:
                    u = -x[3]*.01
                if x[3] > 0:
                    u = -x[3]*.01
            else:
                u=0
        else:
            u = -x[2] *10
        arguments = hstack((x, u, args))     # States, input, and parameters
        dx = array(solve(self.M_func(*arguments), # Solving for the derivatives
                         self.F_func(*arguments))).T[0]

        return dx

    def init_state(self, state=None):
        if not state:
            self.state = hstack(( 0, ones(len(self.q) - 1) +.5, 1e-3 * ones(len(self.u)) )) # Initial conditions, q and u
            self.state = hstack(( 0, pi/2 *ones(len(self.q) - 1) , 1e-3 * zeros(len(self.u)) )) # Initial conditions, q and u
            self.state = hstack(( 0, -pi/2 *ones(len(self.q) - 1) -.1, [0,0] )) # Initial conditions, q and u
        else:
            self.state = state

    def step_state(self, dt):
        t = [0, dt]                                        # Time vector
        y = odeint(self.right_hand_side, self.state, t, args=(self.parameter_vals,))         # Actual integration
        self.state = y[1]

    def render_state(self):
        glTranslatef(self.state[0],0.0, 1)
        cube.render()

        glRotatef(self.state[1]*180/pi, 0, 0, 1)
        glTranslate(self.parameter_vals[2],0,0,1)
        cube.render()

    def control_eq(self):
        self.equilibrium_point = hstack(( 0, pi / 2 * ones(len(self.q) - 1), zeros(len(self.u)) ))
        equilibrium_dict = dict(zip(self.q + self.u, self.equilibrium_point))
        parameter_dict = dict(zip(self.parameters, self.parameter_vals))

        # symbolically linearize about arbitrary equilibrium
        self.linear_state_matrix, self.linear_input_matrix, inputs = self.kane.linearize()
        # sub in the equilibrium point and the parameters
        self.f_A_lin = self.linear_state_matrix.subs(parameter_dict).subs(equilibrium_dict)
        self.f_B_lin = self.linear_input_matrix.subs(parameter_dict).subs(equilibrium_dict)
        m_mat = self.kane.mass_matrix_full.subs(parameter_dict).subs(equilibrium_dict)
        # compute A and B
        from numpy import matrix
        A = matrix(m_mat.inv() * self.f_A_lin)
        B = matrix(m_mat.inv() * self.f_B_lin)

        assert matrix_rank(control.ctrb(A, B)) == A.shape[0]

        self.K, self.X, self.E = control.lqr(A, B, ones(A.shape), 1);

    def ctrl_derivatives(x, t, args):
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
        u = dot(K, equilibrium_point - x)    # The controller
        arguments = hstack((x, u, args))     # States, input, and parameters
        dx = array(solve(M_func(*arguments), # Solving for the derivatives
            F_func(*arguments))).T[0]

        return dx

