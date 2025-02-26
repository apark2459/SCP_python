import sympy as sp
import numpy as np
import cvxpy as cvx
from utils import euler_to_quat
from global_parameters import K

class Model:
    """
    A 3 degree of freedom rocket landing problem with aerodynamic drag.
    """
    n_x = 7
    n_u = 3

    # Mass
    m_wet = 15000.  # 15000 kg
    m_dry = 10000.  # 10000 kg
    w_mass = 100 # final mass penalty weight

    # Flight time guess
    t_f_guess = 15.  # 15 s

    # State constraints
    r_I_init = np.array((0., 200., 200.))  # 2000 m, 200 m, 200 m
    v_I_init = np.array((-50., -100., -50.))  # -300 m/s, 50 m/s, 50 m/s

    r_I_final = np.array((0., 0., 0.))
    v_I_final = np.array((0., 0., 0.))

    # Angles
    max_angle = 15.
    glidelslope_angle = 80.
    final_angle = 1.

    cos_theta_max = np.cos(np.deg2rad(max_angle))
    tan_gamma_gs = np.tan(np.deg2rad(glidelslope_angle))
    cos_gamma_gs = np.cos(np.deg2rad(glidelslope_angle))

    # Thrust limits
    T_max = 250000.  # 250000 [kg*m/s^2]
    T_min = 100000.  # 100000 [kg*m/s^2]

    # Gravity
    g_I = np.array((0., 0., -9.81))  # -9.81 [m/s^2]

    # Up vector
    e_u = np.array((0., 0., 1.))

    # Aerodynamics
    P_amb = 100000.  # 100000 [Pa]
    A_nozzle = 0.5  # 0.5 [m^2]
    density = 1 # 1 [kg/m^3]
    S_D = 10  # 10 [m^2]
    C_D = 1.0  # 1.0 [-]

    # Fuel consumption
    alpha_m = 1 / (300 * 9.81)  # 1 / (300 * 9.81) [s/m]

    def set_random_initial_state(self):
        self.r_I_init[0:2] = np.random.uniform(-300, 300, size=2)

        self.v_I_init[2] = np.random.uniform(-100, -60)
        self.v_I_init[0:2] = np.random.uniform(-0.5, -0.2, size=2) * self.r_I_init[0:2]

    
    def set_initial_state(self):
        self.r_I_init[0] = 500. # 500 m
        self.r_I_init[1] = 0. # 0 m
        self.r_I_init[2] = 500. # 500 m

        self.v_I_init[0] = 0. # 0 m/s
        self.v_I_init[1] = 50. # 50 m/s
        self.v_I_init[2] = -50. # -50 m/s


    # ------------------------------------------ Start normalization stuff
    def __init__(self):
        """
        A large r_scale for a small scale problem will
        lead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """

        # self.set_random_initial_state() # Random initial state
        self.set_initial_state() # Fixed initial state

        self.x_init = np.concatenate(((self.m_wet,), self.r_I_init, self.v_I_init))
        self.x_final = np.concatenate(((self.m_dry,), self.r_I_final, self.v_I_final))

        self.r_scale = np.linalg.norm(self.r_I_init)
        self.m_scale = self.m_wet


    def nondimensionalize(self):
        """ nondimensionalize all parameters and boundaries """

        self.alpha_m *= self.r_scale  # s
        self.g_I /= self.r_scale  # 1/s^2

        self.x_init = self.x_nondim(self.x_init)
        self.x_final = self.x_nondim(self.x_final)

        self.T_max = self.u_nondim(self.T_max)
        self.T_min = self.u_nondim(self.T_min)

        self.m_wet /= self.m_scale
        self.m_dry /= self.m_scale

        self.S_D /= (self.r_scale ** 2)
        self.A_nozzle /= (self.r_scale ** 2)

        self.density *= (self.r_scale ** 3) / self.m_scale

        self.P_amb /= (self.m_scale / self.r_scale)


    def x_nondim(self, x):
        """ nondimensionalize a single x row """

        x[0] /= self.m_scale
        x[1:4] /= self.r_scale
        x[4:7] /= self.r_scale

        return x

    def u_nondim(self, u):
        """ nondimensionalize u, or in general any force in Newtons"""
        return u / (self.m_scale * self.r_scale)

    def redimensionalize(self):
        """ redimensionalize all parameters """

        self.alpha_m /= self.r_scale  # s
        self.g_I *= self.r_scale

        self.T_max = self.u_redim(self.T_max)
        self.T_min = self.u_redim(self.T_min)

        self.m_wet *= self.m_scale
        self.m_dry *= self.m_scale

        self.S_D *= (self.r_scale ** 2)
        self.A_nozzle *= (self.r_scale ** 2)
        self.density *= self.m_scale / (self.r_scale ** 3)
        self.P_amb *= (self.m_scale / self.r_scale)

    def x_redim(self, x):
        """ redimensionalize x, assumed to have the shape of a solution """

        x[0, :] *= self.m_scale
        x[1:4, :] *= self.r_scale
        x[4:7, :] *= self.r_scale

        return x

    def u_redim(self, u):
        """ redimensionalize u """
        return u * (self.m_scale * self.r_scale)

    # ------------------------------------------ End normalization stuff

    def get_equations(self):
        """
        :return: Functions to calculate A, B and f given state x and input u
        """
        f = sp.zeros(7, 1)

        x = sp.Matrix(sp.symbols('m rx ry rz vx vy vz', real=True))
        u = sp.Matrix(sp.symbols('ux uy uz', real=True))
        v = x[4:7, 0]
        v_norm = v.norm()
        gamma = sp.symbols('gamma', real=True)

        g_I = sp.Matrix(self.g_I)

        f[0, 0] = - self.alpha_m * gamma - self.alpha_m * self.P_amb * self.A_nozzle # mass dynamics
        f[1:4, 0] = x[4:7, 0] # position dynamics
        f[4:7, 0] = 1 / x[0, 0] * (u - 1/2 * self.density * self.C_D * self.S_D * v_norm * v ) + g_I # velocity dynamics

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))
        G = sp.simplify(f.diff(gamma))

        f_func = sp.lambdify((x, u, gamma), f, 'numpy')
        A_func = sp.lambdify((x, u, gamma), A, 'numpy')
        B_func = sp.lambdify((x, u, gamma), B, 'numpy')
        G_func = sp.lambdify((x, u, gamma), G, 'numpy')

        return f_func, A_func, B_func, G_func

    def initialize_trajectory(self, X, U, Gamma):
        """
        Initialize the trajectory.

        :param X: Numpy array of states to be initialized
        :param U: Numpy array of inputs to be initialized
        :return: The initialized X and U
        """

        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K

            m_k = (alpha1 * self.x_init[0] + alpha2 * self.x_final[0],)
            r_I_k = alpha1 * self.x_init[1:4] + alpha2 * self.x_final[1:4]
            v_I_k = alpha1 * self.x_init[4:7] + alpha2 * self.x_final[4:7]

            X[:, k] = np.concatenate((m_k, r_I_k, v_I_k))
            U[:, k] = (self.T_max - self.T_min) / 2 * np.array([0, 0, 1])
            Gamma[0, k] = (self.T_max - self.T_min)/2

        return X, U, Gamma

    def get_objective(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific objective to be minimized.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A cvx objective function.
        """
        # Maximize final mass
        cost = -self.w_mass * X_v[0, -1]
        return cvx.Minimize(cost)

    def get_constraints(self, X_v, U_v, Gamma_v, X_last_p, U_last_p):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints
        """
        # Boundary conditions:
        constraints = [
            X_v[0, 0] == self.x_init[0],
            X_v[1:4, 0] == self.x_init[1:4],
            X_v[4:7, 0] == self.x_init[4:7],
            X_v[1:, -1] == self.x_final[1:],
            np.cos(np.deg2rad(1)) * cvx.norm(U_v[:, -1]) <= U_v[2,-1] # Final thrust must be vertical...could replace norm(U_v[:, -1]) with Gamma_v[-1] and get same solution
        ]

        constraints += [
            # State constraints:
            X_v[0, :] >= self.m_dry,  # lower bound on mass
            # cvx.norm(X_v[1:3, :], axis=0) <= X_v[3, :] / self.tan_gamma_gs,  # glideslope (gamma_gs defined from horizontal)
            cvx.norm(X_v[1:3, :], axis=0) <= X_v[3, :] * self.tan_gamma_gs,  # glideslope (gamma_gs defined from vertical)

            # Control constraints:
            self.cos_theta_max * Gamma_v <= U_v[2, :],

            # Lossless convexification:
            Gamma_v <= self.T_max,
            self.T_min <= Gamma_v,
            cvx.norm(U_v, axis=0) <= Gamma_v
        ]

        return constraints

    def get_linear_cost(self):
        return 0

    def get_nonlinear_cost(self, X=None, U=None):
        magnitude = np.linalg.norm(U, 2, axis=0)
        is_violated = magnitude < self.T_min
        violation = self.T_min - magnitude
        cost = np.sum(is_violated * violation)
        return cost
