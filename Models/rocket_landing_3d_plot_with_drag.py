import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d

figures_i = 0

# vector scaling
thrust_scale = 0.00004
attitude_scale = 20


def key_press_event(event):
    global figures_i
    fig = event.canvas.figure

    if event.key == 'q' or event.key == 'escape':
        plt.close(event.canvas.figure)
        return

    if event.key == 'right':
        figures_i = (figures_i + 1) % figures_N
    elif event.key == 'left':
        figures_i = (figures_i - 1) % figures_N

    fig.clear()
    my_plot(fig, figures_i)
    plt.draw()


def plot_trajectories(X_i, U_i, t_f, K):
    """
    Plots the state trajectories X_i in one figure
    and the control inputs U_i in another figure,
    using a custom time axis t = (t_f * k) / K and
    user-defined labels for states and controls.
    
    Parameters
    ----------
    X_i : np.ndarray
        Shape (S, T). The states for a single iteration (figures_i).
        S = number of state variables, T = number of timesteps.
        
    U_i : np.ndarray
        Shape (M, T). The control inputs for a single iteration (figures_i).
        M = number of control variables, T = number of timesteps.

    t_f : float
        Final time to map onto the x-axis.

    K : int
        Total number of time steps (should match T).
    """

    # -----------------------------------------------------------
    # 1) Define label dictionaries for your states and controls
    # -----------------------------------------------------------
    # Edit or expand these if you have more states/controls.

    state_labels = {
        0: r"Mass [kg]",
        1: r"X, east [m]",
        2: r"Y, north [m]",
        3: r"Z, up [m]",
        4: r"$V_X$ [m/s]",
        5: r"$V_Y$ [m/s]",
        6: r"$V_Z$ [m/s]",
    }

    control_labels = {
        0: r"$T_X$ [N]",
        1: r"$T_Y$ [N]",
        2: r"$T_Z$ [N]",
    }

    # -----------------------------------------------------------
    # 2) Check array dimensions vs. K
    # -----------------------------------------------------------
    S, T_X = X_i.shape  # number of states, number of time steps
    M, T_U = U_i.shape  # number of controls, number of time steps

    if T_X != K or T_U != K:
        raise ValueError(
            f"Inconsistent dimensions: X_i has {T_X} timesteps, "
            f"U_i has {T_U}, but K is {K}."
        )

    # -----------------------------------------------------------
    # 3) Create the time array
    # -----------------------------------------------------------
    # time[k] = (t_f * k) / K for k in [0..K-1].
    # If you'd like the last point to be exactly t_f, you could do:
    #     time = np.linspace(0, t_f, K)
    # instead. 
    time = np.array([(t_f * k) / float(K) for k in range(K)])

    # -----------------------------------------------------------
    # 4) Plot states in one figure
    # -----------------------------------------------------------
    fig_states, axes_states = plt.subplots(
        nrows=S, ncols=1, figsize=(8, 2 * S), sharex=True
    )
    if not isinstance(axes_states, np.ndarray):
        # If there's only 1 state, 'axes_states' is not an array
        axes_states = np.array([axes_states])

    for s in range(S):
        axes_states[s].plot(time, X_i[s, :], label=f"State {s}", color='blue')
        # Use our dictionary for a custom y-label if it exists
        y_label = state_labels.get(s, f"X[{s}]")
        axes_states[s].set_ylabel(y_label)

    axes_states[-1].set_xlabel("Time (s)")
    fig_states.suptitle("State Trajectories", fontsize=14)
    fig_states.tight_layout()

    # -----------------------------------------------------------
    # 5) Plot controls in another figure
    # -----------------------------------------------------------
    fig_controls, axes_controls = plt.subplots(
        nrows=M, ncols=1, figsize=(8, 2 * M), sharex=True
    )
    if not isinstance(axes_controls, np.ndarray):
        axes_controls = np.array([axes_controls])

    for m in range(M):
        axes_controls[m].plot(time, U_i[m, :], label=f"Control {m}", color='red')
        # Use our dictionary for a custom y-label if it exists
        y_label = control_labels.get(m, f"U[{m}]")
        axes_controls[m].set_ylabel(y_label)

    axes_controls[-1].set_xlabel("Time (s)")
    fig_controls.suptitle("Control Inputs", fontsize=14)
    fig_controls.tight_layout()

    # -----------------------------------------------------------
    # 6) Show both figures
    # -----------------------------------------------------------
    plt.show()

def plot_nu_norm(all_nu):
    """
    Plots the 2-norm of virtual control nu  against its iteration index. 
    Parameters
    """
    # Check dimensions
    if all_nu.ndim == 1:
        # shape (N,)
        # Each element is already a scalar for each iteration,
        # so the "L2 norm" is just the absolute value.
        norms = np.abs(all_nu)
        N = len(all_nu)
    elif all_nu.ndim == 2:
        # shape (N, D)
        # Compute the L2 norm along axis=1
        norms = np.linalg.norm(all_nu, ord=2, axis=1)
        N = all_nu.shape[0]
    else:
        raise ValueError("all_nu must be 1D or 2D, but has shape {}".format(all_nu.shape))
    
    # Create iteration array
    iterations = np.arange(N)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(iterations, norms, marker='o', label=r"$\|\nu\|_2$", color='blue')

    # Axis labels (Latex for y-axis)
    plt.xlabel("Iteration number")
    plt.ylabel(r"$\|\nu\|_2$")
    plt.title(r"L2 Norm of $\nu$ vs. Iteration Number")
    plt.tight_layout()
    plt.show()


def my_plot(fig, figures_i):
    ax = fig.add_subplot(111, projection='3d')

    X_i = X[figures_i, :, :]
    U_i = U[figures_i, :, :]
    K = X_i.shape[1]

    ax.set_xlabel('X, east')
    ax.set_ylabel('Y, north')
    ax.set_zlabel('Z, up')

    for k in range(K):
        rx, ry, rz = X_i[1:4, k]
        Fx, Fy, Fz = U_i[:, k]

        mag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        if mag > 1e-12:
            Fx_unit = Fx / mag
            Fy_unit = Fy / mag
            Fz_unit = Fz / mag
        else:
            # If the vector is basically zero, no direction
            Fx_unit = 0.0
            Fy_unit = 0.0
            Fz_unit = 0.0

        # attitude vector
        ax.quiver(rx, ry, rz, Fx_unit, Fy_unit, Fz_unit, length=attitude_scale, arrow_length_ratio=0.0, color='blue')

        # thrust vector
        ax.quiver(rx, ry, rz, -Fx, -Fy, -Fz, length=thrust_scale, arrow_length_ratio=0.0, color='red')

    scale = X_i[3, 0]
    ax.auto_scale_xyz([-scale / 2, scale / 2], [-scale / 2, scale / 2], [0, scale])

    pad = plt.Circle((0, 0), 20, color='lightgray')
    ax.add_patch(pad)
    art3d.pathpatch_2d_to_3d(pad)

    ax.set_title("Iteration " + str(figures_i))
    ax.plot(X_i[1, :], X_i[2, :], X_i[3, :], color='lightgrey')
    ax.set_aspect('equal')

def plot(X_in, U_in, sigma_in):
    global figures_N
    figures_N = X_in.shape[0]
    figures_i = figures_N - 1

    global X, U
    X = X_in
    U = U_in

    fig = plt.figure(figsize=(10, 12))
    my_plot(fig, figures_i)
    cid = fig.canvas.mpl_connect('key_press_event', key_press_event)
    plt.show()


if __name__ == "__main__":
    import os

    folder_number = str(int(max(os.listdir('output/trajectory/')))).zfill(3)

    X_in = np.load(f"output/trajectory/{folder_number}/X.npy")
    U_in = np.load(f"output/trajectory/{folder_number}/U.npy")
    sigma_in = np.load(f"output/trajectory/{folder_number}/sigma.npy")

    plot(X_in, U_in, sigma_in)
