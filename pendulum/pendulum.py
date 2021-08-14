import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import functools


GRAVITY_ACC_M_PER_S_PER_S = 9.81
DAMPING_FACTOR_KG_S = 0.5
MASS_KG = 1
LENGTH_M = 1
T_SPAN = (0, 25)
INITIAL_STATE = np.array([np.pi / 2, 0.])


def pendulum_step(
    t: float,
    state: np.ndarray,
    damping_factor_kg_s: float = DAMPING_FACTOR_KG_S,
    length: float = LENGTH_M,
    mass: float = MASS_KG
) -> np.ndarray:
    """Differential equation for determining the next state of a pendulum.

    Args:
        t (float): Current timestep
        state (np.ndarray): Current state
        damping_factor_kg_s (float, optional): Pendulum damping factor
        length (float, optional): Pendulum length
        mass (float, optional): Pendulum mass

    Returns:
        np.ndarray: New state delta from previous state
    """

    new_ang_accel = (-damping_factor_kg_s / mass) * state[1] \
        + (-GRAVITY_ACC_M_PER_S_PER_S / length) * np.sin(state[0])

    return np.hstack([state[1], new_ang_accel])


def pendulum():
    """Simulates pendulum motion, and plots the result.
    """
    state = INITIAL_STATE
    t_s = np.linspace(T_SPAN[0], T_SPAN[1], 1000)
    state_history = solve_ivp(pendulum_step, T_SPAN, state, t_eval=t_s).y

    plt.rcParams["figure.figsize"] = [16, 9]
    fig = plt.subplot(1, 2, 1)
    fig.plot(t_s, state_history[0, :], label="Angular Displacement (rad)")
    fig.plot(t_s, state_history[1, :], label="Angular Velocity (rad/s)")
    fig.legend()
    fig.set_xlabel("Time (s)")

    fig = plt.subplot(1, 2, 2, projection="3d")
    fig.scatter3D(
        t_s,
        state_history[0, :],
        state_history[1, :]
    )

    fig.set_xlabel("Time (s)")
    fig.set_ylabel("Angular Displacement (rad)")
    fig.set_zlabel("Angular Velocity (rad/s)")

    plt.show()


if __name__ == "__main__":
    pendulum()
