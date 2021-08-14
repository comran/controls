import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import functools

from pendulum import (
    pendulum_step, DAMPING_FACTOR_KG_S, INITIAL_STATE, T_SPAN
)


def pendulum_tune():
    """Tune pendulum parameters over time to minimize integrated position error.
    """
    custom_damping = DAMPING_FACTOR_KG_S / 2.
    best_performance = None
    worst_performance = None
    t_span = (T_SPAN[0], T_SPAN[1] * 0.25)
    t_s = np.linspace(t_span[0], t_span[1], 1000)
    sim_runs = 80

    state_histories = []
    performance_histories = []

    for i in range(sim_runs):
        state = INITIAL_STATE

        custom_damping_pendulum_step = functools.partial(
            pendulum_step,
            damping_factor_kg_s=custom_damping
        )

        state_history = solve_ivp(
            custom_damping_pendulum_step,
            t_span,
            state,
            t_eval=t_s
        ).y

        performance = np.trapz(np.absolute(state_history[0, :]))

        if best_performance is None \
            or performance < performance_histories[best_performance][1]:

            best_performance = i

        if worst_performance is None \
            or performance > performance_histories[worst_performance][1]:

            worst_performance = i

        performance_histories.append([custom_damping, performance])
        state_histories.append(state_history)

        # Set a new custom damping to try out next run based on the learning
        # rate.
        custom_damping = performance_histories[best_performance][0] \
            + np.random.uniform(0, 0.1 + 0.15 * (1. - (i / sim_runs)))

    print(f"Damping with best performance: {performance_histories[best_performance][0]}")
    print(f"Max absolute integrated error: {performance_histories[best_performance][1]}")
    print()
    print(f"Damping with worst performance: {performance_histories[worst_performance][0]}")
    print(f"Max absolute integrated error: {performance_histories[worst_performance][1]}")

    performance_history_np = np.array(performance_histories)

    best_state_history = state_histories[best_performance]
    plt.rcParams["figure.figsize"] = [21, 9]

    fig = plt.subplot(1, 3, 1)
    fig.set_title("Best Damping Sim Path")
    fig.plot(t_s, best_state_history[0, :], label="Angular Displacement (rad)")
    fig.plot(t_s, best_state_history[1, :], label="Angular Velocity (rad/s)")
    fig.legend()
    fig.set_xlabel("Time (s)")

    fig = plt.subplot(1, 3, 2, projection="3d")

    for i, state_history in enumerate(state_histories):

        if i != best_performance and i != worst_performance \
                and i > 10:
            continue

        size = 3 if (i == best_performance or i == worst_performance) else 0.1
        color = "green" if i == best_performance else (
            "red" if i == worst_performance else "blue"
        )
        fig.scatter3D(
            t_s,
            state_history[0, :],
            state_history[1, :],
            c=color,
            s=size
        )

    fig.set_title("Sim Histories")
    fig.set_xlabel("Time (s)")
    fig.set_ylabel("Angular Displacement (rad)")
    fig.set_zlabel("Angular Velocity (rad/s)")

    fig = plt.subplot(1, 3, 3, projection="3d")

    fig.scatter3D(
        np.arange(performance_history_np.shape[0]),
        performance_history_np[:, 0],
        performance_history_np[:, 1]
    )

    fig.set_title("Learning Progress")
    fig.set_xlabel("Sim index")
    fig.set_ylabel("Damping (ks/s)")
    fig.set_zlabel("Integrated Absolute Error")

    plt.show()


if __name__ == "__main__":
    pendulum_tune()
