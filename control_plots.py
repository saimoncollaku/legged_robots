import matplotlib.pyplot as plt
from collections import deque
import numpy as np

def setup_pd_plots(plot_window=1000):
    """
    Set up a single figure with a 3x2 grid of subplots:
      - Top row: Joint positions (Joint 1 and Joint 2)
      - Middle row: Joint velocities (Joint 1 and Joint 2)
      - Bottom row: Actuation torques (Joint 1 and Joint 2)
    Returns a dictionary with figure handles, axes, line objects, and history deques.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    
    # Axes handles:
    ax_pos_1, ax_pos_2 = axs[0, 0], axs[0, 1]
    ax_vel_1, ax_vel_2 = axs[1, 0], axs[1, 1]
    ax_tau_1, ax_tau_2 = axs[2, 0], axs[2, 1]
    
    # History deques (for time and each signal)
    history = {
        "time": deque(maxlen=plot_window),
        "q1_des": deque(maxlen=plot_window),
        "q1_act": deque(maxlen=plot_window),
        "q2_des": deque(maxlen=plot_window),
        "q2_act": deque(maxlen=plot_window),
        "qd1_des": deque(maxlen=plot_window),
        "qd1_act": deque(maxlen=plot_window),
        "qd2_des": deque(maxlen=plot_window),
        "qd2_act": deque(maxlen=plot_window),
        "tau1": deque(maxlen=plot_window),
        "tau2": deque(maxlen=plot_window),
    }
    
    # Create line objects for positions
    line_q1_des, = ax_pos_1.plot([], [], '--', label='Joint 1 Desired')
    line_q1_act, = ax_pos_1.plot([], [], label='Joint 1 Actual')
    ax_pos_1.set_title('Joint 1 Position Tracking')
    ax_pos_1.set_ylabel('Radians')
    ax_pos_1.legend()
    
    line_q2_des, = ax_pos_2.plot([], [], '--', label='Joint 2 Desired')
    line_q2_act, = ax_pos_2.plot([], [], label='Joint 2 Actual')
    ax_pos_2.set_title('Joint 2 Position Tracking')
    ax_pos_2.set_ylabel('Radians')
    ax_pos_2.legend()
    
    # Create line objects for velocities
    line_qd1_des, = ax_vel_1.plot([], [], '--', label='Joint 1 Desired')
    line_qd1_act, = ax_vel_1.plot([], [], label='Joint 1 Actual')
    ax_vel_1.set_title('Joint 1 Velocity Tracking')
    ax_vel_1.set_ylabel('Rad/s')
    ax_vel_1.legend()
    
    line_qd2_des, = ax_vel_2.plot([], [], '--', label='Joint 2 Desired')
    line_qd2_act, = ax_vel_2.plot([], [], label='Joint 2 Actual')
    ax_vel_2.set_title('Joint 2 Velocity Tracking')
    ax_vel_2.set_ylabel('Rad/s')
    ax_vel_2.legend()
    
    # Create line objects for actuation torques (tau)
    line_tau1, = ax_tau_1.plot([], [], '--', label='Joint 1 Actuation')
    # You might want to plot the commanded torque if available, but here we plot the computed tau.
    line_tau1_act, = ax_tau_1.plot([], [], label='Joint 1 Commanded')
    ax_tau_1.set_title('Joint 1 Actuation Torque')
    ax_tau_1.set_ylabel('N·m')
    ax_tau_1.legend()
    
    line_tau2, = ax_tau_2.plot([], [], '--', label='Joint 2 Actuation')
    line_tau2_act, = ax_tau_2.plot([], [], label='Joint 2 Commanded')
    ax_tau_2.set_title('Joint 2 Actuation Torque')
    ax_tau_2.set_ylabel('N·m')
    ax_tau_2.legend()
    
    # Label x-axis for bottom row
    ax_tau_1.set_xlabel('Time Step')
    ax_tau_2.set_xlabel('Time Step')
    
    plt.tight_layout()
    
    plots = {
        "fig": fig,
        "axs": axs,
        "ax_pos_1": ax_pos_1, "ax_pos_2": ax_pos_2,
        "ax_vel_1": ax_vel_1, "ax_vel_2": ax_vel_2,
        "ax_tau_1": ax_tau_1, "ax_tau_2": ax_tau_2,
        "line_q1_des": line_q1_des, "line_q1_act": line_q1_act,
        "line_q2_des": line_q2_des, "line_q2_act": line_q2_act,
        "line_qd1_des": line_qd1_des, "line_qd1_act": line_qd1_act,
        "line_qd2_des": line_qd2_des, "line_qd2_act": line_qd2_act,
        "line_tau1": line_tau1, "line_tau2": line_tau2,
        "history": history,
    }
    
    return plots


def update_pd_plots(plots):
    history = plots["history"]
    # Update line objects with history data:
    plots["line_q1_des"].set_data(history["time"], list(history["q1_des"]))
    plots["line_q1_act"].set_data(history["time"], list(history["q1_act"]))
    plots["line_q2_des"].set_data(history["time"], list(history["q2_des"]))
    plots["line_q2_act"].set_data(history["time"], list(history["q2_act"]))
    
    plots["line_qd1_des"].set_data(history["time"], list(history["qd1_des"]))
    plots["line_qd1_act"].set_data(history["time"], list(history["qd1_act"]))
    plots["line_qd2_des"].set_data(history["time"], list(history["qd2_des"]))
    plots["line_qd2_act"].set_data(history["time"], list(history["qd2_act"]))
    
    plots["line_tau1"].set_data(history["time"], list(history["tau1"]))
    plots["line_tau2"].set_data(history["time"], list(history["tau2"]))
    
    # Rescale all axes in one loop:
    for ax in plots["axs"].flatten():
        ax.relim()
        ax.autoscale_view()
    
    plots["fig"].canvas.draw()
    plots["fig"].canvas.flush_events()
    plt.pause(0.001)
