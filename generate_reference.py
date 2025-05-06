from constants import *
from utils import (
    B_T_Bj,
    mult_homog_point_np,
    rot_mat_2d_np,
    rot_mat_np,
    flatten_state_np,
    homog_np,
)
import numpy as np
from numpy import pi
from scipy.interpolate import interp1d
from scipy.interpolate import CubicHermiteSpline


# linearly interpolate x and y, evaluate at t
def linear_interp_t(x, y, t):
    f = interp1d(x, y)
    return f(t)


# interpolate x and y, evaluate at t using cubic splines with zero derivatives
# this creates an interpolation similar to linear interpolation, but with
# smoothed corners
def cubic_interp_t(x, y, t):
    f = CubicHermiteSpline(x, y, np.zeros_like(x))
    return f(t)


# sinusoidal function evaluated at t defined using oscillation period, minimum
# and maximum values
def sinusoid(period, min_val, max_val, t, phase_offset=0):
    return (max_val - min_val) / 2.0 * (
            1 - np.cos(2 * np.pi / period * t + phase_offset)
    ) + min_val


# sinusoidal function evaluated at t defined using oscillation period, minimum
# and maximum values
def double_sinusoid(period, min_val, max_val, t, phase_offset=0):
    sig = (max_val - min_val) / 2.0 * (
            1 - np.cos(2 * np.pi / (period / 2) * t + phase_offset)
    ) + min_val
    # return sig
    if (t % period) >= period / 2:
        return 0.0
    else:
        return sig


def generate_reference(motion_type):
    if motion_type == "squat":
        tf = 10.0
    elif motion_type == "trot":
        tf = 10.0
    elif motion_type == "walk":
        tf = 10.0
    elif motion_type == "jump":
        tf = 2.0
    elif motion_type == "bound":
        tf = 10.0
    elif motion_type == "backflip180":
        tf = 5.0
    elif motion_type == "backflip":
        tf = 5.0
    else:
        raise ValueError("Motion type not defined!")

    N = int(tf * 50)  # number of time steps
    dt = tf / (N)  # 0.02s time step
    t_vals = np.linspace(0, tf, N + 1)

    # we represent the motion with single rigid body in the world frame

    # X = (p, R, pdot, omega): R18x(N+1), state trajectory
    # p: R3, position of the body frame origin in the world frame
    # R: R3x3, rotation matrix of the body frame to the world frame
    # pdot: R3, velocity of the body frame origin in the world frame
    # omega: R3, angular velocity of the body frame in the world frame

    # U = (p_j, f_j): R24x(N+1), control trajectory
    # p_j: position of the foot at the world frame
    # f_j: force applied by the foot

    X = np.zeros((18, N + 1))
    U = np.zeros((24, N + 1))

    for k in range(N + 1):
        if motion_type == "squat":
            if k * dt < 0.5 or k * dt > 9.5:
                body_z = 0.2
            else:
                body_z = sinusoid(
                    period=1.0,
                    min_val=0.15,
                    max_val=0.25,
                    t=t_vals[k],
                    phase_offset=np.pi * 0.5,
                )
            p = np.array([0.0, 0.0, body_z])
            R = np.eye(3)
            p_j = {}
            for leg in legs:
                p_j[leg] = B_p_Bj[leg].copy()
        elif motion_type == "trot":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.3
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.3,
                    max_val=0.3,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_j = {}
            for leg in legs:
                p_j[leg] = B_p_Bj[leg].copy()
                p_j[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    if leg == legs.FL or leg == legs.HR:
                        p_j[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], pi / 2.0)
                        )
                    else:
                        p_j[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], 3.0 * pi / 2.0)
                        )
        elif motion_type == "walk":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.15
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.15,
                    max_val=0.15,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_j = {}
            for leg in legs:
                p_j[leg] = B_p_Bj[leg].copy()
                p_j[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    if leg == legs.FL:
                        sig = sinusoid(1.0, -0.05, 0.05, t_vals[k], 0.0 + pi / 2)
                        if (t_vals[k] + 0.5) % 2.0 >= 1.0:
                            sig = 0.0
                        p_j[leg][2] += max(
                            0.0, sig
                        )
                    # TODO: Ex.3 - Implement the walk gait for the other legs
                    ##### only write your code here #####
                    elif leg == legs.HR:
                        sig = sinusoid(1.0, -0.05, 0.05, t_vals[k], 1.5 * np.pi)
                        if (t_vals[k] + 1) % 2.0 < 1.0:
                            sig = 0.0
                        p_j[leg][2] += max(
                            0.0, sig
                        )
                    elif leg == legs.FR:
                        sig = sinusoid(1.0, -0.05, 0.05, t_vals[k], 0.5* np.pi)
                        if (t_vals[k] + 0.5) % 2.0 < 1.0:
                            sig = 0.0
                        p_j[leg][2] += max(
                            0.0, sig
                        )
                    elif leg == legs.HL:
                        sig = sinusoid(1.0, -0.05, 0.05, t_vals[k], 1.5 * np.pi)
                        if (t_vals[k] + 1) % 2.0 >= 1.0:
                            sig = 0.0
                        p_j[leg][2] += max(
                            0.0, sig
                        )
                    
                    #####################################
        elif motion_type == "jump":
            t_apex = 0.25
            z_apex = np.linalg.norm(g) * t_apex ** 2 / 2.0  # kinematic equation
            body_z = cubic_interp_t(
                [0, 0.2 * tf, 0.2 * tf + t_apex, 0.2 * tf + 2 * t_apex, tf],
                [0, 0, z_apex, 0, 0],
                t_vals[k],
            )
            p = np.array([0.0, 0.0, 0.2])
            p[2] += body_z
            R = np.eye(3)
            p_j = {}
            for leg in legs:
                p_j[leg] = B_p_Bj[leg].copy()
                # TODO: Ex.4 - Fix the foot position for the jump gait
                ##### only write your code here #####
                p_j[leg][2] += body_z
                #####################################
        elif motion_type == "bound":
            if k * dt < 0.5 or k * dt > 9.5:
                body_x = -0.6
            else:
                body_x = sinusoid(
                    period=9.0,
                    min_val=-0.6,
                    max_val=0.6,
                    t=t_vals[k],
                    phase_offset=-2 * np.pi / 9.0 * 0.5,
                )
            p = np.array([body_x, 0.0, 0.25])
            R = np.eye(3)
            p_j = {}
            for leg in legs:
                p_j[leg] = B_p_Bj[leg].copy()
                p_j[leg][0] += body_x
                if k * dt < 0.5 or k * dt > 9.5:
                    pass
                else:
                    if leg == legs.FL or leg == legs.FR:
                        p_j[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], pi / 2.0)
                        )
                    else:
                        p_j[leg][2] += max(
                            0.0, sinusoid(0.5, -0.05, 0.05, t_vals[k], 3.0 * pi / 2.0)
                        )
        elif motion_type == "backflip180":
            body_height = 0.2
            # base angle of the body
            angle = cubic_interp_t([0, 2.0, 3.2, tf], [0, 0, np.pi, np.pi], t_vals[k])
            p = np.array([-l_Bx / 2.0, 0.0, body_height])
            p_xz = rot_mat_2d_np(angle) @ np.array([l_Bx / 2.0, 0.0])
            p += np.array([p_xz[0], 0.0, p_xz[1]])
            # TODO: Ex.6 - Implement the 180-backflip for the legs
            ##### only write your code here #####
            
            q = (B_p_Bj[legs.HL] + B_p_Bj[legs.HR]) / 2.0  
            p0 = np.array([0.0, 0.0, body_height])
            R0 = np.eye(3)
            shoulder0 = p0 + R0 @ q
            R_y = np.array([
                [np.cos(-angle), 0, np.sin(-angle)],
                [0,             1, 0],
                [-np.sin(-angle),0, np.cos(-angle)]
            ])
            p = shoulder0 + R_y @ (p0 - shoulder0)
            R = R_y.copy()

            p_j = {}
            for leg in legs:
                if leg in [legs.FL, legs.FR]:
                    shoulder_pos = p + R@ B_p_Bj[leg]
                    r = body_height
                    angle__leg = cubic_interp_t([0, 2.0, 3.2, tf], [0, 0, 2* np.pi, 2 * np.pi], t_vals[k])
                    new_x = shoulder_pos[0] + r * np.sin(angle__leg)
                    new_z = shoulder_pos[2] - r * np.cos(angle__leg)
                    p_j[leg] = np.array([new_x, shoulder_pos[1], new_z])
                else:
                    p_j[leg] = B_p_Bj[leg].copy()
            #####################################
        elif motion_type == "backflip":
            # TODO: Ex.7 - Implement the 360 backflip reference
            ##### only write your code here #####

            # Useful params
            body_height = 0.2
            p0 = np.array([0.0, 0.0, body_height])
            R0 = np.eye(3)
            T_B_initial = homog_np(p0, R0)
            q_hind = (B_p_Bj[legs.HL] + B_p_Bj[legs.HR]) / 2.0
            pivot_hind = p0 + R0 @ q_hind
            pivot2 = np.array([-l_Bx, 0.0, body_height + l_Bx/2.0])
            q_front = (B_p_Bj[legs.FL] + B_p_Bj[legs.FR]) / 2.0
            y_axis = np.array([0, 1, 0])

            # Timings
            t_start = 1.6
            t_end = 2.8
            tf_flip = t_end - t_start
            t1_flip = 0.333 * tf_flip 
            t2_flip = 0.667 * tf_flip 
            t = t_vals[k] 

            # Helper
            def pivot_rotation_transform(pivot, axis, angle):
                R_rot = rot_mat_np(axis, angle)
                T_rot = homog_np(np.zeros(3), R_rot)
                T_pivot = homog_np(pivot, np.eye(3))
                T_inv_pivot = homog_np(-pivot, np.eye(3))
                return T_pivot @ T_rot @ T_inv_pivot
            
            # Poses of the body
            if t < t_start:
                T_B = T_B_initial.copy()

            elif t <= t_end:
                t_shift = t - t_start 

                if t_shift <= t1_flip:
                    angle1 = -cubic_interp_t([0, t1_flip], [0, np.pi/2], t_shift)
                    T_stage1_transform = pivot_rotation_transform(pivot_hind, y_axis, angle1)
                    T_B = T_stage1_transform @ T_B_initial

                elif t_shift <= t2_flip:
                    angle1_end = -np.pi/2
                    T_stage1_final = pivot_rotation_transform(pivot_hind, y_axis, angle1_end)
                    T_B1 = T_stage1_final @ T_B_initial
                    angle2 = -cubic_interp_t([t1_flip, t2_flip], [0, np.pi], t_shift)
                    T_stage2_relative = pivot_rotation_transform(pivot2, y_axis, angle2)
                    T_B = T_stage2_relative @ T_B1
                    
                else:
                    angle1_end = -np.pi/2
                    T_stage1_final = pivot_rotation_transform(pivot_hind, y_axis, angle1_end)
                    T_B1 = T_stage1_final @ T_B_initial
                    angle2_end = -np.pi
                    T_stage2_final_relative = pivot_rotation_transform(pivot2, y_axis, angle2_end)
                    T_B2 = T_stage2_final_relative @ T_B1
                    pivot_front = T_B2[:3, 3] + T_B2[:3, :3] @ q_front
                    angle3 = -cubic_interp_t([t2_flip, tf_flip], [0, np.pi/2], t_shift)
                    T_stage3_relative = pivot_rotation_transform(pivot_front, y_axis, angle3)
                    T_B = T_stage3_relative @ T_B2

            else:
                angle1_end = -np.pi/2
                T_stage1_final = pivot_rotation_transform(pivot_hind, y_axis, angle1_end)
                T_B1 = T_stage1_final @ T_B_initial

                angle2_end = -np.pi
                T_stage2_final_relative = pivot_rotation_transform(pivot2, y_axis, angle2_end)
                T_B2 = T_stage2_final_relative @ T_B1

                pivot_front = T_B2[:3, 3] + T_B2[:3, :3] @ q_front
                angle3_final = -np.pi/2
                T_stage3_final_relative = pivot_rotation_transform(pivot_front, y_axis, angle3_final)
                T_B = T_stage3_final_relative @ T_B2

            p = T_B[:3, 3]
            R = T_B[:3, :3]


            # Poses of the feet
            p_j = {}
            for leg in legs:
                shoulder_pos = p + R@ B_p_Bj[leg]
                r = body_height
                if leg in [legs.FL, legs.FR]:
                    angle__leg = cubic_interp_t([0, t_start, t1_flip + t_start, t2_flip + t_start, t_end, tf],
                                                [0, 0, np.pi, 2 * np.pi, 2* np.pi, 2 * np.pi], t_vals[k])
                    new_x = shoulder_pos[0] + r * np.sin(angle__leg)
                    new_z = shoulder_pos[2] - r * np.cos(angle__leg)
                    p_j[leg] = np.array([new_x, shoulder_pos[1], new_z])
                else: 
                    angle__leg = cubic_interp_t([0, t_start, t1_flip + t_start, t2_flip + t_start, t_end, tf],
                                                [0, 0, 0, np.pi, 2 * np.pi, 2 * np.pi], t_vals[k])
                    new_x = shoulder_pos[0] + r * np.sin(angle__leg)
                    new_z = shoulder_pos[2] - r * np.cos(angle__leg)
                    p_j[leg] = np.array([new_x, shoulder_pos[1], new_z])

            #####################################
        else:
            raise ValueError("Motion type not defined!")

        pdot = np.array([0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        f_j = {}
        for leg in legs:
            f_j[leg] = np.array([0.0, 0.0, 0.0])
            if p_j[leg][2] <= eps:
                # this is only a rough approximation of the force
                f_j[leg][2] = m * np.linalg.norm(g) / 4.0  # quarter the weight
        X[:, k], U[:, k] = flatten_state_np(p, R, pdot, omega, p_j, f_j)

    return X, U, dt
