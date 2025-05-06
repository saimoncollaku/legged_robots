import argparse
import os
import enum

import numpy as np
import csv
import mujoco as mj
from mj_viewer import MujocoViewer

import matplotlib.pyplot as plt
from collections import deque

from control_plots import setup_pd_plots, update_pd_plots
from constants import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="csv file to load", type=str, default=None)
    parser.add_argument("-m", "--mode", help="mode to run the simulation, available modes: PLAYBACK, PD, FF, FF_PD",
                        type=str, default="PLAYBACK")

    args = parser.parse_args()
    if args.file is None:
        # there is not csv file to load, throw an error
        raise ValueError("Please provide a csv file to load")
    else:
        # check if the file exists
        if not os.path.exists("csv/" + args.file + ".csv" if ".csv" not in args.file else "csv/" + args.file):
            raise ValueError("File {} does not exist".format(args.file))
        else:
            print("Loading file: {}".format(args.file))

    # load the csv file
    with open("csv/" + args.file + ".csv" if ".csv" not in args.file else "csv/" + args.file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    motion_data = np.array(data)

    # extract data
    # TODO: Ex.8 - Extract the data from the csv file
    ##### only write your code here #####
    t_orig = motion_data[1:, 0].astype(float)

    # Pos
    p = motion_data[1:, 1:4].astype(float)
    quat = motion_data[1:, 4:8].astype(float)
    q_joints = motion_data[1:, 14:22].astype(float)
    qpos = np.hstack((p, quat, q_joints))

    # Vel
    pdot = motion_data[1:, 8:11].astype(float)
    omega = motion_data[1:, 11:14].astype(float)
    qvel_joints = motion_data[1:, 22:30].astype(float)
    qvel = np.hstack((pdot, omega, qvel_joints))

    # Upsampling
    t_end = t_orig[-1]
    t_sim = np.linspace(t_orig[0], t_end, int((t_end - t_orig[0]) * 500) + 1)
    indices = np.searchsorted(t_orig, t_sim, side='right') - 1
    indices = np.clip(indices, 0, len(t_orig)-1)
    t = t_sim
    qpos = qpos[indices, :]
    qvel = qvel[indices, :]
    
    # Acc
    q_ddot_joints= np.gradient(qvel_joints, axis=0)  
    qacc = q_ddot_joints[indices, :]
    
    # Feedforward
    tau_jac = motion_data[1:, 30:38].astype(float)
    tau_jac = tau_jac[indices, :]
    
    ## ? Gains for other motions
        
    # PD gains 
    # k_p_hip = 5
    # k_d_hip = 0.12
    # k_p_knee = 7
    # k_d_knee = 0.12
    # k_p = np.zeros(8)
    # k_d = np.zeros(8)
    # k_p[0::2] = k_p_hip 
    # k_p[1::2] = k_p_knee 
    # k_d[0::2] = k_d_hip
    # k_d[1::2] = k_d_knee
    
    # # Other gains
    # k_ff_hip = 5
    # k_ff_knee = 5
    # k_a_hip = 0.7
    # k_a_knee = 0.5
    # k_ff = np.zeros(8)
    # k_a = np.zeros(8)
    # k_a[0::2] = k_a_hip 
    # k_a[1::2] = k_a_knee 
    # k_ff[0::2] = k_ff_hip
    # k_ff[1::2] = k_ff_knee
    # # k_a[[0, 2]] = 0.24
    # # k_a[[1, 3]] = 0.335
    
    #### ? Gains for backflip 
    k_p_hip_front = 3.5
    k_p_knee_front = 5
    k_p_hip_hind = 5.5
    k_p_knee_hind = 7
    k_d_hip_front = 0.1
    k_d_knee_front = 0.15
    k_d_hip_hind = 0.12
    k_d_knee_hind = 0.18
    k_p = np.zeros(8)
    k_d = np.zeros(8)
    k_p[[0, 2]] = k_p_hip_front
    k_p[[1, 3]] = k_p_knee_front 
    k_p[[4, 6]] = k_p_hip_hind   
    k_p[[5, 7]] = k_p_knee_hind  
    k_d[[0, 2]] = k_d_hip_front
    k_d[[1, 3]] = k_d_knee_front
    k_d[[4, 6]] = k_d_hip_hind
    k_d[[5, 7]] = k_d_knee_hind
    k_ff_hip = 0.2
    k_ff_knee = 0.80
    k_a_hip = 0.7
    k_a_knee = 2.8
    k_ff = np.zeros(8)
    k_a = np.zeros(8)
    k_a[0::2] = k_a_hip 
    k_a[1::2] = k_a_knee 
    k_ff[0::2] = k_ff_hip
    k_ff[1::2] = k_ff_knee
    k_a[[0, 2]] = 0.24
    k_a[[1, 3]] = 0.335

    
    plot_en = False
    #####################################

    # load the mujoco model
    model = mj.MjModel.from_xml_path('assets/scene.xml')
    data = mj.MjData(model)

    viewer = MujocoViewer(model, data, hide_menus=False)

    time_counter = 0
    idx = 0

    # reset the initial state
    def reset_state():
        # TODO: Ex.8 - Reset the initial state
        ##### only write your code here #####
        data.qpos[:] = qpos[0]
        data.qvel[:] = qvel[0]
        data.time = t[0]  
        #####################################

    mj.mj_forward(model, data)

    # simulation rate: 500Hz
    # data rate: 50Hz

    if args.mode == "PLAYBACK":
        # play back the generated motion
        while True:
            if viewer.is_alive:
                # TODO: Ex.8 - Play back the generated motion
                ##### only write your code here #####
                data.qpos[:] = qpos[idx]
                data.qvel[:] = qvel[idx]
                data.time = t[idx]
                idx += 1 
                #####################################
                mj.mj_forward(model, data)
                viewer.render()

            else:
                break
            time_counter += 1

            if idx >= len(t):
                idx = 0  # reset the index, loop the motion
                time_counter = 0
                reset_state()
                print("Looping the motion")

    elif args.mode == "PD":
        # apply a joint PD controller
        
        plots = setup_pd_plots(plot_window=1000)

        while True:
            if viewer.is_alive:
                # TODO: Ex.9 - Apply a joint PD controller
                ##### only write your code here #####

                data.time = t[idx]
                if idx == 0: reset_state()
                
                desired_joint_pos = qpos[idx, 7:]
                desired_joint_vel = qvel[idx, 6:]
                current_joint_pos = data.qpos[7:]
                current_joint_vel = data.qvel[6:]
                tau_pd = k_p * (desired_joint_pos - current_joint_pos) + k_d * (desired_joint_vel - current_joint_vel)
                data.ctrl[:] = tau_pd
                
                idx += 1
                
                # PLOT
                plots["history"]["time"].append(time_counter)
                plots["history"]["q1_des"].append(desired_joint_pos[0])
                plots["history"]["q1_act"].append(current_joint_pos[0])
                plots["history"]["q2_des"].append(desired_joint_pos[1])
                plots["history"]["q2_act"].append(current_joint_pos[1])
                plots["history"]["qd1_des"].append(desired_joint_vel[0])
                plots["history"]["qd1_act"].append(current_joint_vel[0])
                plots["history"]["qd2_des"].append(desired_joint_vel[1])
                plots["history"]["qd2_act"].append(current_joint_vel[1])
                plots["history"]["tau1"].append(data.ctrl[0])
                plots["history"]["tau2"].append(data.ctrl[1])
                if time_counter % 50 == 0 and plot_en == 1:
                    update_pd_plots(plots)
                
                #####################################
                mj.mj_step(model, data)
                viewer.render()

            else:
                break
            time_counter += 1

            if idx >= len(t):
                idx = 0  # reset the index, loop the motion
                time_counter = 0
                reset_state()
                print("Looping the motion")

    elif args.mode == "FF":
        plots = setup_pd_plots(plot_window=1000)

        while True:
            if viewer.is_alive:
                # TODO: Ex.10 - Apply the optimized torque directly
                ##### only write your code here #####

                data.time = t[idx]
                if idx == 0: reset_state()
                
                desired_joint_pos = qpos[idx, 7:]
                desired_joint_vel = qvel[idx, 6:]
                current_joint_pos = data.qpos[7:]
                current_joint_vel = data.qvel[6:]
                desired_joint_vel = qvel[idx, 6:]
                desired_joint_vel = qvel[idx, 6:]
                current_joint_vel = data.qvel[6:]
                
                tau_ff = k_ff * tau_jac[idx, :] 
                data.ctrl[:] = tau_ff
                
                idx += 1
                
                # PLOT
                plots["history"]["time"].append(time_counter)
                plots["history"]["q1_des"].append(desired_joint_pos[0])
                plots["history"]["q1_act"].append(current_joint_pos[0])
                plots["history"]["q2_des"].append(desired_joint_pos[1])
                plots["history"]["q2_act"].append(current_joint_pos[1])
                plots["history"]["qd1_des"].append(desired_joint_vel[0])
                plots["history"]["qd1_act"].append(current_joint_vel[0])
                plots["history"]["qd2_des"].append(desired_joint_vel[1])
                plots["history"]["qd2_act"].append(current_joint_vel[1])
                plots["history"]["tau1"].append(data.ctrl[0])
                plots["history"]["tau2"].append(data.ctrl[1])
                if time_counter % 50 == 0 and plot_en == 1:
                    update_pd_plots(plots)
                
                #####################################
                mj.mj_step(model, data)
                viewer.render()

            else:
                break
            time_counter += 1

            if idx >= len(t):
                idx = 0  # reset the index, loop the motion
                time_counter = 0
                reset_state()
                print("Looping the motion")

    elif args.mode == "FF_PD":
        plots = setup_pd_plots(plot_window=1000)

        while True:
            if viewer.is_alive:
                # TODO: Ex.11 - Apply the optimized torque and a joint PD controller
                ##### only write your code here #####

                data.time = t[idx]
                
                if idx == 0: reset_state()
                
                desired_joint_pos = qpos[idx, 7:]
                desired_joint_vel = qvel[idx, 6:]
                desired_joint_acc = qacc[idx, :]
                current_joint_pos = data.qpos[7:]
                current_joint_vel = data.qvel[6:]
                current_joint_acc = data.qacc[6:]
                
                tau_pd = k_p * (desired_joint_pos - current_joint_pos) + k_d * (desired_joint_vel - current_joint_vel)
                tau_a = - k_a * qacc[idx, :]  # ! USE ONLY FOR BACKFLIP
                tau_ff = k_ff * tau_jac[idx, :]
                data.ctrl[:] = tau_pd + tau_ff + tau_a
                
                idx += 1
                
                # PLOT
                plots["history"]["time"].append(time_counter)
                plots["history"]["q1_des"].append(desired_joint_pos[0])
                plots["history"]["q1_act"].append(current_joint_pos[0])
                plots["history"]["q2_des"].append(desired_joint_pos[1])
                plots["history"]["q2_act"].append(current_joint_pos[1])
                plots["history"]["qd1_des"].append(desired_joint_vel[0])
                plots["history"]["qd1_act"].append(current_joint_vel[0])
                plots["history"]["qd2_des"].append(desired_joint_vel[1])
                plots["history"]["qd2_act"].append(current_joint_vel[1])
                plots["history"]["tau1"].append(data.ctrl[0])
                plots["history"]["tau2"].append(data.ctrl[1])
                if time_counter % 50 == 0 and plot_en == 1:
                    update_pd_plots(plots)
                
                
                #####################################
                mj.mj_step(model, data)
                viewer.render()

            else:
                break
            time_counter += 1

            if idx >= len(t):
                idx = 0  # reset the index, loop the motion
                time_counter = 0
                reset_state()
                print("Looping the motion")
    else:
        raise ValueError("Invalid mode")
