from constants import *
from utils import (
    derive_skew_ca,
    derive_rot_mat_ca,
    derive_homog_ca,
    derive_reverse_homog_ca,
    derive_mult_homog_point_ca,
    B_T_Bj,
    extract_state_np,
    extract_state_ca,
)
import numpy as np
import casadi as ca



def traj_opt(X_ref, U_ref, dt):
    skew_ca = derive_skew_ca()
    rot_mat_ca = derive_rot_mat_ca()
    homog_ca = derive_homog_ca()
    reverse_homog_ca = derive_reverse_homog_ca()
    mult_homog_point_ca = derive_mult_homog_point_ca()

    N = X_ref.shape[1]

    opti = ca.Opti()
    X = opti.variable(18, N)
    U = opti.variable(24, N)
    J = ca.MX(1, 1)

    for k in range(N):
        # extract state
        p, R, pdot, omega, p_j, f_j = extract_state_ca(X, U, k)
        if k != (N - 1):
            (
                p_next,
                R_next,
                pdot_next,
                omega_next,
                p_i_next,
                f_i_next,
            ) = extract_state_ca(X, U, k + 1)
        else:
            p_next, R_next, pdot_next, omega_next, p_i_next, f_i_next = (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # extract reference
        p_ref, R_ref, pdot_ref, omega_ref, p_i_ref, f_i_ref = extract_state_ca(
            X_ref, U_ref, k
        )

        # LQR weights
        Q_p = np.array([1000.0, 1000.0, 1000.0])
        Q_p_j = np.array([500, 500.0, 500.0])
        Q_pdot = np.array([10.0, 10.0, 10.0])
        Q_omega = np.array([1.0, 1.0, 1.0])
        Q_f_j = np.array([0.1, 0.1, 0.1])
        Q_R = np.eye(3) * 200.0

        # objective function
        # TODO: Ex.1 - Implement the objective function
        ##### only write your code here #####
        J += ca.mtimes([(p - p_ref).T, ca.diag(Q_p), (p - p_ref)])
        J += (ca.trace(Q_R - Q_R @ R_ref.T @ R))**2
        J += ca.mtimes([(pdot - pdot_ref).T, ca.diag(Q_pdot), (pdot - pdot_ref)])
        J += ca.mtimes([(omega - omega_ref).T, ca.diag(Q_omega), (omega - omega_ref)])
        for leg in p_j.keys():
            J += ca.mtimes([(p_j[leg] - p_i_ref[leg]).T, ca.diag(Q_p_j), (p_j[leg] - p_i_ref[leg])])
            J += ca.mtimes([(f_j[leg] - f_i_ref[leg]).T, ca.diag(Q_f_j), (f_j[leg] - f_i_ref[leg])])
        #####################################

        # dynamics constraints
        # TODO: Ex.1 - Implement the dynamics constraints
        ##### only write your code here #####
        if k != (N - 1):
            # 1
            p_next_est = p + dt * pdot
            opti.subject_to(p_next == p_next_est)
            
            # 2
            sum_forces = ca.MX.zeros(3, 1)
            for leg in f_j.keys():
                sum_forces += f_j[leg]
            pdot_next_est = pdot + dt * ((1 / m) * sum_forces + ca.DM(g))
            opti.subject_to(pdot_next == pdot_next_est)
            
            # 3
            R_next_est = R @ rot_mat_ca(omega, dt)
            opti.subject_to(R_next == R_next_est)
            
            # 4
            sum_moments = ca.MX.zeros(3, 1)
            for leg in p_j.keys():
                r_j = p_j[leg] - p
                sum_moments += ca.cross(r_j, f_j[leg])
            omega_next_est = omega + B_I_inv @ (R.T @ sum_moments - skew_ca(omega) @ B_I @ omega) * dt
            opti.subject_to(omega_next == omega_next_est)
        #####################################

        # contact constraints
        # TODO: Ex.2 - Implement the contact constraints
        ##### only write your code here #####
        for leg in p_j.keys():
            opti.subject_to(p_j[leg][2] >= 0)
            opti.subject_to(f_j[leg][2] * p_j[leg][2] == 0)
            if k != (N - 1):
                opti.subject_to(f_j[leg][2] * (p_i_next[leg][0] - p_j[leg][0]) == 0)
                opti.subject_to(f_j[leg][2] * (p_i_next[leg][1] - p_j[leg][1]) == 0)
                
                # # Magic feet sniffer/smoother
                Q__zzz = np.array([10000.0, 1000.0, 1000.0])
                J += ca.mtimes([(p_i_next[leg] - p_j[leg]).T, ca.diag(Q__zzz), (p_i_next[leg] - p_j[leg])])
        #####################################

        # kinematics constraints
        # TODO: Ex.3 & 4 - Implement the kinematics constraints
        ##### only write your code here #####
        # This makes the solver converge incredibly fast
        for leg in p_j.keys():
            B_p_Bj_leg = ca.DM(B_p_Bj[leg])
            shoulder_pos = p_ref + ca.mtimes(R_ref, B_p_Bj_leg)
            opti.subject_to(p_j[leg][1] == shoulder_pos[1])
            
        # Ex 3 and 4 (3 formulated with slack)
        for leg in p_j.keys():
            B_p_Bj_leg = ca.DM(B_p_Bj[leg])
            shoulder_pos = p + ca.mtimes(R, B_p_Bj_leg)
            shoulder_to_foot = p_j[leg] - shoulder_pos
            leg_extension = ca.norm_2(shoulder_to_foot)
            J += 1e4 * ca.sumsqr(p_j[leg][1] - shoulder_pos[1])
            opti.subject_to(leg_extension <= l_thigh + l_calf)
        #####################################

        f_lim = 20.0  # max vertical force in newtons
        mu = 0.9  # friction coefficient
        # friction pyramid constraints
        # TODO: Ex.5 - Implement the friction pyramid constraints
        ##### only write your code here #####
        for leg in f_j.keys():
            opti.subject_to(f_j[leg][2] >= 0)
            opti.subject_to(f_j[leg][2] <= f_lim)
            # opti.subject_to(ca.sqrt(f_j[leg][0]**2) <= mu * f_j[leg][2])
            # opti.subject_to(ca.sqrt(f_j[leg][1]**2 ) <= mu * f_j[leg][2])
            violation_x = ca.fmax(0, ca.fabs(f_j[leg][0]) - mu * f_j[leg][2])
            violation_y = ca.fmax(0, ca.fabs(f_j[leg][1]) - mu * f_j[leg][2])
            J += 5e3 * (violation_x**2 + violation_y**2)
        #####################################

    # apply objective function
    opti.minimize(J)

    # initial conditions constraint
    # TODO: Ex.1 - Implement the initial condition constraints
    ##### only write your code here #####
    p0, R0, pdot0, omega0, p0_j, _ = extract_state_ca(X, U, 0)
    p0_ref, R0_ref, pdot0_ref, omega0_ref, p0_j_ref, _ = extract_state_ca(X_ref, U_ref, 0)
    opti.subject_to(p0 == p0_ref)
    opti.subject_to(R0 == R0_ref)
    opti.subject_to(pdot0 == pdot0_ref)
    opti.subject_to(omega0 == omega0_ref)
    for leg in p0_j.keys():
        opti.subject_to(p0_j[leg] == p0_j_ref[leg])
        
    #####################################

    # initial solution guess
    opti.set_initial(X, X_ref)
    opti.set_initial(U, U_ref)

    # solve NLP
    p_opts = {}
    s_opts = {
        "print_level": 5,
        "max_iter": 500,
        "mumps_mem_percent": 32000,  # ! ADDED  
        
    }
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()

    # extract solution as numpy array
    X_sol = np.array(sol.value(X))
    U_sol = np.array(sol.value(U))

    return X_sol, U_sol

