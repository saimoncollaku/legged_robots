## Project 2 (optim branch) - Trajectory Optimization for a Quadruped

- **Objective:** Implement a trajectory‐optimization pipeline (using CasADi) and evaluate optimized motions in MuJoCo for a planar 8-DOF quadruped.

- **Key Components:**
  - **Reference Generation**: Hand‐designed kinematic sketches for motions (e.g. backflip360).
  - **Trajectory Optimization**: Multiple‐shooting formulation with SRB dynamics, cost on state/control tracking, dynamics & contact constraints.
  - **Controllers**: PLAYBACK (no physics), PD, FF, and FF+PD controllers in Mujoco physics sim.
   
    - 
- **Demo Video**:
<div align="center">
    <img src="playback.gif"">
</div>


