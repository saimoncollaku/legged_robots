## Repo structure
This is the repo of my work for the course of "Computational Models of Motion" taught at ETH Zurich.
The repo is divide into 3 branches (with a brief explanation), one for each assignment:
  - **main**
  - **optim**
  - **rl**


## Project 1 (main branch) - Kinematic walking controller

- **Objective**: Develop a kinematic walking controller for quadruped and hexapod robots.

- **Key Components Implemented:**
  - **Forward Kinematics**: Calculated foot positions from joint angles.
  - **Finite-Difference Jacobian**: Estimated the Jacobian of each foot via central differences.
  - **Gauss–Newton IK Solver**: Solved per-leg inverse kinematics and assembled full-body commands.
  - **Base Trajectory Planning**: Integrated user velocity commands to update target base pose.
  - **Uneven-Terrain Adaptation**: Offset foot and base targets using height maps for bumpy ground.
  - **Hexapod Gait Design**: Defined a coordinated six-leg swing/stance pattern for stable hexapod walking.

- **Demo Videos**:
https://github.com/user-attachments/assets/3bf80637-9cbf-4db9-bb7a-4c35e54ef575
https://github.com/user-attachments/assets/9f064297-c26b-4841-9b27-251f7d8b7606

