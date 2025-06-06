#pragma once

#include <robot/GeneralizedCoordinatesRobotRepresentation.h>
#include <robot/Robot.h>

namespace crl {

struct IK_EndEffectorTargets {
    RB *rb = nullptr;
    P3D p;       // local coordinates of end effector in rb's frame
    P3D target;  // target position in world frame
};

class IK_Solver {
public:
    IK_Solver(Robot *robot) : robot(robot) {}

    ~IK_Solver(void) {}

    /**
     * add IK end effector target to solver. Specify the end effector point p, which 
     * is specified in the local coordinates of rb and its target expressed in world frame.
     */
    void addEndEffectorTarget(RB *rb, P3D p, P3D target) {
        endEffectorTargets.push_back(IK_EndEffectorTargets());
        endEffectorTargets.back().rb = rb;
        endEffectorTargets.back().p = p;
        endEffectorTargets.back().target = target;
    }

    void solve(dVector &q, int nSteps = 10) {
        GeneralizedCoordinatesRobotRepresentation gcrr(robot);
        
        for (uint i = 0; i < nSteps; i++) {
            gcrr.getQ(q);

            // get current generalized coordinates of the robots

            // TODO: Ex.2-2 Inverse Kinematics
            //
            // update generalized coordinates of the robot by solving IK.

            // remember, we don't update base pose since we assume it's already at
            // the target position and orientation
            dVector deltaq(q.size() - 6);
            deltaq.setZero();

            // TODO: here, compute deltaq using GD, Newton, or Gauss-Newton.
            // end effector targets are stored in endEffectorTargets vector.
            //
            // Hint:
            // - use gcrr.estimate_linear_jacobian(p, rb, dpdq) function for Jacobian matrix.
            // - if you already implemented analytic Jacobian, you can use gcrr.compute_dpdq(const P3D &p, RB *rb, Matrix &dpdq)
            // - don't forget we use only last q.size() - 6 columns (use block(0,6,3,q.size() - 6) function)
            // - when you compute inverse of the matrix, use ldlt().solve() instead of inverse() function. this is numerically more stable.
            //   see https://eigen.tuxfamily.org/dox-devel/group__LeastSquares.html

            Matrix J;
            dVector e(3 * endEffectorTargets.size()); // Error vector

            // Iterating through the parts
            for (size_t j = 0; j < endEffectorTargets.size(); j++) {
                RB *rb = endEffectorTargets[j].rb;
                P3D pLocal = endEffectorTargets[j].p;
                P3D pTarget = endEffectorTargets[j].target;
                P3D pCurrent = gcrr.getWorldCoordinates(pLocal, rb);

                // Compute position error
                V3D error = V3D(pTarget - pCurrent);
                e.segment<3>(j * 3) = error;

                // Compute Jacobian
                Matrix J_full;
                gcrr.estimate_linear_jacobian(pLocal, rb, J_full);

                // Expanding the jacobian for the additional end-effectors
                J.conservativeResize(3 * (j + 1), q.size() - 6);
                J.block(j * 3, 0, 3, q.size() - 6) = J_full.block(0, 6, 3, q.size() - 6).eval();
            }

            // Solving for DELTAq via Gauss-Newton + Cholesky for inverse
            deltaq = (J.transpose() * J).ldlt().solve(J.transpose() * e);

            q.tail(q.size() - 6) += deltaq;

            gcrr.setQ(q);
        }

        gcrr.syncRobotStateWithGeneralizedCoordinates();

        // clear end effector targets
        // we will add targets in the next step again.
        endEffectorTargets.clear();
    }

private:
    Robot *robot;
    std::vector<IK_EndEffectorTargets> endEffectorTargets;
};

}  // namespace crl