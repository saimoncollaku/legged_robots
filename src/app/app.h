#pragma once

#include <gui/application.h>
#include <gui/camera.h>
#include <gui/light.h>
#include <gui/renderer.h>
#include <gui/shader.h>
#include <robot/GeneralizedCoordinatesRobotRepresentation.h>
#include <kinematics/IK_Solver.h>
#include <locomotion/FootFallPattern.h>
#include <locomotion/KinematicTrackingController.h>
#include <locomotion/SimpleLocomotionTrajectoryPlanner.h>
#include <robot/Robot.h>
#include <utils/logger.h>

#include "robots.h"

using namespace crl::gui;

namespace crl {
namespace app {
namespace locomotion {

class App : public Basic3DAppWithShadows {

// * NEW EX 4   
private:
    gui::Model terrain;

public:
    App(const char *title = "CRL Playground - Locomotion App - kinematic",
        std::string iconPath = CRL_DATA_FOLDER "/crl_icon_grey.png")
        // * NEW EX 4
        : Basic3DAppWithShadows(title, iconPath){
        camera = TrackingCamera(5);
        camera.rotAboutUpAxis = -0.75;
        camera.rotAboutRightAxis = 0.5;
        light.s = 0.04f;
        shadowbias = 0.00001f;
        camera.aspectRatio = float(width) / height;
        glEnable(GL_DEPTH_TEST);

        showConsole = true;
        automanageConsole = true;
        Logger::maxConsoleLineCount = 10;
        consoleHeight = 225;

        this->targetFramerate = 30;
        this->limitFramerate = true;

        // ------------------------------------------------------------------------
        // TODO: ex 5 change the robot model from index 0 to 1
        setupRobotAndController(robotModels[1]);
    }

    virtual ~App() override {
        if (controller) {
            delete controller->planner;
            delete controller;
        }
    }

    virtual void resizeWindow(int width, int height) override {
        camera.aspectRatio = float(width) / height;
        return Application::resizeWindow(width, height);
    }

    bool mouseMove(double xpos, double ypos) override {
        camera.processMouseMove(mouseState, keyboardState);
        return true;
    }

    virtual bool scrollWheel(double xoffset, double yoffset) override {
        camera.processMouseScroll(xoffset, yoffset);
        return true;
    }

    void saveTrajectoryToJson() {
        std::ofstream file(CRL_DATA_FOLDER "/trajectory.json");
        if (file.is_open()) {
            file << "{\n";
            file << "\"dt\": " << dt << ",\n";

            file << "\"joint_names\": [\n";
            file << "\"" << "base_x" << "\",\n";
            file << "\"" << "base_y" << "\",\n";
            file << "\"" << "base_z" << "\",\n";
            file << "\"" << "base_roll" << "\",\n";
            file << "\"" << "base_pitch" << "\",\n";
            file << "\"" << "base_yaw" << "\",\n";

            GeneralizedCoordinatesRobotRepresentation gcrr(robot);
            for (int i = 6; i < gcrr.getDOFCount(); i++) {
                file << "\"" << gcrr.getJointForQIdx(i)->name << "\"";
                if (i < gcrr.getDOFCount() - 1) file << ",\n";
            }
            file << "],\n";
            file << "\"trajectory\": [\n";
            for (uint i = 0; i < controller->trajectory.size(); i++) {
                file << "[";
                for (uint j = 0; j < controller->trajectory[i].size(); j++) {
                    file << controller->trajectory[i][j];
                    if (j < controller->trajectory[i].size() - 1) file << ", ";
                }
                file << "]";
                if (i < controller->trajectory.size() - 1) file << ",\n";
            }
            file << "]\n";
            file << "}\n";
            file.close();
        }
    }

    void process() override {
        if (appIsRunning == false) return;

        controller->planner->appendPeriodicGaitIfNeeded(getPeriodicGait(robot));

        double simTime = 0;
        while (simTime < 1.0 / targetFramerate) {
            if (printIKTargets) {
                double t = controller->planner->getSimTime() + dt;
                P3D basePosTarget =
                    controller->planner->getTargetTrunkPositionAtTime(t);
                Quaternion baseOriTarget =
                    controller->planner->getTargetTrunkOrientationAtTime(t);
                Logger::consolePrint(
                    "Base: (%lf, %lf, %lf), (%lf, %lf, %lf, %lf)",
                    basePosTarget.x, basePosTarget.y, basePosTarget.z,
                    baseOriTarget.w(), baseOriTarget.x(), baseOriTarget.y(),
                    baseOriTarget.z());

                for (uint i = 0; i < robot->limbs.size(); i++) {
                    P3D footPosTarget =
                        controller->planner->getTargetLimbEEPositionAtTime(
                            robot->limbs[i], t);
                    Logger::consolePrint("Foot %d: (%lf, %lf, %lf)", i,
                                         footPosTarget.x, footPosTarget.y,
                                         footPosTarget.z);
                }
            }

            simTime += dt;
            // this is where IK is triggered and resulting conf q applied
            if (recordTrajectory && !prevRecordTrajectory) {
                // start recording
                controller->resetRecordedTrajectory();
                Logger::consolePrint("Start recording!");
            } else if (!recordTrajectory && prevRecordTrajectory) {
                // stop recording
                GeneralizedCoordinatesRobotRepresentation gcrr(robot);
                dVector lastTrajectoryPoint = controller->trajectory.back();
                std::ostringstream oss;
                for (int i = 6; i < lastTrajectoryPoint.size(); i++) {
                    if (i != 0) oss << ", ";
                    auto joint = gcrr.getJointForQIdx(i);
                    oss << joint->name.c_str();
                }
                Logger::consolePrint("%s", oss.str().c_str());
                saveTrajectoryToJson();
                Logger::consolePrint("Stop recording!");
            }

            if (recordTrajectory && prevRecordTrajectory) {
                dVector lastTrajectoryPoint = controller->trajectory.back();
                std::ostringstream oss;
                for (int i = 0; i < lastTrajectoryPoint.size(); ++i) {
                    if (i != 0) oss << ", ";
                    oss << lastTrajectoryPoint[i];
                }
                Logger::consolePrint("traj pt [%d]: %s", controller->trajectory.size(), oss.str().c_str());
            }

            controller->computeAndApplyControlSignals(dt, recordTrajectory);
            controller->advanceInTime(dt);

            if (cameraShouldFollowRobot) {
                camera.target.x = robot->trunk->state.pos.x;
                camera.target.z = robot->trunk->state.pos.z;
            }

            light.target.x() = robot->trunk->state.pos.x;
            light.target.z() = robot->trunk->state.pos.z;

            // update recording button state
            prevRecordTrajectory = recordTrajectory;
        }

        controller->generateMotionTrajectories();
    }

    virtual void drawAuxiliaryInfo() override {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawFPS();

        drawConsole();

        drawImGui();

        ImGui::EndFrame();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    // objects drawn with a shadowMapRenderer (during shadow pass) will cast a
    // shadow
    virtual void drawShadowCastingObjects() override {
        robot->draw(shadowMapRenderer);
    }

    // objects drawn with a shadowShader (during the render pass) will have
    // shadows cast on them
    virtual void drawObjectsWithShadows() override {
        ground.draw(shadowShader, V3D(0.6, 0.6, 0.8));
        // * NEW EX 4
        terrain.draw(shadowShader, V3D(0.60, 0.46, 0.32));  
        robot->draw(shadowShader);
    }

    // objects drawn with basic shadowShader (during the render pass) will not
    // have shadows cast on them
    virtual void drawObjectsWithoutShadows() override {
        // draw sphere around feet
        // if Ex.1 is correctly implemented, green sphere will be rendered around the feet
        GeneralizedCoordinatesRobotRepresentation gcrr(robot);
        for (uint i = 0; i < robot->limbs.size(); i++) {
            P3D eePos = gcrr.getWorldCoordinates(
                robot->limbs[i]->ee->endEffectorOffset, robot->limbs[i]->eeRB);
            drawSphere(eePos, 0.025, basicShader, V3D(0, 1, 0));
        }

        if (drawDebugInfo) controller->drawDebugInfo(&basicShader);
    }

    virtual bool keyPressed(int key, int mods) override {
        bool dirty = false;
        if (key == GLFW_KEY_SPACE) {
            appIsRunning = !appIsRunning;
        }
        if (key == GLFW_KEY_BACKSPACE) {
            screenIsRecording = !screenIsRecording;
        }
        if (key == GLFW_KEY_UP) {
            controller->planner->speedForward += 0.1;
            dirty = true;
        }
        if (key == GLFW_KEY_DOWN) {
            controller->planner->speedForward -= 0.1;
            dirty = true;
        }
        if (key == GLFW_KEY_LEFT) {
            controller->planner->turningSpeed += 0.1;
            dirty = true;
        }
        if (key == GLFW_KEY_RIGHT) {
            controller->planner->turningSpeed -= 0.1;
            dirty = true;
        }

        if (dirty) {
            controller->planner->appendPeriodicGaitIfNeeded(
                getPeriodicGait(robot));
            controller->generateMotionTrajectories();

            return true;
        }

        return false;
    }

    virtual void drawImGui() {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Once);
        ImGui::Begin("Main Menu");

        ImGui::Text("Play:");
        ImGui::SameLine();
        PlayPauseButton("Play App", &appIsRunning);

        if (appIsRunning == false) {
            ImGui::SameLine();
            if (ImGui::ArrowButton("tmp", ImGuiDir_Right)) {
                appIsRunning = true;
                process();
                appIsRunning = false;
            }
        }

        ImGui::Checkbox("Follow Robot With Camera", &cameraShouldFollowRobot);

        ImGui::Text("Record trajectory to JSON file:");
        ImGui::SameLine();
        ToggleButton("RecordTraj", &recordTrajectory);

        ImGui::Text("Controller options...");

        ImGui::Text("Locomotion Control options...");

        bool rebuildPlan = false;

        rebuildPlan |= ImGui::InputDouble(
            "Speed forward", &controller->planner->speedForward, 0.01);
        rebuildPlan |= ImGui::InputDouble(
            "Speed sideways", &controller->planner->speedSideways, 0.01);
        rebuildPlan |= ImGui::InputDouble(
            "Turning speed", &controller->planner->turningSpeed, 0.01);

        rebuildPlan |= ImGui::InputDouble(
            "Body Height", &controller->planner->trunkHeight, 0.01);
        rebuildPlan |= ImGui::InputDouble(
            "Swing Foot Height", &controller->planner->targetStepHeight, 0.01);
        rebuildPlan |= ImGui::InputDouble(
            "Step Width Ratio", &controller->planner->stepWidthModifier, 0.01);

        bool updateDrawing = false;

        if (ImGui::TreeNode("Draw options...")) {
            updateDrawing |= ImGui::Checkbox("Draw Meshes", &showMeshes);
            updateDrawing |= ImGui::Checkbox("Draw Skeleton", &showSkeleton);
            updateDrawing |= ImGui::Checkbox("Draw Joint Axes", &showJointAxes);
            updateDrawing |=
                ImGui::Checkbox("Draw Joint Limits", &showJointLimits);
            updateDrawing |= ImGui::Checkbox("Draw Collision Primitives",
                                             &showCollisionSpheres);
            updateDrawing |=
                ImGui::Checkbox("Draw End Effectors", &showEndEffectors);
            if (showEndEffectors)
                updateDrawing |= ImGui::SliderFloat(
                    "End Effector R", &eeDrawRadius, 0.01f, 0.1f, "R = %.2f");
            updateDrawing |= ImGui::Checkbox("Draw MOI box", &showMOI);

            ImGui::TreePop();
        }

        if (updateDrawing) {
            updateDrawingOption(robot);
        }

        if (ImGui::TreeNode("Debug options...")) {
            ImGui::Checkbox("Draw debug info", &drawDebugInfo);
            ImGui::Checkbox("Print ik targets", &printIKTargets);
            ImGui::TreePop();
        }

        ImGui::End();

        controller->planner->visualizeContactSchedule();
        controller->planner->visualizeParameters();
    }

    virtual bool drop(int count, const char **fileNames) override {
        return true;
    }

private:
    void setupRobotAndController(const RobotModel &model) {
        // kinematic
        if (robot) delete robot;
        if (controller) {
            delete controller->planner;
            delete controller;
        }
        robot =
            new LeggedRobot(model.filePath.c_str(), model.statePath.c_str());
        robot->setRootState(P3D(0, model.baseDropHeight, 0),
                            Quaternion::Identity());
        for (auto limbName : model.limb_names) {
            robot->addLimb(limbName, robot->getRBByName(limbName.c_str()));
        }
        controller = new KinematicTrackingController(
            // * NEW EX 4
            new SimpleLocomotionTrajectoryPlanner(robot, &terrain));

        controller->planner->trunkHeight = model.baseTargetHeight;
        controller->planner->targetStepHeight = model.swingFootHeight;
        controller->planner->appendPeriodicGaitIfNeeded(getPeriodicGait(robot));

        controller->generateMotionTrajectories();

        updateDrawingOption(robot);
    }

    PeriodicGait getPeriodicGait(LeggedRobot *robot) {
        PeriodicGait pg;

        // Foot contact timeline for the quadrupe robot
        double tOffset = -0.0;
        // * NEW EX 5 
        // ? bipod-b best in robots
        pg.addSwingPhaseForLimb(robot->limbs[2], 0 - tOffset, 0.333 + tOffset);
        pg.addSwingPhaseForLimb(robot->limbs[1], 0.333 - tOffset, 0.666 + tOffset);
        pg.addSwingPhaseForLimb(robot->limbs[0], 0.666 - tOffset, 1 + tOffset);
        pg.addSwingPhaseForLimb(robot->limbs[3], 0 - tOffset, 0.333 + tOffset);
        pg.addSwingPhaseForLimb(robot->limbs[4], 0.333 - tOffset, 0.666 + tOffset);
        pg.addSwingPhaseForLimb(robot->limbs[5], 0.666 - tOffset, 1 + tOffset);

        // ? tripod best in insects
        // pg.addSwingPhaseForLimb(robot->limbs[2], 0 - tOffset, 0.5 + tOffset);
        // pg.addSwingPhaseForLimb(robot->limbs[1], 0.5 - tOffset, 1 + tOffset);
        // pg.addSwingPhaseForLimb(robot->limbs[0], 0 - tOffset, 0.5 + tOffset);
        // pg.addSwingPhaseForLimb(robot->limbs[3], 0.5 - tOffset, 1 + tOffset);
        // pg.addSwingPhaseForLimb(robot->limbs[4], 0 - tOffset, 0.5 + tOffset);
        // pg.addSwingPhaseForLimb(robot->limbs[5], 0.5 - tOffset, 1 + tOffset);
        
        pg.strideDuration = 0.7;

        // TODO: ex 5 comment the code above and replace with your own gait pattern for the hexpod
        // ----------------------------------------------------------------

        return pg;
    }

    void updateDrawingOption(LeggedRobot *robot) {
        robot->showMeshes = showMeshes;
        robot->showSkeleton = showSkeleton;
        robot->showJointAxes = showJointAxes;
        robot->showJointLimits = showJointLimits;
        robot->showCollisionSpheres = showCollisionSpheres;
        robot->showEndEffectors = showEndEffectors;
        robot->showMOI = showMOI;

        for (auto rb : robot->rbList) {
            rb->rbProps.endEffectorRadius = eeDrawRadius;
        }
    }

public:
    SimpleGroundModel ground;

    LeggedRobot *robot = nullptr;

    // controllers
    KinematicTrackingController *controller = nullptr;

    double dt = 1 / 30.0;

    bool drawDebugInfo = true;
    bool printIKTargets = false;
    bool appIsRunning = false;
    bool cameraShouldFollowRobot = true;
    bool recordTrajectory = false;
    bool prevRecordTrajectory = false;

    // UI
    bool showMeshes = true;
    bool showSkeleton = false;
    bool showJointAxes = false;
    bool showJointLimits = false;
    bool showCollisionSpheres = false;
    bool showEndEffectors = false;
    bool showMOI = false;
    float eeDrawRadius = 0.01f;
};

}  // namespace locomotion
}  // namespace app
}  // namespace crl