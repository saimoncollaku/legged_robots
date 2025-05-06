#include <string.h>

#include <vector>

namespace crl {
namespace app {
namespace locomotion {

/**
 * Robot file, initial state file, feet name...
 */
class RobotModel {
public:
    RobotModel(std::string name,         //
               std::string fpath,        //
               std::string spath,        //
               std::vector<std::string> limb_names, //
               double baseDropHeight,    //
               double baseTargetHeight,  //
               double swingFootHeight)
        : name(name),
          filePath(fpath),
          statePath(spath),
          limb_names(limb_names),
          baseDropHeight(baseDropHeight),
          baseTargetHeight(baseTargetHeight),
          swingFootHeight(swingFootHeight) {}

    std::string name;
    std::string filePath;
    std::string statePath;
    std::vector<std::string> limb_names;
    double baseDropHeight; // 
    double baseTargetHeight; // high-level targets that tracking objectives will be made of
    double swingFootHeight;
};

/* global variable */
std::vector<RobotModel> robotModels = {
    // simple robot models
    RobotModel("Dogbot",                                    //
               CRL_DATA_FOLDER "/robots/cora/cora_v4.rbs",  //
               "",                                          //
               {"tibia_0", "tibia_1", "tibia_2", "tibia_3"},//
               0.42,                                        //
               0.42,                                        //
               0.10),
    RobotModel("Spiderpi",                                    //
               CRL_DATA_FOLDER "/robots/spiderpi/spiderpi.rbs",  //
               "",                                          //
               {"leg_0_3", "leg_1_3", "leg_2_3", 
                "leg_3_3", "leg_4_3", "leg_5_3"},
               0.17,                                        //
               0.17,                                        //
               0.035),
};

}  // namespace locomotion
}  // namespace app
}  // namespace crl
