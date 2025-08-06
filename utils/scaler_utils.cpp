#include "scaler_utils.h"

std::tuple<std::vector<std::string>,
           std::unordered_map<std::string,double>,
           std::unordered_map<std::string,double>>
load_scaler_params()
{
    static const std::vector<std::string> order = {
        "CL",
        "CD",
        "CMZ"
    };

    static const std::unordered_map<std::string,double> means = {
        {"CL",  0.018082208225102592},
        {"CD",  0.6111908141458435},
        {"CMZ",  0.04303411976211617}
    };

    static const std::unordered_map<std::string,double> scales = {
        {"CL",  0.7791480812163655},
        {"CD",  0.46287369547352497},
        {"CMZ",  0.3292234104871026}
    };

    return { order, means, scales };
}
