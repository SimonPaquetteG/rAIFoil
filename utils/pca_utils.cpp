#include "pca_utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

static std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>
parse_pca_stream(std::istream& in) {
    std::string line;
    std::vector<double> mean;
    std::vector<std::vector<double>> comps;
    std::vector<double> expl;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "mean") {
            double v;
            while (iss >> v)
                mean.push_back(v);
        } else if (key.rfind("component_", 0) == 0) {
            std::vector<double> c;
            double v;
            while (iss >> v)
                c.push_back(v);
            comps.push_back(c);
        } else if (key == "explained_variance") {
            double v;
            while (iss >> v)
                expl.push_back(v);
        }
    }

    if (mean.empty() || comps.empty() || expl.empty())
        throw std::runtime_error("Malformed PCA .dat data");
    if (expl.size() != comps.size())
        throw std::runtime_error("Explained variance size mismatch");

    return {mean, comps, expl};
}

// Overload: load from file path
std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>
load_pca_dat(const std::string& dat_path) {
    std::ifstream in(dat_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open PCA file: " + dat_path);
    return parse_pca_stream(in);
}

// Overload: load from any istream (e.g., in-memory buffer)
std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>
load_pca_dat(std::istream& in_stream) {
    if (!in_stream)
        throw std::runtime_error("Invalid PCA data stream");
    return parse_pca_stream(in_stream);
}

// PCA application unchanged
std::vector<double> apply_pca(
    const std::vector<double>& mean,
    const std::vector<std::vector<double>>& components,
    const std::vector<double>& explained_variance,
    const std::vector<double>& x
) {
    int n = mean.size();
    if ((int)x.size() != n)
        throw std::runtime_error("PCA input size mismatch");

    std::vector<double> centered(n);
    for (int i = 0; i < n; ++i)
        centered[i] = x[i] - mean[i];

    int m = components.size();
    std::vector<double> y(m);
    for (int i = 0; i < m; ++i) {
        double dot = 0.0;
        for (int j = 0; j < n; ++j)
            dot += components[i][j] * centered[j];
        y[i] = dot / std::sqrt(explained_variance[i]);
    }
    return y;
}
