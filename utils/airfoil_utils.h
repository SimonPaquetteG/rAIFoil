#ifndef AIRFOIL_UTILS_H
#define AIRFOIL_UTILS_H

#include <vector>
#include <string>
#include <utility>

// Reads a Selig .dat airfoil, interpolates and redistributes points
// filename: path to .dat file
// num_points: desired total points (upper+lower)
// cluster_factor: endpoint clustering strength
// fine_samples: samples to estimate midpoint
std::pair<std::vector<double>, std::vector<double>> import_and_interpolate_airfoil(
    const std::string& filename,
    int num_points,
    double cluster_factor,
    int fine_samples
);

#endif