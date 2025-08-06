#ifndef PCA_UTILS_H
#define PCA_UTILS_H

#pragma once
#include <string>
#include <vector>
#include <tuple>
#include <istream>

// Load PCA from a .dat file on disk
std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>
load_pca_dat(const std::string& dat_path);

// Load PCA from any std::istream (e.g., in-memory buffer)
std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>
load_pca_dat(std::istream& in_stream);

// Apply PCA transform (unchanged)
std::vector<double> apply_pca(
    const std::vector<double>& mean,
    const std::vector<std::vector<double>>& components,
    const std::vector<double>& explained_variance,
    const std::vector<double>& x
);

#endif