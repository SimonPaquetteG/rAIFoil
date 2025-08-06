#ifndef SCALER_UTILS_H
#define SCALER_UTILS_H

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>

// Loads output scaler params (feature, mean, scale)
std::tuple<std::vector<std::string>, std::unordered_map<std::string,double>, std::unordered_map<std::string,double>>
load_scaler_params();

#endif