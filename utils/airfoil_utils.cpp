#include "airfoil_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Simple natural cubic spline (second derivative zero at endpoints)
class CubicSpline {
public:
    void set_points(const std::vector<double>& x_in,
                    const std::vector<double>& y_in) {
        x = x_in;
        a = y_in;
        int n = x.size();
        if (n < 2) return;

        std::vector<double> h(n-1), alpha(n-1);
        for (int i = 0; i < n-1; ++i) h[i] = x[i+1] - x[i];
        for (int i = 1; i < n-1; ++i) {
            alpha[i] = (3.0/h[i]) * (a[i+1] - a[i])
                     - (3.0/h[i-1]) * (a[i] - a[i-1]);
        }

        std::vector<double> l(n), mu(n), z(n);
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;
        for (int i = 1; i < n-1; ++i) {
            l[i] = 2.0 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i];
        }
        // Allocate storage before writing to c_vec
        b.resize(n-1);
        c_vec.resize(n);
        d.resize(n-1);

        l[n-1] = 1.0;
        z[n-1] = 0.0;
        c_vec[n-1] = 0.0;

        for (int j = n-2; j >= 0; --j) {
            c_vec[j] = z[j] - mu[j] * c_vec[j+1];
            b[j] = (a[j+1] - a[j]) / h[j]
                 - h[j] * (c_vec[j+1] + 2.0 * c_vec[j]) / 3.0;
            d[j] = (c_vec[j+1] - c_vec[j]) / (3.0 * h[j]);
        }
    }

    double operator()(double xq) const {
        auto it = std::upper_bound(x.begin(), x.end(), xq);
        int i = std::max(int(it - x.begin()) - 1, 0);
        double dx = xq - x[i];
        return a[i] + b[i] * dx + c_vec[i] * dx*dx + d[i] * dx*dx*dx;
    }

private:
    std::vector<double> x, a, b, c_vec, d;
};

// Maps [0,1] -> [0,1] clustering near endpoints
static double tanh_map(double u, double a) {
    double t1 = std::tanh(a * (u - 0.5));
    double t2 = std::tanh(a * 0.5);
    return (t1 + t2) / (2.0 * t2);
}

std::pair<std::vector<double>, std::vector<double>> import_and_interpolate_airfoil(
    const std::string& filename,
    int num_points,
    double cluster_factor,
    int fine_samples
) {
    // Load raw data
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);
    std::string line;
    std::getline(in, line); // skip header
    std::vector<double> x_raw, y_raw;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        double xi, yi;
        if (!(ss >> xi >> yi)) continue;
        x_raw.push_back(xi);
        y_raw.push_back(yi);
    }
    int n_raw = x_raw.size();
    if (n_raw < 2) throw std::runtime_error("Not enough data points in file");

    // Parameterize by cumulative distance
    std::vector<double> t_raw(n_raw, 0.0);
    for (int i = 1; i < n_raw; ++i) {
        double dx = x_raw[i] - x_raw[i-1];
        double dy = y_raw[i] - y_raw[i-1];
        t_raw[i] = t_raw[i-1] + std::hypot(dx, dy);
    }
    double L = t_raw.back();
    if (L > 0.0) for (double& v : t_raw) v /= L;
    else for (int i = 0; i < n_raw; ++i) t_raw[i] = double(i) / (n_raw - 1);

    // Build splines
    CubicSpline splx, sply;
    splx.set_points(t_raw, x_raw);
    sply.set_points(t_raw, y_raw);

    // Estimate midpoint
    std::vector<double> t_fine(fine_samples);
    for (int i = 0; i < fine_samples; ++i) t_fine[i] = double(i) / (fine_samples - 1);
    std::vector<double> cum(fine_samples, 0.0);
    for (int i = 1; i < fine_samples; ++i) {
        double dx = splx(t_fine[i]) - splx(t_fine[i-1]);
        double dy = sply(t_fine[i]) - sply(t_fine[i-1]);
        cum[i] = cum[i-1] + std::hypot(dx, dy);
    }
    double half = cum.back() / 2.0;
    double t_mid = 0.5;
    for (int i = 1; i < fine_samples; ++i) {
        if (cum[i] >= half) {
            double frac = (half - cum[i-1]) / (cum[i] - cum[i-1]);
            t_mid = t_fine[i-1] + frac * (t_fine[i] - t_fine[i-1]);
            break;
        }
    }

    // Generate clustered parameters
    int p = int(std::ceil(num_points / 2.0));
    int q = num_points - p + 1;
    std::vector<double> t_new;
    t_new.reserve(num_points);
    for (int i = 0; i < p; ++i) {
        double u = double(i) / (p - 1);
        t_new.push_back(tanh_map(u, cluster_factor) * t_mid);
    }
    for (int i = 1; i < q; ++i) {
        double u = double(i) / (q - 1);
        t_new.push_back(t_mid + tanh_map(u, cluster_factor) * (1.0 - t_mid));
    }

    // Interpolate points
    std::vector<double> x_out(num_points), y_out(num_points);
    for (int i = 0; i < num_points; ++i) {
        x_out[i] = splx(t_new[i]);
        y_out[i] = sply(t_new[i]);
    }
    return {x_out, y_out};
}