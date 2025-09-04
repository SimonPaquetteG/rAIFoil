#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>

#include <torch/script.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <boost/program_options.hpp>
#include <boost/math/distributions/normal.hpp>

#include "utils/help.h"
#include "utils/airfoil_utils.h"
#include "utils/pca_utils.h"
#include "utils/scaler_utils.h"
#include "ensemble_data.h"

namespace po = boost::program_options;
using Clock = std::chrono::high_resolution_clock;
namespace fs = std::filesystem;

// Helper struct to hold PCA parameters
struct PcaTransform {
    std::vector<double> mean;
    std::vector<std::vector<double>> comps;
    std::vector<double> expl;
};

int main(int argc, char** argv) {
    auto t0_all = Clock::now();
    auto t0_init = Clock::now();

    // Parse command-line options
    po::options_description cmd_desc("Allowed options");
    cmd_desc.add_options()
        ("help,h", "Show detailed usage information")
        ("config,i", po::value<std::string>()->required(), "Path to input config file");
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, cmd_desc), vm);

        if (vm.count("help")) {
            std::cout << help_text;
            return 0;
        }

        po::notify(vm);
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n\n"
                  << cmd_desc << "\n";
        return 1;
    }

    // ASCII banner
    auto t0_ascii = Clock::now();
    std::cout << R"(
███████████████████████████████
               ██                      ████                  █████        ████████     ███████████                ██    ██
           ███████████                ██  █                  ██████         ████       ██                              ███
        ███     ██    ██             ██  ██       ██████    ███ ████        ████      ██             ██████      ██    ██
      ██        ██      ██████████████████       ███       ████  ███        ████      ██           ██     ███    ██    ██
     ██        ██                      ██        ██        ███   ████       ████      █████████   ██       ██   ███   ███
     ███████████         ███████████████         ██       ████    ████      ████     ███          ██       ██   ██    ██
     ██                 ██                      ███      █████████████      ████     ██           ██       ██   ██    ██
      ████ █████████ ████                       ██       ███       ████     ████     ██           ██     ███    ██   ███
   ██     █         █                           ██      ████        ████  ████████  ███            ███████     ██    ██
    ███████████████████████
)" << std::endl;
    std::cout << "Written by Simon Paquette under the supervision of Professors David Vidal and Eric Laurendeau.\n\n";
    auto t1_ascii = Clock::now();

    torch::Device device = 
        torch::cuda::is_available()
          ? torch::Device(torch::kCUDA, 0)
          : torch::Device(torch::kCPU, 0);

    if (device.is_cuda()) {
        std::cout << "Device: CUDA\n";
    } else {
        std::cout << "Device: CPU\n";
    }

    // If it’s CUDA, query the GPU name via the CUDA runtime API
    if (device.is_cuda()) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device.index());
        if (err == cudaSuccess) {
            std::cout << "Name: " << prop.name << "\n";
        } else {
            std::cerr << "Failed to get CUDA device properties: "
                      << cudaGetErrorString(err) << "\n";
        }
    }

    std::cout << "\n";

    std::cout << std::fixed << std::setprecision(2);

    // Parse config file
    po::options_description cfg_desc("Config file options");
    cfg_desc.add_options()
        ("OUTPUT_FILE",      po::value<std::string>()->required(), "Output filename prefix without extension")
        ("OUTPUT_FOLDER",    po::value<std::string>()->default_value("."), "Directory to write output files")
        ("GEOMETRY",         po::value<std::string>(),             "Airfoil .dat file path")
        ("GEOMETRY_FOLDER",  po::value<std::string>(),             "Directory containing multiple .dat airfoil files")
        ("AOA_MIN",          po::value<double>()->required(),      "Minimum angle of attack (deg)")
        ("AOA_MAX",          po::value<double>()->required(),      "Maximum angle of attack (deg)")
        ("AOA_STEP",         po::value<double>()->required(),      "AOA step (deg)")
        ("MACH_MIN",         po::value<double>()->required(),      "Minimum Mach number")
        ("MACH_MAX",         po::value<double>()->required(),      "Maximum Mach number")
        ("MACH_STEP",        po::value<double>()->required(),      "Mach step size")
        ("REYNOLDS_MIN",     po::value<double>()->required(),      "Minimum Reynolds number")
        ("REYNOLDS_MAX",     po::value<double>()->required(),      "Maximum Reynolds number")
        ("REYNOLDS_STEP",    po::value<double>()->required(),      "Reynolds step size")
        ("CI_LEVEL",         po::value<double>()->required(),      "Confidence level for CI (e.g., 0.95)")
        ("OUTPUT_FORMAT",    po::value<std::string>()->default_value("csv"),
                              "Output format: 'csv' (default) or 'c81'");
    std::ifstream cfg_stream(vm["config"].as<std::string>());
    if (!cfg_stream) {
        std::cerr << "Error: cannot open config file " << vm["config"].as<std::string>() << "\n";
        return 1;
    }
    po::store(po::parse_config_file(cfg_stream, cfg_desc, true), vm);
    po::notify(vm);

    // Normalize OUTPUT_FORMAT
    std::string output_format = vm["OUTPUT_FORMAT"].as<std::string>();
    std::transform(output_format.begin(), output_format.end(), output_format.begin(), ::tolower);
    if (output_format != "csv" && output_format != "c81") {
        std::cerr << "Error: OUTPUT_FORMAT must be 'csv' or 'c81'.\n";
        return 1;
    }

    // Validate geometry inputs
    bool hasFile   = vm.count("GEOMETRY");
    bool hasFolder = vm.count("GEOMETRY_FOLDER");
    if (hasFile && hasFolder) {
        std::cerr << "Error: GEOMETRY and GEOMETRY_FOLDER are mutually exclusive.\n";
        return 1;
    }
    if (!hasFile && !hasFolder) {
        std::cerr << "Error: Either GEOMETRY or GEOMETRY_FOLDER must be specified.\n";
        return 1;
    }

    // Gather geometry files
    std::vector<std::string> geo_files;
    if (hasFolder) {
        for (auto& p : fs::directory_iterator(vm["GEOMETRY_FOLDER"].as<std::string>()))
            if (p.is_regular_file() && p.path().extension() == ".dat")
                geo_files.push_back(p.path().string());
        if (geo_files.empty()) {
            std::cerr << "Error: No .dat files found in folder " << vm["GEOMETRY_FOLDER"].as<std::string>() << "\n";
            return 1;
        }
    } else {
        geo_files.push_back(vm["GEOMETRY"].as<std::string>());
    }

    size_t num_geoms = geo_files.size();

    std::string output_folder = vm["OUTPUT_FOLDER"].as<std::string>();
    fs::create_directories(output_folder);

    // Common config parameters
    std::string out_prefix = vm["OUTPUT_FILE"].as<std::string>();
    double aoa_min  = vm["AOA_MIN"].as<double>();
    double aoa_max  = vm["AOA_MAX"].as<double>();
    double aoa_step = vm["AOA_STEP"].as<double>();
    double mach_min = vm["MACH_MIN"].as<double>();
    double mach_max = vm["MACH_MAX"].as<double>();
    double mach_step= vm["MACH_STEP"].as<double>();
    double re_min   = vm["REYNOLDS_MIN"].as<double>();
    double re_max   = vm["REYNOLDS_MAX"].as<double>();
    double re_step  = vm["REYNOLDS_STEP"].as<double>();
    double ci_level = vm["CI_LEVEL"].as<double>();
    if (ci_level <= 0.0 || ci_level >= 1.0) {
        std::cerr << "Error: CI_LEVEL must be in (0,1).\n";
        return 1;
    }
    auto t1_init = Clock::now();

    // Progress bar setup
    size_t num_aoa  = static_cast<size_t>(std::floor((aoa_max - aoa_min)/aoa_step + 1 + 1e-8));
    size_t num_mach = static_cast<size_t>(std::floor((mach_max - mach_min)/mach_step + 1 + 1e-8));
    size_t num_re   = static_cast<size_t>(std::floor((re_max   - re_min)  /re_step  + 1 + 1e-8));
    size_t preds_per_geom = num_aoa * num_mach * num_re;
    size_t total_preds    = preds_per_geom * num_geoms;
    size_t done_preds     = 0;
    constexpr int BAR_WIDTH = 50;

    auto print_progress = [&](size_t done) {
        float ratio = float(done) / float(total_preds);
        int   pos   = int(ratio * BAR_WIDTH);
        std::cout << "[";
        for (int i = 0; i < BAR_WIDTH; ++i) std::cout << (i < pos ? '=' : ' ');
        std::cout << "] "
                  << std::setw(3) << int(ratio * 100) << "% "
                  << "(" << done << "/" << total_preds << ")\r"
                  << std::flush;
    };

    print_progress(done_preds);

    // Load models & PCAs
    auto t0_load = Clock::now();
    int E = ENSEMBLE_SIZE;
    std::vector<torch::jit::script::Module> models; models.reserve(E);
    std::vector<PcaTransform> pcas;      pcas.reserve(E);
    for (int e = 0; e < E; ++e) {
        const auto* pca_ptr = ensemble_members[e].pca_data;
        size_t pca_len      = ensemble_members[e].pca_len;
        std::string pca_str(reinterpret_cast<const char*>(pca_ptr), pca_len);
        std::istringstream pca_stream(pca_str, std::ios::binary);
        auto [mean, comps, expl] = load_pca_dat(pca_stream);
        pcas.push_back({std::move(mean), std::move(comps), std::move(expl)});

        const auto* mdl_ptr = ensemble_members[e].model_data;
        size_t mdl_len      = ensemble_members[e].model_len;
        std::string mdl_str(reinterpret_cast<const char*>(mdl_ptr), mdl_len);
        std::istringstream mdl_stream(mdl_str, std::ios::binary);
        torch::jit::script::Module model = torch::jit::load(mdl_stream, device);
        model.to(device); model.eval();
        models.emplace_back(std::move(model));
    }
    auto t1_load = Clock::now();

    double total_prep_ms = 0.0, total_infer_ms = 0.0, total_write_ms = 0.0;

    // Loop over geometries
    for (const auto& geo_file : geo_files) {
        // PREP
        auto t0_prep = Clock::now();
        fs::path base = hasFile
            ? fs::path(output_folder) / out_prefix
            : fs::path(output_folder) / (out_prefix + "_" + fs::path(geo_file).stem().string());
        std::string out_base = base.string();

        auto [x_int, y_int] = import_and_interpolate_airfoil(geo_file, 257, 2.5, 1000);
        std::vector<double> coords; coords.reserve(x_int.size()*2);
        for (size_t i = 0; i < x_int.size(); ++i) coords.push_back(x_int[i]), coords.push_back(y_int[i]);

        struct Combo { double aoa, mach, re; };
        std::vector<Combo> combos;
        for (double a = aoa_min; a <= aoa_max + 1e-8; a += aoa_step)
            for (double m = mach_min; m <= mach_max + 1e-8; m += mach_step)
                for (double r = re_min; r <= re_max + 1e-8; r += re_step)
                    combos.push_back({a, m, r});
        if (!preds_per_geom) preds_per_geom = combos.size();
        total_prep_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0_prep).count();

        // INFERENCE
        auto t0_infer = Clock::now();
        std::vector<torch::Tensor> outs; outs.reserve(E);
        for (int e = 0; e < E; ++e) {
            auto pc = apply_pca(pcas[e].mean, pcas[e].comps, pcas[e].expl, coords);
            size_t dim = 3 + pc.size();
            std::vector<float> data; data.reserve(combos.size()*dim);
            for (auto& c : combos) {
                data.push_back(c.aoa/180.0f);
                data.push_back(c.mach);
                data.push_back(c.re/30e6f);
                for (auto v : pc) data.push_back(static_cast<float>(v));
            }
            auto tensor = torch::from_blob(data.data(), {(long)combos.size(), (long)dim}, torch::kFloat).to(device);
            torch::NoGradGuard ng;
            outs.push_back(models[e].forward({tensor}).toTensor().cpu());
        }
        total_infer_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0_infer).count();

        // WRITE
        auto t0_write = Clock::now();
        auto stacked = torch::stack(outs);
        auto mean_out = torch::mean(stacked, 0);
        auto std_out  = torch::std(stacked, 0);
        auto se_out   = std_out / std::sqrt((double)E);
        boost::math::normal dist(0,1);
        double alpha = 1.0 - ci_level;
        double z     = boost::math::quantile(dist, 1.0 - alpha/2.0);
        auto margin  = se_out * z;

        auto [feat_order, out_means, out_scales] = load_scaler_params();

        if (output_format == "csv") {
            std::ofstream ofs(out_base + ".csv");
            ofs << std::fixed << std::setprecision(6) << "AOA,MACH,REYNOLDS";
            for (auto& k : feat_order) ofs << "," << k << "," << k << "_CI";
            ofs << "\n";
            for (size_t i = 0; i < combos.size(); ++i) {
                ofs << combos[i].aoa << "," << combos[i].mach << "," << combos[i].re;
                for (size_t j = 0; j < feat_order.size(); ++j) {
                    double raw  = mean_out[i][j].item<float>();
                    double mean = raw*out_scales.at(feat_order[j]) + out_means.at(feat_order[j]);
                    double ci   = margin[i][j].item<float>()*out_scales.at(feat_order[j]);
                    ofs << "," << mean << "," << ci;
                }
                ofs << "\n";
            }
        } else {
            std::vector<double> aoas, machs, ress;
            for (double a = aoa_min; a <= aoa_max + 1e-8; a += aoa_step) aoas.push_back(a);
            for (double m = mach_min; m <= mach_max + 1e-8; m += mach_step) machs.push_back(m);
            for (double r = re_min; r <= re_max + 1e-8; r += re_step) ress.push_back(r);
            int A = aoas.size(), M = machs.size(), R = ress.size(), F = feat_order.size();

            std::ofstream ofs(out_base + ".c81");
            ofs << R << "\n";
            for (int ri = 0; ri < R; ++ri) {
                ofs << "COMMENT#" << ri+1 << "\n";
                ofs << ress[ri] << "\n";
                ofs << fs::path(geo_file).filename().stem().string() << " " << M << " " << A << " " << M << " " << A<< " " << M << " " << A << "\n";
                for (int fi = 0; fi < F; ++fi) {
                    for (double m : machs) ofs << " " << m;
                    ofs << "\n";
                    for (int ai = 0; ai < A; ++ai) {
                        ofs << aoas[ai];
                        for (int mi = 0; mi < M; ++mi) {
                            int idx = ai*(M*R) + mi*R + ri;
                            double raw  = mean_out[idx][fi].item<float>();
                            double mean = raw*out_scales.at(feat_order[fi]) + out_means.at(feat_order[fi]);
                            ofs << " " << mean;
                        }
                        ofs << "\n";
                    }
                }
            }
        }
        total_write_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0_write).count();

        // update progress
        done_preds += preds_per_geom;
        print_progress(done_preds);
        if (done_preds == total_preds) std::cout << std::endl;
    }

    std::cout << " \n";
    std::cout << "[ASCII]:       " << std::chrono::duration<double, std::milli>(t1_ascii - t0_ascii).count() << " ms\n";
    std::cout << "[INIT]:        " << std::chrono::duration<double, std::milli>(t1_init - t0_init).count() << " ms\n";
    std::cout << "[MODEL LOAD]:  " << std::chrono::duration<double, std::milli>(t1_load - t0_load).count() << " ms\n";
    std::cout << "[PREP]:        " << total_prep_ms    << " ms\n";
    std::cout << "[INFERENCE]:   " << total_infer_ms  << " ms\n";
    std::cout << "[WRITE]:       " << total_write_ms  << " ms\n";
    std::cout << "[TOTAL]:       " << std::chrono::duration<double, std::milli>(Clock::now()-t0_all).count() << " ms\n\n";

    if (hasFile) {
        std::cout << preds_per_geom << " predictions written to file: " << output_folder << "/" << out_prefix << "\n";
    } else {
        size_t total_preds_final = preds_per_geom * num_geoms;
        std::cout << num_geoms << " geometries\n"
                  << preds_per_geom << " predictions per geometry\n"
                  << total_preds_final << " predictions written to folder: " << output_folder << "\n";
    }
    return 0;
}

