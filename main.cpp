#include <iostream>
#include <memory>
#include <boost/program_options.hpp>

#include "Image.h"
#include "PatchMatchAlg.h"
#include "patchmatch_gpu.h"
#include "helper_timer.h"


using namespace cv;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {

    po::options_description generic("Generic options");
    generic.add_options()
        ("version,v", "print version string")
        ("help,h", "produce help message")
        ("prj_file", po::value<std::string>(), "project file path");
    po::options_description prj_file_option("Project File");
    prj_file_option.add_options()
        ("help,h", "produce help message")
        ("left_img", po::value<std::string>(), "left image file path")
        ("right_img", po::value<std::string>(), "right image file path")
        ("max_disp", po::value<int>(), "max disparity")
        ("window_radius", po::value<int>(), "window radius used in aggregated cost")
        ("gamma", po::value<float>(), "gamma value used in weight computation")
        ("alpha", po::value<float>(), "alpha value used in cost computation")
        ("trunc_col", po::value<float>(), "max difference of intensity value")
        ("trunc_grad", po::value<float>(), "max difference of gradiance value")
        ("result_dir", po::value<std::string>(), "result image dir path")
        ("iter_count", po::value<int>(), "iteration count of propagation")
        ("use_gpu", po::value<bool>(), "use gpu to speed up the process");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, generic), vm);
    po::notify(vm);

    if (vm.count("help") || vm.empty()) {
        std::cout << generic << "\n";
        return 1;
    }

    auto imgL = std::make_shared<Image>();
    auto imgR = std::make_shared<Image>();
    PatchMatchAlg patch_match_alg;
    PatchMatchConfig cfg;
    bool use_gpu = false;
    if (vm.count("prj_file")) {
        const std::string prj_file = vm["prj_file"].as<std::string>();        

        std::ifstream ifs(prj_file);
        if (ifs) {
            vm.clear();
            store(po::parse_config_file(ifs, prj_file_option), vm);
            notify(vm);
            if (vm.count("help") || vm.empty()) {
                std::cout << prj_file_option << "\n";
                return 1;
            }

            if (vm.count("left_img")) {
                imgL->load(vm["left_img"].as<std::string>());
                cfg.rows = imgL->rows_;
                cfg.cols = imgL->cols_;
            } else {
                std::cout << "there is no left img" << std::endl;
                return 1;
            }

            if (vm.count("right_img")) {
                imgR->load(vm["right_img"].as<std::string>());
            }
            else {
                std::cout << "there is no right img" << std::endl;
                return 1;
            }

            if (vm.count("max_disp")) {
                patch_match_alg.max_disparity_ = vm["max_disp"].as<int>();
                cfg.max_disp = patch_match_alg.max_disparity_;
            }
            else {
                std::cout << "there is no max_disp" << std::endl;
                return 1;
            }

            if (vm.count("window_radius")) {
                patch_match_alg.window_radius_ = vm["window_radius"].as<int>();
                cfg.window_radius = patch_match_alg.window_radius_;
            }
            else {
                std::cout << "there is no window_radius" << std::endl;
                return 1;
            }

            if (vm.count("gamma")) {
                patch_match_alg.gamma_ = vm["gamma"].as<float>();
                cfg.gamma = patch_match_alg.gamma_;
            }
            else {
                std::cout << "there is no gamma" << std::endl;
                return 1;
            }

            if (vm.count("alpha")) {
                patch_match_alg.alpha_ = vm["alpha"].as<float>();
                cfg.alpha = patch_match_alg.alpha_;
            }
            else {
                std::cout << "there is no alpha" << std::endl;
                return 1;
            }

            if (vm.count("trunc_col")) {
                patch_match_alg.trunc_col_ = vm["trunc_col"].as<float>();
                cfg.density_diff_max = patch_match_alg.trunc_col_;
            }
            else {
                std::cout << "there is no trunc_col" << std::endl;
                return 1;
            }

            if (vm.count("trunc_grad")) {
                patch_match_alg.trunc_grad_ = vm["trunc_grad"].as<float>();
                cfg.grad_diff_max = patch_match_alg.trunc_grad_;
            }
            else {
                std::cout << "there is no trunc_grad" << std::endl;
                return 1;
            }

            if (vm.count("use_gpu")) {
                use_gpu = vm["use_gpu"].as<bool>();
            }
            else {
                std::cout << "there is no use_gpu" << std::endl;
                return 1;
            }

            if (vm.count("iter_count")) {
                cfg.iter_count = vm["iter_count"].as<int>();
            }
            else {
                std::cout << "there is no iter_count" << std::endl;
                return 1;
            }
        }

        if (use_gpu) {
            cfg.max_cost_single = (1.0f - cfg.alpha) * cfg.density_diff_max + cfg.alpha * cfg.grad_diff_max;
            cfg.neighbor_lists_len = 20;
            cfg.h_neighbor_lists = new int2[cfg.neighbor_lists_len];
            //up + 1
            cfg.h_neighbor_lists[0] = make_int2(0, -1);
            //up + 3
            cfg.h_neighbor_lists[1] = make_int2(0, -3);
            //up + 5
            cfg.h_neighbor_lists[2] = make_int2(0, -5);
            //down + 1
            cfg.h_neighbor_lists[3] = make_int2(0, 1);
            //down + 3
            cfg.h_neighbor_lists[4] = make_int2(0, 3);
            //down + 5
            cfg.h_neighbor_lists[5] = make_int2(0, 5);
            //left + 1
            cfg.h_neighbor_lists[6] = make_int2(-1, 0);
            //left + 3
            cfg.h_neighbor_lists[7] = make_int2(-3, 0);
            //left + 5
            cfg.h_neighbor_lists[8] = make_int2(-5, 0);
            //right + 1
            cfg.h_neighbor_lists[9] = make_int2(1, 0);
            //right + 3
            cfg.h_neighbor_lists[10] = make_int2(3, 0);
            //right + 5
            cfg.h_neighbor_lists[11] = make_int2(3, 0);
            //up left 1
            cfg.h_neighbor_lists[12] = make_int2(-1, -2);
            //up left 2
            cfg.h_neighbor_lists[13] = make_int2(-2, -1);
            //up right 1
            cfg.h_neighbor_lists[14] = make_int2(1, -2);
            //up right 2
            cfg.h_neighbor_lists[15] = make_int2(2, -1);
            //down left 1
            cfg.h_neighbor_lists[16] = make_int2(-1, 2);
            //down left 2
            cfg.h_neighbor_lists[17] = make_int2(-2, 1);
            //down right 1
            cfg.h_neighbor_lists[18] = make_int2(1, 2);
            //down right 2
            cfg.h_neighbor_lists[19] = make_int2(2, 1);

            StopWatchWin timer;

            timer.start();
            init(imgL.get(), cfg);
            std::cout << timer.getTime() << std::endl;
            timer.reset();
            timer.start();
            init(imgR.get(), cfg);
            std::cout << timer.getTime() << std::endl;
            timer.reset();
            timer.start();
            solve(imgL.get(), imgR.get(), cfg);
            std::cout << timer.getTime() << std::endl;

            cudaFree(imgL->d_image_);
            cudaFree(imgL->d_cost_);
            cudaFree(imgL->d_grad_);
            cudaFree(imgL->d_normal_);
            cudaFree(imgL->d_plane_);
            cudaFree(imgR->d_image_);
            cudaFree(imgR->d_cost_);
            cudaFree(imgR->d_grad_);
            cudaFree(imgR->d_normal_);
            cudaFree(imgR->d_plane_);
        } else {
            patch_match_alg.solve(imgL, imgR);

            if (vm.count("result_dir")) {
                const std::string result_dir = vm["result_dir"].as<std::string>();
                patch_match_alg.write_result(result_dir);
            }
            else {
                std::cout << "there is no trunc_grad" << std::endl;
                return 1;
            }
        }

        
    }

    return 0;
}