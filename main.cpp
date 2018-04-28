#include <iostream>
#include <memory>
#include <boost/program_options.hpp>

#include "Image.h"
#include "PatchMatchAlg.h"


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
        ("result_dir", po::value<std::string>(), "result image dir path");
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
            }
            else {
                std::cout << "there is no max_disp" << std::endl;
                return 1;
            }

            if (vm.count("window_radius")) {
                patch_match_alg.window_radius_ = vm["window_radius"].as<int>();
            }
            else {
                std::cout << "there is no window_radius" << std::endl;
                return 1;
            }

            if (vm.count("gamma")) {
                patch_match_alg.gamma_ = vm["gamma"].as<float>();
            }
            else {
                std::cout << "there is no gamma" << std::endl;
                return 1;
            }

            if (vm.count("alpha")) {
                patch_match_alg.alpha_ = vm["alpha"].as<float>();
            }
            else {
                std::cout << "there is no alpha" << std::endl;
                return 1;
            }

            if (vm.count("trunc_col")) {
                patch_match_alg.trunc_col_ = vm["trunc_col"].as<float>();
            }
            else {
                std::cout << "there is no trunc_col" << std::endl;
                return 1;
            }

            if (vm.count("trunc_grad")) {
                patch_match_alg.trunc_grad_ = vm["trunc_grad"].as<float>();
            }
            else {
                std::cout << "there is no trunc_grad" << std::endl;
                return 1;
            }
        }

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

    return 0;
}