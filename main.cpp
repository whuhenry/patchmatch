#include <iostream>
#include <memory>
#include <ctime>

#include "Image.h"
#include "PatchMatchAlg.h"

using namespace cv;

int main(int argc, char* argv[]) {

	auto imgL = std::make_shared<Image>();
	auto imgR = std::make_shared<Image>();
#ifdef _WIN64
	imgL->load(R"(F:\Data\Benchmark\teddy\im2.png)");
	imgR->load(R"(F:\Data\Benchmark\teddy\im6.png)");
#else
	imgL->load(R"(/home/henry/project/data/teddy/im2.png)");
	imgR->load(R"(/home/henry/project/data/teddy/im6.png)");
#endif
	const clock_t begin_time = clock();
	PatchMatchAlg patch_match_alg;
    patch_match_alg.max_disparity_ = 64;
	patch_match_alg.solve(imgL, imgR);
	
    patch_match_alg.save_disp_map(R"(F:\1.ply)");

    return 0;
}