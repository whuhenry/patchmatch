#include <iostream>
#include <memory>
#include <ctime>

#include "image.h"
#include "PatchMatchAlg.h"

using namespace cv;

int main(int argc, char* argv[]) {

	auto imgL = std::make_shared<Image>();
	auto imgR = std::make_shared<Image>();
#ifdef _WIN64
	imgL.load("F:\\Data\\Benchmark\\cones\\im2.png");
	imgR.load("F:\\Data\\Benchmark\\cones\\im6.png");
#else
	imgL->load(R"(/mnt/f/Data/Benchmark/teddy/im2.png)");
	imgR->load(R"(/mnt/f/Data/Benchmark/teddy/im6.png)");
#endif
	const clock_t begin_time = clock();
	PatchMatchAlg patch_match_alg;
	patch_match_alg.solve(imgL, imgR);
	std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    return 0;
}