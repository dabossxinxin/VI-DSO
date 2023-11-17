#include "util/IndexThreadReduce.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void SUM(int first, int end, int* status, int tid)
{
	for (int it = first; it < end; ++it)
	{
		printf("thread %d,num %d\n", std::this_thread::get_id(), it);
		*status += it;
	}
};

void test_boost_thread_pool()
{
	dso::IndexThreadReduce<int> *reduce =
		new dso::IndexThreadReduce<int>();
	
	clock_t ts_multi_thread = clock();
	reduce->reduce(std::bind(SUM, std::placeholders::_1, std::placeholders::_2,
		std::placeholders::_3, std::placeholders::_4), 0, 1000, 50);
	clock_t te_multi_thread = clock();
	
	int result = 0;
	clock_t ts_single_thread = clock();
	SUM(0, 1000, &result, 0);
	clock_t te_single_thread = clock();

	printf("multi thread time %d, result %d\n", te_multi_thread - ts_multi_thread, reduce->stats);
	printf("single thread time %d, result %d\n", te_single_thread - ts_single_thread, result);
}

int main(int argc, char** argv)
{
	std::cout << std::atan2(1.0000, 0.0000) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.9239, 0.3827) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.9808, 0.1951) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.3827, 0.9239) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.7071, 0.7071) / M_PI * 180 << std::endl;
	std::cout << std::atan2(-0.9239, 0.3827) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.5556, 0.8315) / M_PI * 180 << std::endl;
	std::cout << std::atan2(-0.5556, 0.8315) / M_PI * 180 << std::endl;
	std::cout << std::atan2(-0.8315, 0.5556) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.1951, 0.9808) / M_PI * 180 << std::endl;
	std::cout << std::atan2(-0.3827, 0.9239) / M_PI * 180 << std::endl;
	std::cout << std::atan2(-0.7071, 0.7071) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.8315, 0.5556) / M_PI * 180 << std::endl;
	std::cout << std::atan2(-0.1951, 0.9808) / M_PI * 180 << std::endl;
	std::cout << std::atan2(0.0000, 1.0000) / M_PI * 180 << std::endl;
	std::cout << std::atan2(-0.9808, 0.1951) / M_PI * 180 << std::endl;

	//test_std_thread_pool();
	//test_boost_thread_pool(); 

	std::string path = "C:\\Users\\Admin\\Desktop\\CAM1.jpg";
	cv::Mat cam1 = cv::imread(path);
	path = "C:\\Users\\Admin\\Desktop\\CAM3.jpg";
	cv::Mat cam3 = cv::imread(path);

	cv::Mat cam1Tmp = cam1(cv::Rect(1651, 1892, 752, 480)).clone();
	cv::Mat cam3Tmp = cam3(cv::Rect(1980, 2281, 752, 480)).clone();

	path = "C:\\Users\\Admin\\Desktop\\CAM1TMP.jpg";
	cv::imwrite(path, cam1Tmp);
	path = "C:\\Users\\Admin\\Desktop\\CAM3TMP.jpg";
	cv::imwrite(path, cam3Tmp);

	unsigned long long a = 49999872;
	unsigned long long b = 1403636579713555712;
	unsigned long long c = b-a;
	std::cout << c << std::endl;

	return 0;
}  