#include "util/IndexThreadReduceNB.h"

/// <summary>
/// 测试所实现的线程池的加速效果
/// </summary>
/// <param name=""></param>
void test_thread_pool(void)
{
	int NUM_TASKS = 100;
	dso::ThreadPool threadPool(NUM_THREADS);
	std::vector<std::future<void>> futures;
	futures.resize(NUM_TASKS);

	clock_t ts_multi_thread = clock();
	for (int it = 0; it < NUM_TASKS; ++it)
		futures[it] = threadPool.AddTask(
			[]() {std::this_thread::sleep_for(std::chrono::milliseconds(1)); });
	
	threadPool.Wait();
	clock_t te_multi_thread = clock();

	clock_t ts_single_thread = clock();
	for (int it = 0; it < NUM_TASKS; ++it)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	clock_t te_single_thread = clock();

	printf("multi thread time %dms\n", te_multi_thread - ts_multi_thread);
	printf("single thread time %dms\n", te_single_thread - ts_single_thread);
}

int main(int argc, char* argv)
{
	test_thread_pool();

	return 0;
}  