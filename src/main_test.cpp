#include "util/threading.h"
#include "util/IndexThreadReduce.h"

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

/// <summary>
/// 测试所实现的线程池的加速效果
/// </summary>
/// <param name=""></param>
void test_std_thread_pool(void)
{
	int NUM_TASKS = 100;
	dso::ThreadPool threadPool(NUM_THREADS);
	std::vector<std::future<void>> futures;
	futures.resize(NUM_TASKS);

	clock_t ts_multi_thread = clock();
	for (int it = 0; it < NUM_TASKS; ++it)
		futures[it] = threadPool.AddTask(
			[]() {std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
	
	threadPool.Wait();
	clock_t te_multi_thread = clock();

	clock_t ts_single_thread = clock();
	for (int it = 0; it < NUM_TASKS; ++it)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	clock_t te_single_thread = clock();

	printf("multi thread time %dms\n", te_multi_thread - ts_multi_thread);
	printf("single thread time %dms\n", te_single_thread - ts_single_thread);
}

int main(int argc, char* argv)
{
	test_std_thread_pool();

	test_boost_thread_pool();

	return 0;
}  