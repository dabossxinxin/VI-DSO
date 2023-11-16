#include "util/threading.h"

namespace dso
{
	inline int GetEffectiveNumThreads(const int num_threads)
	{
		int num_effective_threads = num_threads;
		if (num_threads < 0)
			num_effective_threads = std::thread::hardware_concurrency();

		if (num_effective_threads <= 0)
			num_effective_threads = 1;

		return num_effective_threads;
	}

	/// <summary>
	/// 线程池初始化工作线程
	/// </summary>
	/// <param name="num_threads">工作线程数量</param>
	ThreadPool::ThreadPool(const int num_threads)
		:stopped_(false), num_active_workers_(0)
	{
		const int num_effective_threads = GetEffectiveNumThreads(num_threads);
		for (int it = 0; it < num_effective_threads; ++it)
		{
			std::function<void(void)> worker =
				std::bind(&ThreadPool::WorkerFunc, this, it);
			workers_.emplace_back(worker);
		}
	}

	ThreadPool::~ThreadPool()
	{
		Stop();
	}

	void ThreadPool::Stop()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		if (stopped_) return;
		stopped_ = true;
		std::queue<std::function<void()>> empty_tasks;
		std::swap(tasks_, empty_tasks);
		lock.unlock();

		task_condition_.notify_all();
		for (auto& worker : workers_)
			worker.join();
		finished_condition_.notify_all();
	}

	void ThreadPool::Wait()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		if (!tasks_.empty() || num_active_workers_ > 0)
		{
			finished_condition_.wait(lock, [this]() {
				return tasks_.empty() && num_active_workers_;
				});
		}
	}

	/// <summary>
	/// 工作线程工作函数
	/// </summary>
	/// <param name="index">工作线程索引</param>
	void ThreadPool::WorkerFunc(const int index)
	{
		{
			std::unique_lock<std::mutex> lock(mutex_);
			thread_id_to_index_.emplace(GetThreadId(), index);
		}

		while (true)
		{
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> lock(mutex_);
				task_condition_.wait(lock, [this]() {return stopped_ || !tasks_.empty(); });
				if (stopped_ && tasks_.empty()) return;
				task = std::move(tasks_.front());
				tasks_.pop();
				num_active_workers_ += 1;
			}

			task();

			{
				std::unique_lock<std::mutex> lock(mutex_);
				num_active_workers_ -= 1;
			}

			finished_condition_.notify_all();
		}
	}

	std::thread::id ThreadPool::GetThreadId() const
	{
		return std::this_thread::get_id();
	}

	size_t ThreadPool::NumThreads() const
	{
		return workers_.size();
	}

	int ThreadPool::GetThreadIndex()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		return thread_id_to_index_.at(GetThreadId());
	}
}