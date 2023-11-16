#pragma once

#include <atomic>
#include <climits>
#include <functional>
#include <future>
#include <list>
#include <queue>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <condition_variable>

#include "util/settings.h"

namespace dso
{
	class ThreadPool
	{
	public:
		static const int kMaxNumThreads = -1;

		explicit ThreadPool(const int num_threads = kMaxNumThreads);
		~ThreadPool();

		inline size_t NumThreads() const;

		template<class func_t, class... args_t>
		auto AddTask(func_t&& f, args_t&&... args)
			->std::future<typename std::result_of<func_t(args_t...)>::type>
		{
			typedef typename std::result_of<func_t(args_t...)>::type return_t;

			auto task = std::make_shared<std::packaged_task<return_t()>>(
				std::bind(std::forward<func_t>(f), std::forward<args_t>(args)...));

			std::future<return_t> result = task->get_future();

			{
				std::unique_lock<std::mutex> lock(mutex_);
				if (stopped_) throw std::runtime_error("Cannot add task to stopped thread pool.");
				tasks_.emplace([task]() {(*task)(); });
			}

			task_condition_.notify_one();

			return result;
		}

		void Stop();

		void Wait();

		std::thread::id GetThreadId() const;

		int GetThreadIndex();

	private:
		void WorkerFunc(const int index);

		std::vector<std::thread> workers_;
		std::queue<std::function<void()>> tasks_;

		std::mutex mutex_;
		std::condition_variable task_condition_;
		std::condition_variable finished_condition_;

		bool stopped_;
		int num_active_workers_;

		std::unordered_map<std::thread::id, int> thread_id_to_index_;
	};
}