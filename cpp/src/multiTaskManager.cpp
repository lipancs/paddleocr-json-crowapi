#include "include/multiTaskManager.h"
#include <iostream>

namespace PaddleOCR
{
    multiTaskManager::multiTaskManager(size_t num_instances) : stop(false) {
        // **创建多个 task 实例**
       
        for (size_t i = 0; i < num_instances; ++i) {
            auto task_instance = std::make_unique<Task>();
            task_instance->init_engines();  // **确保 PPOCR 初始化**
            tasks_instances.push_back(std::move(task_instance));
            task_ids.push_back(i);// **给 Task 赋 ID**
        }

        // 启动 worker 线程，每个线程绑定一个 Task
        for (size_t i = 0; i < num_instances; ++i) {
            workers.emplace_back(&multiTaskManager::worker_thread, this,i);
        }
    }
    // **析构函数**
    multiTaskManager::~multiTaskManager() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();

        // 等待所有工作线程退出
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // **提交 OCR 任务**
    std::future<std::string> multiTaskManager::submit_task(const cv::Mat& img) {
        auto promise = std::make_shared<std::promise<std::string>>();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.emplace(TaskItem{ img, promise });
        }
        condition.notify_one();  // 唤醒一个工作线程
        return promise->get_future();
    }

    // **工作线程**
    void multiTaskManager::worker_thread(size_t thread_id) {
        while (true) {
            TaskItem task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                condition.wait(lock, [this] { return !task_queue.empty() || stop; });

                if (stop && task_queue.empty()) return;

                task = task_queue.front();
                task_queue.pop();
            }

            // 只让当前线程的 Task 处理任务
            std::string result_json = tasks_instances[thread_id]->ocr_uploadimg_mode(task.img);
            // **输出 Task ID，告诉是哪个实例处理的**
            std::cout << "Task Instance " << task_ids[thread_id]
                << " (Thread " << thread_id << ") processed OCR result: "
                    << result_json << std::endl;
                // **通过 promise 设置 OCR 结果**
                task.promise->set_value(result_json);
        }
    }

}