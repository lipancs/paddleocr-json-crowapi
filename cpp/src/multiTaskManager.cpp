#include "include/multiTaskManager.h"
#include <iostream>

namespace PaddleOCR
{
    multiTaskManager::multiTaskManager(size_t num_instances) : stop(false) {
        // **������� task ʵ��**
       
        for (size_t i = 0; i < num_instances; ++i) {
            auto task_instance = std::make_unique<Task>();
            task_instance->init_engines();  // **ȷ�� PPOCR ��ʼ��**
            tasks_instances.push_back(std::move(task_instance));
            task_ids.push_back(i);// **�� Task �� ID**
        }

        // ���� worker �̣߳�ÿ���̰߳�һ�� Task
        for (size_t i = 0; i < num_instances; ++i) {
            workers.emplace_back(&multiTaskManager::worker_thread, this,i);
        }
    }
    // **��������**
    multiTaskManager::~multiTaskManager() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();

        // �ȴ����й����߳��˳�
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // **�ύ OCR ����**
    std::future<std::string> multiTaskManager::submit_task(const cv::Mat& img) {
        auto promise = std::make_shared<std::promise<std::string>>();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.emplace(TaskItem{ img, promise });
        }
        condition.notify_one();  // ����һ�������߳�
        return promise->get_future();
    }

    // **�����߳�**
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

            // ֻ�õ�ǰ�̵߳� Task ��������
            std::string result_json = tasks_instances[thread_id]->ocr_uploadimg_mode(task.img);
            // **��� Task ID���������ĸ�ʵ�������**
            std::cout << "Task Instance " << task_ids[thread_id]
                << " (Thread " << thread_id << ") processed OCR result: "
                    << result_json << std::endl;
                // **ͨ�� promise ���� OCR ���**
                task.promise->set_value(result_json);
        }
    }

}