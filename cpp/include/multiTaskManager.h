#ifndef MULTITASKMANAGER_H
#define MULTITASKMANAGER_H
#include <opencv2/opencv.hpp> // 添加 OpenCV 头文件
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "include/paddleocr.h"
#include "include/task.h"
#include <future>
#include <memory>
// 这里才包含 task.h，避免未定义错误
namespace PaddleOCR
{
class multiTaskManager {
public:
    explicit multiTaskManager(size_t num_instances);
    ~multiTaskManager();

    // 向任务队列提交 OCR 任务
    std::future<std::string> submit_task(const cv::Mat& img);

private:
    struct TaskItem {
        cv::Mat img;
        std::shared_ptr<std::promise<std::string>> promise;
    };
    void worker_thread(size_t thread_id); // 线程处理函数
    std::vector<std::unique_ptr<PaddleOCR::Task>> tasks_instances;    // 多个 task 实例
    std::queue<TaskItem> task_queue;  // 任务队列
    std::vector<std::thread> workers;  // 工作线程池
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    std::vector<size_t> task_ids;
};
}
#endif // MULTITASKMANAGER_H