#ifndef MULTITASKMANAGER_H
#define MULTITASKMANAGER_H
#include <opencv2/opencv.hpp> // ��� OpenCV ͷ�ļ�
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "include/paddleocr.h"
#include "include/task.h"
#include <future>
#include <memory>
// ����Ű��� task.h������δ�������
namespace PaddleOCR
{
class multiTaskManager {
public:
    explicit multiTaskManager(size_t num_instances);
    ~multiTaskManager();

    // ����������ύ OCR ����
    std::future<std::string> submit_task(const cv::Mat& img);

private:
    struct TaskItem {
        cv::Mat img;
        std::shared_ptr<std::promise<std::string>> promise;
    };
    void worker_thread(size_t thread_id); // �̴߳�����
    std::vector<std::unique_ptr<PaddleOCR::Task>> tasks_instances;    // ��� task ʵ��
    std::queue<TaskItem> task_queue;  // �������
    std::vector<std::thread> workers;  // �����̳߳�
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    std::vector<size_t> task_ids;
};
}
#endif // MULTITASKMANAGER_H