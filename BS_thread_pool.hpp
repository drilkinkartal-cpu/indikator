// Minimal BS::thread_pool - yeterli submit/future API'si sağlayan basit implementasyon
// Bu dosya, indicator.cpp içindeki `BS::thread_pool pool(n); pool.submit(...)` çağrılarını
// destekleyecek şekilde yazılmıştır. Orijinal kütüphanenin tüm özelliklerini kapsamaz.

#ifndef BS_THREAD_POOL_HPP
#define BS_THREAD_POOL_HPP

#include <vector>
#include <thread>
#include <queue>
#include <future>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace BS {

class thread_pool {
public:
    explicit thread_pool(size_t n_threads = std::thread::hardware_concurrency()) : stop_flag(false) {
        if (n_threads == 0) n_threads = 1;
        for (size_t i = 0; i < n_threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop_flag || !this->tasks.empty(); });
                        if (this->stop_flag && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~thread_pool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop_flag = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    // submit: callable + args -> future<return_type>
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        using return_type = typename std::invoke_result_t<F, Args...>;

        auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task_ptr->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // Eğer pool yoksa exception yerine std::future exception üretmek tercih edilebilir;
            // burada basitçe task'ı kuyruğa ekliyoruz.
            tasks.emplace([task_ptr]() { (*task_ptr)(); });
        }
        condition.notify_one();
        return res;
    }

    // Thread sayısını döndür
    size_t size() const noexcept { return workers.size(); }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop_flag;
};

} // namespace BS

#endif // BS_THREAD_POOL_HPP
