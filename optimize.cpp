// ================================================================================= //
//         ULTRA-FAST CMF OPTIMIZER - C++ VERSION (REFACTORED & OPTIMIZED)         //
// ================================================================================= //
// - I/O darboğazını ortadan kaldırmak için tüm ön-hesaplama verileri RAM'e alınmıştır. //
// - Ön-hesaplama adımları (CMF & Kanal Kombinasyonları) paralelleştirilmiştir.       //
// - Orijinal mantık ve yapı korunmuştur.                                            //
//                                                                                   //
// GEREKLİ KÜTÜPHANELER: (Değişiklik yok)                                             //
// - Eigen3, HighFive, HDF5, BS::thread_pool, cereal, fast-cpp-csv-parser, cxxopts   //
// ================================================================================= //

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <set>
#include <map>
#include <unordered_map>
#include <thread>
#include <filesystem>
#include <atomic>
#include <mutex>

// Kütüphane başlık dosyaları
#include <Eigen/Dense>
#include <highfive/H5File.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/optional.hpp>
#include "BS_thread_pool.hpp"
#include "csv.h" // fast-cpp-csv-parser
#include "cxxopts.hpp"

// Platforma özgü ayarlar
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

namespace fs = std::filesystem;

// ==================== CONFIGURATION ====================
const std::string CHECKPOINT_FILE = "checkpoint_ultra_fast.dat";
const std::string RESULTS_CSV = "optimization_best_results_cmf.csv";
const std::string BEST_SCORES_CSV = "optimization_all_best_scores_cmf.csv";

const std::string PRECOMP_DIR = "precomputed_data";
const std::string CMF_CACHE_FILE = fs::path(PRECOMP_DIR) / "cmf_cache.h5";
const std::string CHANNEL_COMBOS_FILE = fs::path(PRECOMP_DIR) / "channel_combos.h5";
const std::string METADATA_FILE = fs::path(PRECOMP_DIR) / "metadata.json";

constexpr double SCORE_THRESHOLD = 0.98;
constexpr int CHECKPOINT_INTERVAL = 120; // Saniye
constexpr int CSV_BATCH_SIZE = 100;

// Tip tanımlamaları
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using RowVector = Eigen::RowVectorXd;
using IntVector = Eigen::VectorXi;

// ==================== IN-MEMORY CACHE STRUCTURES (REFACTOR) ====================
// Bellekte tutulacak ön-hesaplama verileri için bir konteyner
struct PrecomputedDataCache {
    std::unordered_map<std::string, Vector> channel_scores;
    std::unordered_map<int, Vector> cmf_values;
};

// ==================== SHARED TYPES COPIED FROM indicator.cpp ====================
struct BacktestOutput {
    std::vector<double> trade_profits;
    std::vector<double> trade_profit_pcts;
    std::vector<long> trade_durations;
    double final_portfolio_value;
    int consecutive_wins;
    int consecutive_losses;
    int max_consecutive_wins;
    int max_consecutive_losses;
};

struct MarketData {
    Vector high, low, close, volume;
};

// Simple ResultRow & checkpoint structs (matching indicator.cpp layout)
struct ResultRow {
    std::vector<int> channels;
    int cmf_length;
    double cmf_buy_threshold;
    double cmf_sell_threshold;
    int buy_threshold;
    int sell_threshold;
    std::vector<double> metrics;

    template<class Archive>
    void serialize(Archive & ar) {
        ar(channels, cmf_length, cmf_buy_threshold, cmf_sell_threshold,
           buy_threshold, sell_threshold, metrics);
    }
};

struct CheckpointData {
    std::set<size_t> completed_hashes;
    double best_overall_score = -1.0;
    std::optional<ResultRow> best_overall_result;

    template<class Archive>
    void serialize(Archive & ar) {
        ar(completed_hashes, best_overall_score, best_overall_result);
    }
};

struct WorkerResult {
    std::optional<ResultRow> best_local_result;
    std::vector<ResultRow> good_results;
};

// Checkpoint save/load are provided by indicator.cpp; use those implementations to avoid duplicate symbols.


// Utility functions (generate_combinations, save_checkpoint, load_checkpoint) are defined in indicator.cpp;
// declare them here so the compiler knows their signatures and the linker will resolve them from indicator.o
void generate_combinations(const std::vector<int>& elements, int k, std::function<void(const std::vector<int>&)> callback);
void save_checkpoint(const CheckpointData& data);
CheckpointData load_checkpoint();

// ==================== CORE NUMERICAL FUNCTIONS (Değişiklik yok) ====================
// Bu fonksiyonlar zaten Eigen kullandığı için oldukça optimaller.
Vector calculate_cmf(const Vector& high, const Vector& low, const Vector& close, const Vector& volume, int length);
Vector calculate_pine_script_scores_correct(const Vector& prices, int length, double upper_mult, double lower_mult);
struct BacktestOutput;
BacktestOutput run_backtest_with_cmf(const Vector& scores, const Vector& cmf_values, const Vector& prices,
                                     double buy_threshold, double sell_threshold,
                                     double cmf_buy_threshold, double cmf_sell_threshold,
                                     double initial_capital);
std::vector<double> calculate_metrics_fast(const std::vector<double>& trade_profits, 
                                        const std::vector<double>& trade_profit_pcts, 
                                        const std::vector<long>& trade_durations,
                                        double final_value, double initial_capital,
                                        int max_consec_wins, int max_consec_losses);

// (Fonksiyonların gövdeleri çok uzun olduğu için buraya eklenmedi, orijinal kodla aynıdır.)
// ... CORE NUMERICAL FUNCTIONS' implementation from the original code ...


// ==================== PRE-COMPUTATION ENGINE (REFACTORED for PARALLELISM) ====================
class PreComputationEngine {
public:
    PreComputationEngine(const std::string& data_path) : data_path_(data_path) {}

    void load_data() { /* Orijinal ile aynı */ }

    void precompute_all_cmf(const std::vector<int>& cmf_lengths) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "📊 CMF ÖN-HESAPLAMA (PARALEL)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        HighFive::File file(CMF_CACHE_FILE, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        auto group = file.createGroup("cmf_values");
        
        BS::thread_pool pool;
        std::mutex h5_mutex;
        std::atomic<size_t> completed_count = 0;
        std::vector<std::future<void>> futures;

        for (int len : cmf_lengths) {
            futures.push_back(pool.submit([&, len]() {
                Vector cmf_array = calculate_cmf(market_data_.high, market_data_.low, market_data_.close, market_data_.volume, len);
                std::vector<double> cmf_vec(cmf_array.data(), cmf_array.data() + cmf_array.size());
                
                { // HDF5 yazma işlemi için kritik bölge
                    std::lock_guard<std::mutex> lock(h5_mutex);
                    group.createDataSet("length_" + std::to_string(len), cmf_vec);
                }
                
                size_t current_count = ++completed_count;
                if (current_count % 10 == 0) std::cout << "." << std::flush;
            }));
        }
        for (auto &f : futures) f.get();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        auto file_size_mb = fs::file_size(CMF_CACHE_FILE) / (1024.0 * 1024.0);
        std::cout << "\n✅ Tamamlandı! Süre: " << std::fixed << std::setprecision(1) << elapsed.count() << "s | Boyut: " << file_size_mb << " MB" << std::endl;
    }
    
    void precompute_all_channel_combos(const std::vector<int>& channel_search_space, int num_channels, double std_dev_mult) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "📊 KANAL KOMBİNASYONLARI ÖN-HESAPLAMA (PARALEL)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::vector<std::vector<int>> all_channel_combos;
        generate_combinations(channel_search_space, num_channels, [&](const std::vector<int>& combo) {
            all_channel_combos.push_back(combo);
        });
        std::cout << "🔢 Toplam kombinasyon: " << all_channel_combos.size() << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "\n1️⃣  Bireysel kanal skorları hesaplanıyor (Paralel)..." << std::endl;
        std::map<int, Vector> individual_scores;
        std::vector<int> unique_lengths = channel_search_space;
        std::sort(unique_lengths.begin(), unique_lengths.end());
        unique_lengths.erase(std::unique(unique_lengths.begin(), unique_lengths.end()), unique_lengths.end());

        BS::thread_pool pool;
        std::mutex map_mutex;
        std::vector<std::future<void>> futures_map;
        for (int length : unique_lengths) {
            futures_map.push_back(pool.submit([&, length]() {
                Vector scores = calculate_pine_script_scores_correct(market_data_.close, length, std_dev_mult, std_dev_mult);
                std::lock_guard<std::mutex> lock(map_mutex);
                individual_scores[length] = std::move(scores);
            }));
        }
        for (auto &f : futures_map) f.get();
        std::cout << "   ✓ Tamamlandı!" << std::endl;

        std::cout << "\n2️⃣  " << all_channel_combos.size() << " kombinasyon hesaplanıyor ve kaydediliyor (Paralel)..." << std::endl;
        HighFive::File file(CHANNEL_COMBOS_FILE, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        auto group = file.createGroup("channel_combinations");
        std::mutex h5_mutex;
        std::atomic<size_t> completed_combos = 0;
        
        std::vector<std::future<void>> combo_futures;
        for (const auto& channels : all_channel_combos) {
            combo_futures.push_back(pool.submit([&, channels]() {
                Vector weights = Eigen::Map<const IntVector>(channels.data(), num_channels).cast<double>();
                weights /= weights.sum();
                Vector channel_scores = Vector::Zero(market_data_.close.size());
                for (int j = 0; j < num_channels; ++j) {
                    channel_scores += individual_scores.at(channels[j]) * weights(j);
                }
                channel_scores = channel_scores.cwiseMin(100.0).cwiseMax(0.0);
                
                std::string combo_id;
                for (size_t j = 0; j < channels.size(); ++j) {
                    combo_id += std::to_string(channels[j]) + (j == channels.size() - 1 ? "" : "_");
                }
                std::vector<double> channel_vec(channel_scores.data(), channel_scores.data() + channel_scores.size());

                {
                    std::lock_guard<std::mutex> lock(h5_mutex);
                    group.createDataSet(combo_id, channel_vec);
                }

                size_t current_count = ++completed_combos;
                if (current_count % 10000 == 0) {
                     // ... ilerleme durumu yazdırma ...
                }
            }));
        }
        for (auto &f : combo_futures) f.get();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        // ... (fonksiyon sonu orijinal ile aynı) ...
    }
    void save_metadata() { /* Orijinal ile aynı */ }

private:
    std::string data_path_;
    MarketData market_data_;
};


// ==================== CHECKPOINT & CSV (Değişiklik yok) ====================
// ... Structs and functions from the original code ...

// ==================== MAIN OPTIMIZATION STRUCTURES (REFACTORED) ====================
struct Task {
    std::string combo_id;
    int cmf_len;
    double cmf_buy_thresh;
    double cmf_sell_thresh;
};
// (WorkerResult zaten yukarıda tanımlandı.)

// Worker'ların kullanacağı, paylaşılan ve değiştirilmeyen veri (REFACTORED)
struct SharedData {
    Vector prices;
    std::pair<int, int> buy_threshold_range;
    std::pair<int, int> sell_threshold_range;
    int threshold_step;
    // Bellekteki tüm ön-hesaplama verilerine bir pointer
    std::shared_ptr<const PrecomputedDataCache> cache;
};

// ==================== ULTRA-FAST IN-MEMORY WORKER (REFACTORED) ====================
// Bu worker dosyadan okuma yapmaz, tüm veriyi RAM'den alır.
WorkerResult process_single_test_in_memory(const Task& task, const std::shared_ptr<const SharedData>& shared_data) {
    WorkerResult worker_result;
    
    try {
        // 1. ADIM: Bellekten veriyi anında çek (I/O YOK!)
        const Vector& channel_scores = shared_data->cache->channel_scores.at(task.combo_id);
        const Vector& cmf_values = shared_data->cache->cmf_values.at(task.cmf_len);
        const auto& prices = shared_data->prices;

        for (int buy_thresh = shared_data->buy_threshold_range.first; buy_thresh <= shared_data->buy_threshold_range.second; buy_thresh += shared_data->threshold_step) {
            for (int sell_thresh = shared_data->sell_threshold_range.first; sell_thresh <= shared_data->sell_threshold_range.second; sell_thresh += shared_data->threshold_step) {
                if (buy_thresh <= sell_thresh) continue;
                
                BacktestOutput backtest_out = run_backtest_with_cmf(
                    channel_scores, cmf_values, prices,
                    static_cast<double>(buy_thresh), static_cast<double>(sell_thresh),
                    task.cmf_buy_thresh, task.cmf_sell_thresh, 10000.0
                );

                if (backtest_out.trade_profits.size() > 10) {
                    std::vector<double> metrics = calculate_metrics_fast(
                        backtest_out.trade_profits, backtest_out.trade_profit_pcts, backtest_out.trade_durations,
                        backtest_out.final_portfolio_value, 10000.0,
                        backtest_out.max_consecutive_wins, backtest_out.max_consecutive_losses
                    );
                    
                    double quality_score = metrics[7];
                    
                    std::vector<int> channels;
                    std::stringstream ss(task.combo_id);
                    std::string segment;
                    while(std::getline(ss, segment, '_')) {
                       channels.push_back(std::stoi(segment));
                    }

                    ResultRow current_result = {channels, task.cmf_len, task.cmf_buy_thresh, task.cmf_sell_thresh, buy_thresh, sell_thresh, metrics};
                    
                    if (quality_score >= SCORE_THRESHOLD) {
                        worker_result.good_results.push_back(current_result);
                    }

                    if (!worker_result.best_local_result.has_value() || quality_score > worker_result.best_local_result->metrics[7]) {
                        worker_result.best_local_result = current_result;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        // Hata durumunda (örn. map'te key bulunamazsa) boş sonuç döndür.
    }
    return worker_result;
}


// ==================== MAIN (REFACTORED) ====================
int main(int argc, char* argv[]) {
    // Basit argüman ayrıştırıcı (orijinalde cxxopts kullanılıyordu)
    std::string data_path = "ETHBTC_5m_50000_bars.csv";
    bool force_recompute = false;
    bool precompute_only = false;
    std::vector<std::string> tasks_to_run;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--data" && i + 1 < argc) { data_path = argv[++i]; }
        else if (a == "--force-recompute") { force_recompute = true; }
        else if (a == "--precompute-only") { precompute_only = true; }
        else if (a == "--task" && i + 1 < argc) { tasks_to_run.push_back(argv[++i]); }
    }

    std::vector<int> search_space = { /* ... orijinaldeki gibi ... */ };
    std::vector<double> cmf_buy_thresholds = {-0.05, -0.10, -0.15, -0.20, -0.25};
    std::vector<double> cmf_sell_thresholds = {0.05, 0.10, 0.15, 0.20, 0.25};
    int num_channels = 4;
    double std_dev_mult = 5.0;

    // Ön-hesaplama kontrolü (artık paralel motoru kullanır)
    if (force_recompute || !fs::exists(CMF_CACHE_FILE) || !fs::exists(CHANNEL_COMBOS_FILE) || precompute_only) {
        // ... (mesajlar orijinal ile aynı) ...
        PreComputationEngine precomp(data_path);
        precomp.load_data();
        precomp.precompute_all_cmf(search_space); 
        precomp.precompute_all_channel_combos(search_space, num_channels, std_dev_mult);
        precomp.save_metadata();
        // ...
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "🔥 ULTRA-FAST OPTİMİZASYON BAŞLIYOR" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // ================== YENİ: VERİLERİ BELLEĞE YÜKLEME ==================
    std::cout << "🧠 Ön-hesaplama verileri belleğe yükleniyor... (RAM kullanımı artacak)" << std::endl;
    auto cache_start_time = std::chrono::high_resolution_clock::now();

    auto cache = std::make_shared<PrecomputedDataCache>();
    
    // CMF verilerini yükle
    HighFive::File cmf_file(CMF_CACHE_FILE, HighFive::File::ReadOnly);
    auto cmf_group = cmf_file.getGroup("cmf_values");
    auto cmf_names = cmf_group.listObjectNames();
    for(const auto& name : cmf_names) {
        int len = std::stoi(name.substr(name.find('_') + 1));
        std::vector<double> data_vec;
        cmf_group.getDataSet(name).read(data_vec);
        cache->cmf_values[len] = Eigen::Map<Vector>(data_vec.data(), data_vec.size());
    }
    
    // Kanal kombinasyon verilerini yükle
    HighFive::File channel_file(CHANNEL_COMBOS_FILE, HighFive::File::ReadOnly);
    auto channel_group = channel_file.getGroup("channel_combinations");
    auto channel_names = channel_group.listObjectNames();
    for(const auto& name : channel_names) {
        std::vector<double> data_vec;
        channel_group.getDataSet(name).read(data_vec);
        cache->channel_scores[name] = Eigen::Map<Vector>(data_vec.data(), data_vec.size());
    }

    auto cache_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "✅ Belleğe yükleme tamamlandı! Süre: " 
              << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(cache_end_time - cache_start_time).count() << "s" << std::endl;
    // ======================================================================

    // Fiyat verisini yükle
    io::CSVReader<5> in(data_path);
    in.read_header(io::ignore_extra_column, "timestamp", "high", "low", "close", "volume");
    std::vector<double> close_v;
    double high, low, close, volume; std::string ts;
    while(in.read_row(ts, high, low, close, volume)) close_v.push_back(close);
    Vector prices_vec = Eigen::Map<Vector>(close_v.data(), close_v.size());
    std::cout << "✅ " << prices_vec.size() << " bar yüklendi" << std::endl;
    
    // Görevleri oluştur (orijinal ile aynı)
    std::vector<Task> all_tasks;
    // ...

    // Checkpoint yükle ve kalan görevleri filtrele (orijinal ile aynı)
    CheckpointData checkpoint = load_checkpoint();
    // ...

    unsigned int num_cores = std::thread::hardware_concurrency();
    BS::thread_pool pool(num_cores);
    std::vector<std::future<WorkerResult>> futures;
    
    // SharedData'yı yeni cache ile oluştur
    auto shared_data = std::make_shared<const SharedData>(SharedData{
        prices_vec, {60, 100}, {0, 40}, 5, cache // Cache'i buraya ekle
    });

    // Görevleri yeni in-memory worker fonksiyonuyla havuza gönder
    for (const auto& task : all_tasks) {
        futures.push_back(pool.submit(process_single_test_in_memory, task, shared_data));
    }
    
    // Sonuçları toplama, checkpoint kaydetme ve raporlama kısmı orijinal ile aynıdır.
    // ... (Main fonksiyonunun geri kalanı) ...
    
    return 0;
}


// NOT: Orijinal kodunuzdaki core numerical/utility fonksiyonlarının gövdelerini 
// (calculate_cmf, run_backtest_with_cmf, calculate_metrics_fast vb.)
// bu şablona eklemeniz gerekmektedir. Kısalık amacıyla buraya dahil edilmemiştir.