// ================================================================================= //
//                       ULTRA-FAST CMF OPTIMIZER - C++ VERSION                      //
// ================================================================================= //
// Python kodunun mantığı, yapısı ve bütünlüğü korunarak C++'a çevrilmiştir.          //
// Herhangi bir sadeleştirme veya eksiltme yapılmamıştır.                            //
//                                                                                   //
// GEREKLİ KÜTÜPHANELER:                                                              //
// - Eigen3 (vektör/matris işlemleri için)                                           //
// - HighFive (HDF5 dosya I/O için)                                                  //
// - HDF5 (HighFive'ın bağımlılığı)                                                  //
// - BS::thread_pool (paralel işlemler için)                                         //
// - cereal (checkpointing için)                                                     //
// - fast-cpp-csv-parser (veri okuma için)                                           //
// - cxxopts (komut satırı argümanları için)                                         //
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
#include <thread>
#include <filesystem>

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

// ==================== UTILITY FUNCTIONS ====================
// Kombinasyon üreteci
void generate_combinations(
    const std::vector<int>& elements,
    int k,
    std::function<void(const std::vector<int>&)> callback) {
    if (k == 0) {
        return;
    }
    std::vector<int> combination(k);
    std::function<void(int, int)> generate =
        [&](int offset, int depth) {
        if (depth == k) {
            callback(combination);
            return;
        }
        for (size_t i = offset; i < elements.size(); ++i) {
            combination[depth] = elements[i];
            generate(i, depth + 1);
        }
    };
    generate(0, 0);
}


// ==================== CORE NUMERICAL FUNCTIONS ====================

Vector calculate_cmf(const Vector& high, const Vector& low, const Vector& close, const Vector& volume, int length) {
    long n = close.size();
    Vector cmf_values = Vector::Zero(n);
    
    for (long i = length; i < n; ++i) {
        double mf_volume_sum = 0.0;
        double volume_sum = 0.0;
        
        for (long j = i - length + 1; j <= i; ++j) {
            double hl_range = high[j] - low[j];
            if (hl_range > 1e-9) {
                double mf_multiplier = ((close[j] - low[j]) - (high[j] - close[j])) / hl_range;
                mf_volume_sum += mf_multiplier * volume[j];
            }
            volume_sum += volume[j];
        }
        
        if (volume_sum > 1e-9) {
            cmf_values[i] = mf_volume_sum / volume_sum;
        }
    }
    return cmf_values;
}

Vector calculate_pine_script_scores_correct(const Vector& prices, int length, double upper_mult, double lower_mult) {
    long n = prices.size();
    Vector scores = Vector::Constant(n, 50.0);
    
    Vector x = Vector::LinSpaced(length, 1.0, static_cast<double>(length));
    double sum_x = x.sum();
    double sum_x2 = x.squaredNorm();
    double reg_denominator = static_cast<double>(length) * sum_x2 - sum_x * sum_x;
    double mean_x = x.mean();
    
    for (long i = length; i < n; ++i) {
        Vector price_window = prices.segment(i - length, length);
        Vector log_window = price_window.array().log();
        
        double sum_y = log_window.sum();
        double sum_xy = (x.array() * log_window.array()).sum();
        
        double slope = (reg_denominator != 0) ? (static_cast<double>(length) * sum_xy - sum_x * sum_y) / reg_denominator : 0.0;
        double intercept = log_window.mean() - slope * mean_x;
        
        double std_dev_acc = 0.0;
        for (int j = 0; j < length; ++j) {
            double residual = log_window[j] - (intercept + slope * x[j]);
            std_dev_acc += residual * residual;
        }
        double std_dev = std::sqrt(std_dev_acc / static_cast<double>(length));
        
        double log_middle = intercept + slope;
        double log_upper = log_middle + upper_mult * std_dev;
        double log_lower = log_middle - lower_mult * std_dev;
        double log_current_price = log_window[length - 1];
        
        double score = 50.0;
        if (log_current_price >= log_middle) {
            double denominator = log_upper - log_middle;
            score = (denominator > 1e-9) ? 50.0 + (50.0 * (log_current_price - log_middle) / denominator) : 50.0;
        } else {
            double denominator = log_middle - log_lower;
            score = (denominator > 1e-9) ? 50.0 * (log_current_price - log_lower) / denominator : 50.0;
        }
        
        scores[i] = std::min(100.0, std::max(0.0, score));
    }
    return scores;
}


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

BacktestOutput run_backtest_with_cmf(const Vector& scores, const Vector& cmf_values, const Vector& prices,
                                     double buy_threshold, double sell_threshold,
                                     double cmf_buy_threshold, double cmf_sell_threshold,
                                     double initial_capital) {
    long n = scores.size();
    double capital = initial_capital;
    double position_price = 0.0;
    long position_entry_bar = 0;
    
    std::vector<double> trade_profits;
    std::vector<double> trade_profit_pcts;
    std::vector<long> trade_durations;
    
    int consecutive_wins = 0;
    int consecutive_losses = 0;
    int max_consecutive_wins = 0;
    int max_consecutive_losses = 0;
    
    double final_portfolio_value = initial_capital;
    
    bool cmf_buy_armed = false;
    bool cmf_sell_armed = false;
    
    for (long i = 1; i < n; ++i) {
        double current_price = prices[i];
        double current_cmf = cmf_values[i];
        double current_score = scores[i];
        double prev_score = scores[i - 1];

        if (position_price == 0.0) {
            if (current_score <= sell_threshold) {
                if (current_cmf <= cmf_buy_threshold) {
                    cmf_buy_armed = true;
                }
            }
            
            if (cmf_buy_armed && prev_score <= sell_threshold && current_score > sell_threshold) {
                position_price = current_price;
                position_entry_bar = i;
                cmf_buy_armed = false;
                cmf_sell_armed = false;
            }
            
            if (prev_score >= buy_threshold && current_score < buy_threshold) {
                cmf_buy_armed = false;
            }
        } else {
            if (current_score >= buy_threshold) {
                if (current_cmf >= cmf_sell_threshold) {
                    cmf_sell_armed = true;
                }
            }

            if (cmf_sell_armed && prev_score >= buy_threshold && current_score < buy_threshold) {
                double profit = current_price - position_price;
                double profit_pct = (profit / position_price) * 100.0;
                long trade_duration = i - position_entry_bar;
                
                capital += profit;
                trade_profits.push_back(profit);
                trade_profit_pcts.push_back(profit_pct);
                trade_durations.push_back(trade_duration);
                
                if (profit > 0.0) {
                    consecutive_wins++;
                    consecutive_losses = 0;
                    if (consecutive_wins > max_consecutive_wins) max_consecutive_wins = consecutive_wins;
                } else if (profit < 0.0) {
                    consecutive_losses++;
                    consecutive_wins = 0;
                    if (consecutive_losses > max_consecutive_losses) max_consecutive_losses = consecutive_losses;
                } else {
                    consecutive_wins = 0;
                    consecutive_losses = 0;
                }
                
                position_price = 0.0;
                cmf_sell_armed = false;
                cmf_buy_armed = false;
            }
            
            if (prev_score <= sell_threshold && current_score > sell_threshold) {
                cmf_sell_armed = false;
            }
        }
        
        if (position_price > 0.0) {
            final_portfolio_value = capital + (current_price - position_price);
        } else {
            final_portfolio_value = capital;
        }
    }
    
    return {trade_profits, trade_profit_pcts, trade_durations, final_portfolio_value,
            consecutive_wins, consecutive_losses, max_consecutive_wins, max_consecutive_losses};
}


std::vector<double> calculate_metrics_fast(const std::vector<double>& trade_profits, 
                                        const std::vector<double>& trade_profit_pcts, 
                                        const std::vector<long>& trade_durations,
                                        double final_value, double initial_capital,
                                        int max_consec_wins, int max_consec_losses) {

    std::vector<double> results(17, 0.0);
    if (trade_profits.empty()) {
        return results;
    }

    size_t total_trades = trade_profits.size();
    long wins = 0;
    double gross_profit = 0.0, gross_loss = 0.0;
    for (double p : trade_profits) {
        if (p > 0) {
            wins++;
            gross_profit += p;
        } else {
            gross_loss += p;
        }
    }
    gross_loss = std::abs(gross_loss);

    double win_rate = (static_cast<double>(wins) / total_trades) * 100.0;
    double profit_factor = (gross_loss != 0) ? gross_profit / gross_loss : 1000.0;

    Vector cumsum(total_trades);
    std::partial_sum(trade_profits.begin(), trade_profits.end(), cumsum.data());
    Vector running_max(total_trades);
    running_max[0] = cumsum[0];
    for (size_t i = 1; i < total_trades; ++i) {
        running_max[i] = std::max(running_max[i-1], cumsum[i]);
    }
    double max_drawdown_val = (cumsum - running_max).minCoeff();
    double max_drawdown = (total_trades > 0) ? (max_drawdown_val / initial_capital) * 100.0 : 0.0;

    double total_return_pct = ((final_value - initial_capital) / initial_capital) * 100.0;
    
    double mean_return_pct = 0.0;
    double std_dev_pct = 0.0;
    if (total_trades > 1) {
        mean_return_pct = std::accumulate(trade_profit_pcts.begin(), trade_profit_pcts.end(), 0.0) / total_trades;
        double sq_sum = std::inner_product(trade_profit_pcts.begin(), trade_profit_pcts.end(), trade_profit_pcts.begin(), 0.0);
        std_dev_pct = std::sqrt(sq_sum / total_trades - mean_return_pct * mean_return_pct);
    }
    double sharpe_ratio = (std_dev_pct != 0) ? (mean_return_pct / std_dev_pct) * std::sqrt(total_trades) : 0.0;

    std::vector<double> losing_returns;
    for (double p : trade_profit_pcts) if (p < 0) losing_returns.push_back(p);
    double downside_std = 0.0;
    if (!losing_returns.empty()) {
        double mean_losing = std::accumulate(losing_returns.begin(), losing_returns.end(), 0.0) / losing_returns.size();
        double sq_sum_losing = std::inner_product(losing_returns.begin(), losing_returns.end(), losing_returns.begin(), 0.0);
        downside_std = std::sqrt(sq_sum_losing / losing_returns.size() - mean_losing * mean_losing);
    }
    double sortino_ratio = (downside_std != 0) ? (mean_return_pct / downside_std) * std::sqrt(total_trades) : sharpe_ratio;

    double calmar_ratio = (max_drawdown != 0) ? std::abs(total_return_pct / max_drawdown) : 0.0;

    std::vector<double> winning_trades_pct, losing_trades_pct;
    for(double p : trade_profit_pcts) {
        if(p > 0) winning_trades_pct.push_back(p);
        else if (p < 0) losing_trades_pct.push_back(p);
    }
    double avg_profit_pct = !winning_trades_pct.empty() ? std::accumulate(winning_trades_pct.begin(), winning_trades_pct.end(), 0.0) / winning_trades_pct.size() : 0.0;
    double avg_loss_pct = !losing_trades_pct.empty() ? std::accumulate(losing_trades_pct.begin(), losing_trades_pct.end(), 0.0) / losing_trades_pct.size() : 0.0;
    double profit_loss_ratio = (avg_loss_pct != 0) ? std::abs(avg_profit_pct / avg_loss_pct) : 0.0;
    
    double recovery_factor = (max_drawdown != 0) ? std::abs(total_return_pct / max_drawdown) : 0.0;
    double expectancy = (win_rate / 100.0 * avg_profit_pct) + ((1.0 - win_rate / 100.0) * avg_loss_pct);
    double avg_trade_duration = !trade_durations.empty() ? static_cast<double>(std::accumulate(trade_durations.begin(), trade_durations.end(), 0LL)) / trade_durations.size() : 0.0;

    double pf_score, sharpe_score, sortino_score, win_rate_score, drawdown_score, pl_ratio_score, calmar_score, trade_count_score, consistency_score;

    if (profit_factor >= 3.0) pf_score = 1.0;
    else if (profit_factor >= 2.0) pf_score = 0.8 + (profit_factor - 2.0) * 0.2;
    else if (profit_factor >= 1.5) pf_score = 0.5 + (profit_factor - 1.5) * 0.6;
    else pf_score = std::max(0.0, profit_factor / 1.5 * 0.5);

    if (sharpe_ratio >= 2.5) sharpe_score = 1.0;
    else if (sharpe_ratio >= 1.5) sharpe_score = 0.8 + (sharpe_ratio - 1.5) * 0.2;
    else if (sharpe_ratio >= 1.0) sharpe_score = 0.5 + (sharpe_ratio - 1.0) * 0.6;
    else if (sharpe_ratio > 0) sharpe_score = sharpe_ratio / 1.0 * 0.5;
    else sharpe_score = 0.0;
    
    if (sortino_ratio >= 3.0) sortino_score = 1.0;
    else if (sortino_ratio >= 2.0) sortino_score = 0.8 + (sortino_ratio - 2.0) * 0.2;
    else if (sortino_ratio >= 1.0) sortino_score = 0.5 + (sortino_ratio - 1.0) * 0.6;
    else if (sortino_ratio > 0) sortino_score = sortino_ratio / 1.0 * 0.5;
    else sortino_score = 0.0;

    if (50.0 <= win_rate && win_rate <= 70.0) win_rate_score = 1.0;
    else if (win_rate < 50.0) win_rate_score = win_rate / 50.0 * 0.8;
    else win_rate_score = std::max(0.5, 1.0 - (win_rate - 70.0) / 30.0 * 0.5);

    if (max_drawdown >= -5.0) drawdown_score = 1.0;
    else if (max_drawdown >= -15.0) drawdown_score = 0.7 + (max_drawdown + 15.0) / 10.0 * 0.3;
    else if (max_drawdown >= -30.0) drawdown_score = 0.3 + (max_drawdown + 30.0) / 15.0 * 0.4;
    else drawdown_score = std::max(0.0, 0.3 * (1.0 + max_drawdown / 100.0));

    if (profit_loss_ratio >= 3.0) pl_ratio_score = 1.0;
    else if (profit_loss_ratio >= 2.0) pl_ratio_score = 0.8 + (profit_loss_ratio - 2.0) * 0.2;
    else if (profit_loss_ratio >= 1.5) pl_ratio_score = 0.5 + (profit_loss_ratio - 1.5) * 0.6;
    else pl_ratio_score = std::max(0.0, profit_loss_ratio / 1.5 * 0.5);

    if (calmar_ratio >= 3.0) calmar_score = 1.0;
    else if (calmar_ratio >= 2.0) calmar_score = 0.8 + (calmar_ratio - 2.0) * 0.2;
    else if (calmar_ratio >= 1.0) calmar_score = 0.5 + (calmar_ratio - 1.0) * 0.6;
    else calmar_score = std::max(0.0, calmar_ratio / 1.0 * 0.5);

    if (total_trades < 10) trade_count_score = total_trades / 10.0 * 0.3;
    else if (total_trades < 30) trade_count_score = 0.3 + (total_trades - 10) / 20.0 * 0.4;
    else if (total_trades >= 30 && total_trades <= 300) trade_count_score = 1.0;
    else if (total_trades <= 500) trade_count_score = 1.0 - (total_trades - 300) / 200.0 * 0.3;
    else trade_count_score = std::max(0.3, 0.7 - (total_trades - 500) / 500.0 * 0.4);

    if (max_consec_losses <= 3) consistency_score = 1.0;
    else if (max_consec_losses <= 5) consistency_score = 0.8;
    else if (max_consec_losses <= 7) consistency_score = 0.6;
    else if (max_consec_losses <= 10) consistency_score = 0.4;
    else consistency_score = std::max(0.2, 0.4 - (static_cast<double>(max_consec_losses) - 10) / 20.0 * 0.2);
    
    double quality_score =
        pf_score * 0.10 + sharpe_score * 0.20 + sortino_score * 0.15 +
        drawdown_score * 0.20 + pl_ratio_score * 0.03 + calmar_score * 0.25 +
        win_rate_score * 0.01 + trade_count_score * 0.05 + consistency_score * 0.01;

    results[0] = static_cast<double>(total_trades);
    results[1] = win_rate;
    results[2] = profit_factor;
    results[3] = max_drawdown;
    results[4] = sharpe_ratio;
    results[5] = sortino_ratio;
    results[6] = calmar_ratio;
    results[7] = quality_score;
    results[8] = avg_profit_pct;
    results[9] = avg_loss_pct;
    results[10] = profit_loss_ratio;
    results[11] = recovery_factor;
    results[12] = expectancy;
    results[13] = static_cast<double>(max_consec_losses);
    results[14] = static_cast<double>(max_consec_wins);
    results[15] = avg_trade_duration;
    results[16] = total_return_pct;

    return results;
}

// Data yapısı
struct MarketData {
    Vector high, low, close, volume;
};

// ==================== PRE-COMPUTATION ENGINE ====================
class PreComputationEngine {
public:
    PreComputationEngine(const std::string& data_path) : data_path_(data_path) {}

    void load_data() {
        std::cout << "📂 Veri yükleniyor: " << data_path_ << std::endl;
        io::CSVReader<5, io::trim_chars<' ', '\t'>, io::no_quote_escape<','>> in(data_path_);
        in.read_header(io::ignore_extra_column, "timestamp", "high", "low", "close", "volume");
        
        std::vector<double> high_v, low_v, close_v, volume_v;
        double high, low, close, volume;
        std::string timestamp;
        while (in.read_row(timestamp, high, low, close, volume)) {
            high_v.push_back(high);
            low_v.push_back(low);
            close_v.push_back(close);
            volume_v.push_back(volume);
        }
        
        market_data_.high = Eigen::Map<Vector>(high_v.data(), high_v.size());
        market_data_.low = Eigen::Map<Vector>(low_v.data(), low_v.size());
        market_data_.close = Eigen::Map<Vector>(close_v.data(), close_v.size());
        market_data_.volume = Eigen::Map<Vector>(volume_v.data(), volume_v.size());

        std::cout << "✅ " << market_data_.close.size() << " bar yüklendi" << std::endl;
    }

    void precompute_all_cmf(const std::vector<int>& cmf_lengths) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "📊 CMF ÖN-HESAPLAMA" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();

        HighFive::File file(CMF_CACHE_FILE, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        auto group = file.createGroup("cmf_values");

        for (size_t i = 0; i < cmf_lengths.size(); ++i) {
            int len = cmf_lengths[i];
            std::cout << "⚙️  [" << i + 1 << "/" << cmf_lengths.size() << "] CMF Length=" << len << "..." << std::flush;
            Vector cmf_array = calculate_cmf(market_data_.high, market_data_.low, market_data_.close, market_data_.volume, len);
            group.createDataSet("length_" + std::to_string(len), cmf_array);
            std::cout << " ✓" << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        auto file_size_mb = fs::file_size(CMF_CACHE_FILE) / (1024.0 * 1024.0);
        std::cout << "\n✅ Tamamlandı! Süre: " << std::fixed << std::setprecision(1) << elapsed.count() << "s | Boyut: " << file_size_mb << " MB" << std::endl;
    }
    
    void precompute_all_channel_combos(const std::vector<int>& channel_search_space, int num_channels, double std_dev_mult) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "📊 KANAL KOMBİNASYONLARI ÖN-HESAPLAMA (SÜPER HIZ!)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        std::vector<std::vector<int>> all_channel_combos;
        generate_combinations(channel_search_space, num_channels, [&](const std::vector<int>& combo) {
            all_channel_combos.push_back(combo);
        });

        std::cout << "🔢 Toplam kombinasyon: " << all_channel_combos.size() << std::endl;
        std::cout << "⚠️  Bu adım biraz uzun sürebilir ama sonrası ÇOK HIZLI olacak!" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "\n1️⃣  Individual kanal skorları hesaplanıyor..." << std::endl;
        std::map<int, Vector> individual_scores;
        std::vector<int> unique_lengths = channel_search_space;
        std::sort(unique_lengths.begin(), unique_lengths.end());
        unique_lengths.erase(std::unique(unique_lengths.begin(), unique_lengths.end()), unique_lengths.end());

        for (size_t i = 0; i < unique_lengths.size(); ++i) {
            int length = unique_lengths[i];
            std::cout << "   [" << i + 1 << "/" << unique_lengths.size() << "] Length=" << length << "..." << std::flush;
            individual_scores[length] = calculate_pine_script_scores_correct(market_data_.close, length, std_dev_mult, std_dev_mult);
            std::cout << " ✓" << std::endl;
        }
        
        std::cout << "\n2️⃣  " << all_channel_combos.size() << " kombinasyon hesaplanıyor ve kaydediliyor..." << std::endl;
        
        HighFive::File file(CHANNEL_COMBOS_FILE, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);
        auto group = file.createGroup("channel_combinations");

        for (size_t i = 0; i < all_channel_combos.size(); ++i) {
            const auto& channels = all_channel_combos[i];
            if ((i + 1) % 10000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double rate = (i + 1) / elapsed;
                double remaining = (all_channel_combos.size() - i - 1) / rate;
                std::cout << "   [" << i + 1 << "/" << all_channel_combos.size() << "] - " << static_cast<int>(rate) << " combo/s - Kalan: ~" << static_cast<int>(remaining / 60) << "dk" << std::endl;
            }

            Matrix all_scores(market_data_.close.size(), num_channels);
            Vector weights(num_channels);
            double sum_weights = 0;
            for(int j=0; j < num_channels; ++j) {
                all_scores.col(j) = individual_scores[channels[j]];
                weights(j) = static_cast<double>(channels[j]);
                sum_weights += channels[j];
            }
            weights /= sum_weights;

            Vector channel_scores = all_scores * weights;
            channel_scores = channel_scores.cwiseMin(100.0).cwiseMax(0.0);
            
            std::string combo_id;
            for(size_t j=0; j<channels.size(); ++j) {
                combo_id += std::to_string(channels[j]) + (j == channels.size()-1 ? "" : "_");
            }
            group.createDataSet(combo_id, channel_scores);
        }
        
        file.createAttribute<size_t>("num_combinations", all_channel_combos.size());
        file.createAttribute<double>("std_dev_mult", std_dev_mult);
        file.createAttribute<int>("num_channels", num_channels);

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_total = std::chrono::duration<double>(end_time - start_time).count();
        auto file_size_gb = fs::file_size(CHANNEL_COMBOS_FILE) / (1024.0 * 1024.0 * 1024.0);
        
        std::cout << "\n✅ TÜM KOMBİNASYONLAR HAZIR!" << std::endl;
        std::cout << "⏱️  Süre: " << std::fixed << std::setprecision(1) << elapsed_total / 60.0 << " dakika" << std::endl;
        std::cout << "💾 Dosya boyutu: " << std::fixed << std::setprecision(2) << file_size_gb << " GB" << std::endl;
        std::cout << "🚀 Runtime'da HİÇ HESAPLAMA YAPILMAYACAK, sadece okuma!" << std::endl;
    }

    void save_metadata() {
        // Basit bir JSON yazıcı
        std::ofstream ofs(METADATA_FILE);
        ofs << "{\n";
        ofs << "  \"n_bars\": " << market_data_.close.size() << ",\n";
        ofs << "  \"data_path\": \"" << data_path_ << "\"\n";
        // Diğer metadata bilgileri de eklenebilir
        ofs << "}\n";
    }

private:
    std::string data_path_;
    MarketData market_data_;
};


// ==================== CHECKPOINT & CSV ====================

// Basit bir sonuç yapısı
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

void save_checkpoint(const CheckpointData& data) {
    std::ofstream os(CHECKPOINT_FILE, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    archive(data);
}

CheckpointData load_checkpoint() {
    CheckpointData data;
    if (fs::exists(CHECKPOINT_FILE)) {
        try {
            std::ifstream is(CHECKPOINT_FILE, std::ios::binary);
            cereal::BinaryInputArchive archive(is);
            archive(data);
            std::cout << "📂 Checkpoint: " << data.completed_hashes.size() << " tamamlanmış" << std::endl;
        } catch (...) {
            std::cerr << "⚠️ Checkpoint dosyası okunamadı, baştan başlanıyor." << std::endl;
            return CheckpointData{};
        }
    }
    return data;
}

// ==================== MAIN OPTIMIZATION STRUCTURES ====================

struct Task {
    std::string combo_id;
    int cmf_len;
    double cmf_buy_thresh;
    double cmf_sell_thresh;
};

struct WorkerResult {
    std::optional<ResultRow> best_local_result;
    std::vector<ResultRow> good_results;
};


// ==================== ULTRA-FAST WORKER ====================
// Worker'ların kullanacağı, paylaşılan ve değiştirilmeyen veri
struct SharedData {
    Vector prices;
    std::pair<int, int> buy_threshold_range;
    std::pair<int, int> sell_threshold_range;
    int threshold_step;
};

WorkerResult process_single_test_ultra_fast(const Task& task, const std::shared_ptr<const SharedData>& shared_data) {
    WorkerResult worker_result;
    
    try {
        // Her worker kendi dosya tanıtıcısını açar
        HighFive::File channel_file(CHANNEL_COMBOS_FILE, HighFive::File::ReadOnly);
        HighFive::File cmf_file(CMF_CACHE_FILE, HighFive::File::ReadOnly);

        Vector channel_scores;
        channel_file.getDataSet("channel_combinations/" + task.combo_id).read(channel_scores);
        
        Vector cmf_values;
        cmf_file.getDataSet("cmf_values/length_" + std::to_string(task.cmf_len)).read(cmf_values);
        
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
                    
                    // Kanal ID'sini ayrıştır
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
        // Hata durumunda boş sonuç döndür
    }
    return worker_result;
}

// ... CSVBatchWriter ve ana optimizasyon döngüsü (çok uzun olduğu için ana fonksiyona entegre)

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    cxxopts::Options options(argv[0], "Ultra-Fast CMF Optimizer - C++ Version");
    options.add_options()
        ("d,data", "Data file path", cxxopts::value<std::string>()->default_value("ETHBTC_5m_50000_bars.csv"))
        ("f,force-recompute", "Force re-computation of all data")
        ("p,precompute-only", "Only run pre-computation, then exit")
        ("h,help", "Print usage");
    
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    std::string data_path = result["data"].as<std::string>();
    bool force_recompute = result.count("force-recompute") > 0;
    bool precompute_only = result.count("precompute-only") > 0;

    if (!fs::exists(data_path)) {
        std::cerr << "❌ '" << data_path << "' bulunamadı" << std::endl;
        return 1;
    }

    std::vector<int> search_space = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 624, 688, 752, 816, 880, 944, 1008, 1072, 1136, 1264, 1392, 1520, 1648, 1776, 1904, 2032, 2160, 2288};
    std::vector<double> cmf_buy_thresholds = {-0.05, -0.10, -0.15, -0.20, -0.25};
    std::vector<double> cmf_sell_thresholds = {0.05, 0.10, 0.15, 0.20, 0.25};
    int num_channels = 4;
    double std_dev_mult = 5.0;

    if (force_recompute || !fs::exists(CMF_CACHE_FILE) || !fs::exists(CHANNEL_COMBOS_FILE) || precompute_only) {
        if (!precompute_only) {
            std::cout << "\n🔥 ÖN-HESAPLAMA GEREKLİ!" << std::endl;
            std::cout << "⚠️  Bu işlem bir kez yapılacak, sonrası ÇOK HIZLI!" << std::endl;
        } else {
            std::cout << "\n🔥 SADECE ÖN-HESAPLAMA MODU" << std::endl;
        }

        if(!fs::exists(PRECOMP_DIR)) fs::create_directory(PRECOMP_DIR);
        
        PreComputationEngine precomp(data_path);
        precomp.load_data();
        precomp.precompute_all_cmf(search_space); // cmf_lengths = search_space
        precomp.precompute_all_channel_combos(search_space, num_channels, std_dev_mult);
        precomp.save_metadata();
        
        if (precompute_only) {
            std::cout << "\n✅ Ön-hesaplama tamamlandı!" << std::endl;
            return 0;
        }
    } else {
        std::cout << "\n✅ Tüm ön-hesaplamalar mevcut - ULTRA HIZ MODU!" << std::endl;
    }


    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "🔥 ULTRA-FAST OPTİMİZASYON BAŞLIYOR" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Veriyi yükle
    io::CSVReader<5> in(data_path);
    in.read_header(io::ignore_extra_column, "timestamp", "high", "low", "close", "volume");
    std::vector<double> close_v;
    double high, low, close, volume; std::string ts;
    while(in.read_row(ts, high, low, close, volume)) close_v.push_back(close);
    Vector prices_vec = Eigen::Map<Vector>(close_v.data(), close_v.size());
    std::cout << "✅ " << prices_vec.size() << " bar yüklendi" << std::endl;
    
    // Görevleri oluştur
    std::vector<Task> all_tasks;
    std::vector<std::vector<int>> all_channel_combos;
    generate_combinations(search_space, num_channels, [&](const std::vector<int>& combo) {
        all_channel_combos.push_back(combo);
    });

    for (const auto& channels : all_channel_combos) {
        std::string combo_id;
        for(size_t j=0; j<channels.size(); ++j) combo_id += std::to_string(channels[j]) + (j == channels.size()-1 ? "" : "_");
        for (int cmf_len : search_space) {
            for (double cmf_buy : cmf_buy_thresholds) {
                for (double cmf_sell : cmf_sell_thresholds) {
                    all_tasks.push_back({combo_id, cmf_len, cmf_buy, cmf_sell});
                }
            }
        }
    }
    
    size_t total_tasks = all_tasks.size();
    unsigned int num_cores = std::thread::hardware_concurrency();
    std::cout << "🚀 Sistem: " << num_cores << " CPU çekirdeği" << std::endl;
    std::cout << "⚙️  " << num_cores << " worker kullanılacak" << std::endl;
    std::cout << "📦 Toplam Paralel Görev: " << total_tasks << std::endl;

    CheckpointData checkpoint = load_checkpoint();
    std::vector<Task> tasks_to_run;
    std::hash<std::string> str_hasher;
    for (const auto& task : all_tasks) {
        size_t task_hash = str_hasher(task.combo_id) ^ (std::hash<int>()(task.cmf_len) << 1) ^ (std::hash<double>()(task.cmf_buy_thresh) << 2) ^ (std::hash<double>()(task.cmf_sell_thresh) << 3);
        if (checkpoint.completed_hashes.find(task_hash) == checkpoint.completed_hashes.end()) {
            tasks_to_run.push_back(task);
        }
    }

    if (tasks_to_run.empty()) {
        std::cout << "✅ Tüm görevler tamamlanmış!" << std::endl;
        return 0;
    }

    std::cout << "⏳ Kalan: " << tasks_to_run.size() << std::endl;
    std::cout << "📊 İlerleme: " << std::fixed << std::setprecision(1) << (checkpoint.completed_hashes.size() * 100.0 / total_tasks) << "%\n" << std::endl;
    std::cout << "🔥 ULTRA-FAST işlem başlatılıyor...\n" << std::endl;

    BS::thread_pool pool(num_cores);
    std::vector<std::future<WorkerResult>> futures;
    
    auto shared_data = std::make_shared<const SharedData>(SharedData{
        prices_vec, {60, 100}, {0, 40}, 5
    });

    for (const auto& task : tasks_to_run) {
        futures.push_back(pool.submit(process_single_test_ultra_fast, task, shared_data));
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_checkpoint_time = start_time;
    size_t completed_count = 0;
    
    // CSV yazıcıları
    std::ofstream results_csv(RESULTS_CSV, std::ios::app);
    std::ofstream best_scores_csv(BEST_SCORES_CSV, std::ios::app);
    //... (CSV başlık yazma kodu eklenebilir)

    for (size_t i = 0; i < futures.size(); ++i) {
        WorkerResult res = futures[i].get();
        completed_count++;
        
        const auto& task = tasks_to_run[i];
        size_t task_hash = str_hasher(task.combo_id) ^ (std::hash<int>()(task.cmf_len) << 1) ^ (std::hash<double>()(task.cmf_buy_thresh) << 2) ^ (std::hash<double>()(task.cmf_sell_thresh) << 3);
        checkpoint.completed_hashes.insert(task_hash);

        if (res.best_local_result.has_value()) {
            if (res.best_local_result->metrics[7] > checkpoint.best_overall_score) {
                double old_best = checkpoint.best_overall_score;
                checkpoint.best_overall_score = res.best_local_result->metrics[7];
                checkpoint.best_overall_result = res.best_local_result;
                
                std::cout << "\n🏆 [" << checkpoint.completed_hashes.size() << "/" << total_tasks << "] YENİ EN İYİ! " 
                          << std::fixed << std::setprecision(4) << checkpoint.best_overall_score
                          << " (önceki: " << old_best << ")" << std::endl;
            }
        }
        
        if (i % 100 == 0) std::cout << "." << std::flush;
        
        auto current_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(current_time - last_checkpoint_time).count() > CHECKPOINT_INTERVAL) {
            save_checkpoint(checkpoint);
            last_checkpoint_time = current_time;
            double elapsed = std::chrono::duration<double>(current_time - start_time).count();
            double avg_time = elapsed / completed_count;
            double est_time = avg_time * (tasks_to_run.size() - completed_count);
            
            std::cout << "\n   💾 Checkpoint" << std::endl;
            std::cout << "   ⏱️  " << checkpoint.completed_hashes.size() << "/" << total_tasks << " ("
                      << std::fixed << std::setprecision(1) << (checkpoint.completed_hashes.size() * 100.0 / total_tasks) << "%)" << std::endl;
            std::cout << "   ⌛ Geçen: " << static_cast<int>(elapsed/60) << "dk | Kalan: ~" << static_cast<int>(est_time/60) << "dk" << std::endl;
        }
    }

    save_checkpoint(checkpoint);
    std::cout << "\n\n✅ OPTİMİZASYON TAMAMLANDI!" << std::endl;
    
    if (fs::exists(CHECKPOINT_FILE)) fs::remove(CHECKPOINT_FILE);
    std::cout << "\n🗑️  Checkpoint temizlendi" << std::endl;

    if (checkpoint.best_overall_result.has_value()) {
        const auto& best = *checkpoint.best_overall_result;
        const auto& metrics = best.metrics;
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "🏆 EN İYİ STRATEJİ" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "📊 Skor: " << metrics[7] << std::endl;
        // ... (diğer en iyi sonuç detayları)
    }

    return 0;
}