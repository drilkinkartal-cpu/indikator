import pandas as pd
import numpy as np
import os
import json
import h5py
from itertools import combinations_with_replacement
import warnings
import numba
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
import gc
import pickle

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CHECKPOINT_FILE = 'checkpoint_ultra_fast.dat'
RESULTS_CSV = 'optimization_best_results_cmf.csv'
BEST_SCORES_CSV = 'optimization_all_best_scores_cmf.csv'

PRECOMP_DIR = 'precomputed_data'
CMF_CACHE_FILE = os.path.join(PRECOMP_DIR, 'cmf_cache.h5')
CHANNEL_COMBOS_FILE = os.path.join(PRECOMP_DIR, 'channel_combos.h5')  # YENİ!
METADATA_FILE = os.path.join(PRECOMP_DIR, 'metadata.json')

SCORE_THRESHOLD = 0.98
CHECKPOINT_INTERVAL = 120
CSV_BATCH_SIZE = 100

if __name__ == "__main__":
    NUM_CORES = mp.cpu_count()
    print(f"🚀 Sistem: {NUM_CORES} CPU çekirdeği")
    print(f"⚙️  {NUM_CORES} worker kullanılacak")
    
    mem = psutil.virtual_memory()
    print(f"💾 RAM: {mem.available / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB")

os.makedirs(PRECOMP_DIR, exist_ok=True)

# ==================== NUMBA FUNCTIONS ====================

@numba.jit(nopython=True, fastmath=True)
def calculate_cmf(high, low, close, volume, length):
    n = len(close)
    cmf_values = np.zeros(n, dtype=np.float64)
    
    for i in range(length, n):
        mf_volume_sum = 0.0
        volume_sum = 0.0
        
        for j in range(i - length + 1, i + 1):
            if j >= n:
                continue
                
            hl_range = high[j] - low[j]
            if hl_range > 1e-9:
                mf_multiplier = ((close[j] - low[j]) - (high[j] - close[j])) / hl_range
                mf_volume_sum += mf_multiplier * volume[j]
            
            volume_sum += volume[j]
        
        if volume_sum > 1e-9:
            cmf_values[i] = mf_volume_sum / volume_sum
    
    return cmf_values

@numba.jit(nopython=True, parallel=False, fastmath=True)
def calculate_pine_script_scores_correct(prices, length, upper_mult, lower_mult):
    n = len(prices)
    scores = np.full(n, 50.0, dtype=np.float64)
    x = np.arange(1, length + 1, dtype=np.float64)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x * x)
    reg_denominator = length * sum_x2 - sum_x * sum_x
    mean_x = np.mean(x)
    
    for i in range(length, n):
        price_window = prices[i-length:i]
        log_window = np.log(price_window)
        sum_y = np.sum(log_window)
        sum_xy = np.sum(x * log_window)
        
        slope = (length * sum_xy - sum_x * sum_y) / reg_denominator if reg_denominator != 0 else 0.0
        intercept = np.mean(log_window) - slope * mean_x
        
        std_dev_acc = 0.0
        for j in range(length):
            residual = log_window[j] - (intercept + slope * x[j])
            std_dev_acc += residual * residual
        std_dev = np.sqrt(std_dev_acc / length)
        
        log_middle = intercept + slope
        log_upper = log_middle + upper_mult * std_dev
        log_lower = log_middle - lower_mult * std_dev
        log_current_price = log_window[-1]
        
        if log_current_price >= log_middle:
            denominator = log_upper - log_middle
            score = 50.0 + (50.0 * (log_current_price - log_middle) / denominator) if denominator > 1e-9 else 50.0
        else:
            denominator = log_middle - log_lower
            score = 50.0 * (log_current_price - log_lower) / denominator if denominator > 1e-9 else 50.0
        
        scores[i] = min(max(score, 0.0), 100.0)
    
    return scores

@numba.jit(nopython=True, fastmath=True)
def run_backtest_with_cmf(scores, cmf_values, prices, buy_threshold, sell_threshold, 
                          cmf_buy_threshold, cmf_sell_threshold, initial_capital):
    n = len(scores)
    capital = initial_capital
    position_price = 0.0
    position_entry_bar = 0
    
    max_trades = n // 2
    trade_profits = np.zeros(max_trades, dtype=np.float64)
    trade_profit_pcts = np.zeros(max_trades, dtype=np.float64)
    trade_durations = np.zeros(max_trades, dtype=np.int64)
    
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    trade_count = 0
    
    final_portfolio_value = initial_capital
    
    cmf_buy_armed = False
    cmf_sell_armed = False
    
    for i in range(1, n):
        current_price = prices[i]
        current_cmf = cmf_values[i]
        current_score = scores[i]
        prev_score = scores[i-1]

        if position_price == 0.0:
            if current_score <= sell_threshold:
                if current_cmf <= cmf_buy_threshold:
                    cmf_buy_armed = True
            
            if cmf_buy_armed and prev_score <= sell_threshold and current_score > sell_threshold:
                position_price = current_price
                position_entry_bar = i
                cmf_buy_armed = False
                cmf_sell_armed = False
            
            if prev_score >= buy_threshold and current_score < buy_threshold:
                cmf_buy_armed = False
        else:
            if current_score >= buy_threshold:
                if current_cmf >= cmf_sell_threshold:
                    cmf_sell_armed = True

            if cmf_sell_armed and prev_score >= buy_threshold and current_score < buy_threshold:
                profit = current_price - position_price
                profit_pct = (profit / position_price) * 100.0
                trade_duration = i - position_entry_bar
                
                capital += profit
                trade_profits[trade_count] = profit
                trade_profit_pcts[trade_count] = profit_pct
                trade_durations[trade_count] = trade_duration
                trade_count += 1
                
                if profit > 0.0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    if consecutive_wins > max_consecutive_wins:
                        max_consecutive_wins = consecutive_wins
                elif profit < 0.0:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    if consecutive_losses > max_consecutive_losses:
                        max_consecutive_losses = consecutive_losses
                else:
                    consecutive_wins = 0
                    consecutive_losses = 0
                
                position_price = 0.0
                cmf_sell_armed = False
                cmf_buy_armed = False
            
            if prev_score <= sell_threshold and current_score > sell_threshold:
                cmf_sell_armed = False
        
        if position_price > 0.0:
            final_portfolio_value = capital + (current_price - position_price)
        else:
            final_portfolio_value = capital
    
    return (trade_profits[:trade_count], trade_profit_pcts[:trade_count], 
            trade_durations[:trade_count], final_portfolio_value,
            consecutive_wins, consecutive_losses, max_consecutive_wins, max_consecutive_losses)

@numba.jit(nopython=True, fastmath=True)
def calculate_metrics_fast(trade_profits, trade_profit_pcts, trade_durations, 
                           final_value, initial_capital, max_consec_wins, max_consec_losses):
    
    if len(trade_profits) == 0:
        return np.zeros(17, dtype=np.float64)
    
    total_trades = len(trade_profits)
    wins = np.sum(trade_profits > 0)
    win_rate = (wins / total_trades) * 100.0
    
    gross_profit = np.sum(trade_profits[trade_profits > 0])
    gross_loss = np.abs(np.sum(trade_profits[trade_profits < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 1000.0
    
    # Max drawdown - Numba uyumlu versiyon
    cumsum = np.cumsum(trade_profits)
    running_max = np.empty_like(cumsum)
    running_max[0] = cumsum[0]
    for i in range(1, len(cumsum)):
        running_max[i] = max(running_max[i-1], cumsum[i])
    
    drawdowns = cumsum - running_max
    max_drawdown = (np.min(drawdowns) / initial_capital) * 100.0 if len(drawdowns) > 0 else 0.0
    
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100.0
    
    if total_trades > 1:
        trade_returns = trade_profit_pcts
        sharpe_ratio = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(total_trades) if np.std(trade_returns) != 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    losing_returns = trade_profit_pcts[trade_profit_pcts < 0]
    if len(losing_returns) > 0:
        downside_std = np.std(losing_returns)
        sortino_ratio = (np.mean(trade_profit_pcts) / downside_std) * np.sqrt(total_trades) if downside_std != 0 else 0.0
    else:
        sortino_ratio = sharpe_ratio
    
    calmar_ratio = abs(total_return_pct / max_drawdown) if max_drawdown != 0 else 0.0
    
    winning_trades = trade_profit_pcts[trade_profit_pcts > 0]
    losing_trades = trade_profit_pcts[trade_profit_pcts < 0]
    
    avg_profit_pct = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    avg_loss_pct = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    profit_loss_ratio = abs(avg_profit_pct / avg_loss_pct) if avg_loss_pct != 0 else 0.0
    
    recovery_factor = abs(total_return_pct / max_drawdown) if max_drawdown != 0 else 0.0
    expectancy = (win_rate/100.0 * avg_profit_pct) + ((1.0 - win_rate/100.0) * avg_loss_pct)
    avg_trade_duration = np.mean(trade_durations) if len(trade_durations) > 0 else 0.0
    
    if profit_factor >= 3.0:
        pf_score = 1.0
    elif profit_factor >= 2.0:
        pf_score = 0.8 + (profit_factor - 2.0) * 0.2
    elif profit_factor >= 1.5:
        pf_score = 0.5 + (profit_factor - 1.5) * 0.6
    else:
        pf_score = max(0.0, profit_factor / 1.5 * 0.5)
    
    if sharpe_ratio >= 2.5:
        sharpe_score = 1.0
    elif sharpe_ratio >= 1.5:
        sharpe_score = 0.8 + (sharpe_ratio - 1.5) * 0.2
    elif sharpe_ratio >= 1.0:
        sharpe_score = 0.5 + (sharpe_ratio - 1.0) * 0.6
    elif sharpe_ratio > 0:
        sharpe_score = sharpe_ratio / 1.0 * 0.5
    else:
        sharpe_score = 0.0
    
    if sortino_ratio >= 3.0:
        sortino_score = 1.0
    elif sortino_ratio >= 2.0:
        sortino_score = 0.8 + (sortino_ratio - 2.0) * 0.2
    elif sortino_ratio >= 1.0:
        sortino_score = 0.5 + (sortino_ratio - 1.0) * 0.6
    elif sortino_ratio > 0:
        sortino_score = sortino_ratio / 1.0 * 0.5
    else:
        sortino_score = 0.0
    
    if 50.0 <= win_rate <= 70.0:
        win_rate_score = 1.0
    elif win_rate < 50.0:
        win_rate_score = win_rate / 50.0 * 0.8
    else:
        win_rate_score = max(0.5, 1.0 - (win_rate - 70.0) / 30.0 * 0.5)
    
    if max_drawdown >= -5.0:
        drawdown_score = 1.0
    elif max_drawdown >= -15.0:
        drawdown_score = 0.7 + (max_drawdown + 15.0) / 10.0 * 0.3
    elif max_drawdown >= -30.0:
        drawdown_score = 0.3 + (max_drawdown + 30.0) / 15.0 * 0.4
    else:
        drawdown_score = max(0.0, 0.3 * (1.0 + max_drawdown / 100.0))
    
    if profit_loss_ratio >= 3.0:
        pl_ratio_score = 1.0
    elif profit_loss_ratio >= 2.0:
        pl_ratio_score = 0.8 + (profit_loss_ratio - 2.0) * 0.2
    elif profit_loss_ratio >= 1.5:
        pl_ratio_score = 0.5 + (profit_loss_ratio - 1.5) * 0.6
    else:
        pl_ratio_score = max(0.0, profit_loss_ratio / 1.5 * 0.5)
    
    if calmar_ratio >= 3.0:
        calmar_score = 1.0
    elif calmar_ratio >= 2.0:
        calmar_score = 0.8 + (calmar_ratio - 2.0) * 0.2
    elif calmar_ratio >= 1.0:
        calmar_score = 0.5 + (calmar_ratio - 1.0) * 0.6
    else:
        calmar_score = max(0.0, calmar_ratio / 1.0 * 0.5)
    
    if total_trades < 10:
        trade_count_score = total_trades / 10.0 * 0.3
    elif total_trades < 30:
        trade_count_score = 0.3 + (total_trades - 10) / 20.0 * 0.4
    elif 30 <= total_trades <= 300:
        trade_count_score = 1.0
    elif total_trades <= 500:
        trade_count_score = 1.0 - (total_trades - 300) / 200.0 * 0.3
    else:
        trade_count_score = max(0.3, 0.7 - (total_trades - 500) / 500.0 * 0.4)
    
    if max_consec_losses <= 3:
        consistency_score = 1.0
    elif max_consec_losses <= 5:
        consistency_score = 0.8
    elif max_consec_losses <= 7:
        consistency_score = 0.6
    elif max_consec_losses <= 10:
        consistency_score = 0.4
    else:
        consistency_score = max(0.2, 0.4 - (max_consec_losses - 10) / 20.0 * 0.2)
    
    quality_score = (
        pf_score * 0.10 +
        sharpe_score * 0.20 +
        sortino_score * 0.15 +
        drawdown_score * 0.20 +
        pl_ratio_score * 0.03 +
        calmar_score * 0.25 +
        win_rate_score * 0.01 +
        trade_count_score * 0.05 +
        consistency_score * 0.01
    )
    
    result = np.zeros(17, dtype=np.float64)
    result[0] = float(total_trades)
    result[1] = win_rate
    result[2] = profit_factor
    result[3] = max_drawdown
    result[4] = sharpe_ratio
    result[5] = sortino_ratio
    result[6] = calmar_ratio
    result[7] = quality_score
    result[8] = avg_profit_pct
    result[9] = avg_loss_pct
    result[10] = profit_loss_ratio
    result[11] = recovery_factor
    result[12] = expectancy
    result[13] = float(max_consec_losses)
    result[14] = float(max_consec_wins)
    result[15] = avg_trade_duration
    result[16] = total_return_pct
    
    return result
    
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100.0
    
    if total_trades > 1:
        trade_returns = trade_profit_pcts
        sharpe_ratio = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(total_trades) if np.std(trade_returns) != 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    losing_returns = trade_profit_pcts[trade_profit_pcts < 0]
    if len(losing_returns) > 0:
        downside_std = np.std(losing_returns)
        sortino_ratio = (np.mean(trade_profit_pcts) / downside_std) * np.sqrt(total_trades) if downside_std != 0 else 0.0
    else:
        sortino_ratio = sharpe_ratio
    
    calmar_ratio = abs(total_return_pct / max_drawdown) if max_drawdown != 0 else 0.0
    
    winning_trades = trade_profit_pcts[trade_profit_pcts > 0]
    losing_trades = trade_profit_pcts[trade_profit_pcts < 0]
    
    avg_profit_pct = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    avg_loss_pct = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    profit_loss_ratio = abs(avg_profit_pct / avg_loss_pct) if avg_loss_pct != 0 else 0.0
    
    recovery_factor = abs(total_return_pct / max_drawdown) if max_drawdown != 0 else 0.0
    expectancy = (win_rate/100.0 * avg_profit_pct) + ((1.0 - win_rate/100.0) * avg_loss_pct)
    avg_trade_duration = np.mean(trade_durations) if len(trade_durations) > 0 else 0.0
    
    if profit_factor >= 3.0:
        pf_score = 1.0
    elif profit_factor >= 2.0:
        pf_score = 0.8 + (profit_factor - 2.0) * 0.2
    elif profit_factor >= 1.5:
        pf_score = 0.5 + (profit_factor - 1.5) * 0.6
    else:
        pf_score = max(0.0, profit_factor / 1.5 * 0.5)
    
    if sharpe_ratio >= 2.5:
        sharpe_score = 1.0
    elif sharpe_ratio >= 1.5:
        sharpe_score = 0.8 + (sharpe_ratio - 1.5) * 0.2
    elif sharpe_ratio >= 1.0:
        sharpe_score = 0.5 + (sharpe_ratio - 1.0) * 0.6
    elif sharpe_ratio > 0:
        sharpe_score = sharpe_ratio / 1.0 * 0.5
    else:
        sharpe_score = 0.0
    
    if sortino_ratio >= 3.0:
        sortino_score = 1.0
    elif sortino_ratio >= 2.0:
        sortino_score = 0.8 + (sortino_ratio - 2.0) * 0.2
    elif sortino_ratio >= 1.0:
        sortino_score = 0.5 + (sortino_ratio - 1.0) * 0.6
    elif sortino_ratio > 0:
        sortino_score = sortino_ratio / 1.0 * 0.5
    else:
        sortino_score = 0.0
    
    if 50.0 <= win_rate <= 70.0:
        win_rate_score = 1.0
    elif win_rate < 50.0:
        win_rate_score = win_rate / 50.0 * 0.8
    else:
        win_rate_score = max(0.5, 1.0 - (win_rate - 70.0) / 30.0 * 0.5)
    
    if max_drawdown >= -5.0:
        drawdown_score = 1.0
    elif max_drawdown >= -15.0:
        drawdown_score = 0.7 + (max_drawdown + 15.0) / 10.0 * 0.3
    elif max_drawdown >= -30.0:
        drawdown_score = 0.3 + (max_drawdown + 30.0) / 15.0 * 0.4
    else:
        drawdown_score = max(0.0, 0.3 * (1.0 + max_drawdown / 100.0))
    
    if profit_loss_ratio >= 3.0:
        pl_ratio_score = 1.0
    elif profit_loss_ratio >= 2.0:
        pl_ratio_score = 0.8 + (profit_loss_ratio - 2.0) * 0.2
    elif profit_loss_ratio >= 1.5:
        pl_ratio_score = 0.5 + (profit_loss_ratio - 1.5) * 0.6
    else:
        pl_ratio_score = max(0.0, profit_loss_ratio / 1.5 * 0.5)
    
    if calmar_ratio >= 3.0:
        calmar_score = 1.0
    elif calmar_ratio >= 2.0:
        calmar_score = 0.8 + (calmar_ratio - 2.0) * 0.2
    elif calmar_ratio >= 1.0:
        calmar_score = 0.5 + (calmar_ratio - 1.0) * 0.6
    else:
        calmar_score = max(0.0, calmar_ratio / 1.0 * 0.5)
    
    if total_trades < 10:
        trade_count_score = total_trades / 10.0 * 0.3
    elif total_trades < 30:
        trade_count_score = 0.3 + (total_trades - 10) / 20.0 * 0.4
    elif 30 <= total_trades <= 300:
        trade_count_score = 1.0
    elif total_trades <= 500:
        trade_count_score = 1.0 - (total_trades - 300) / 200.0 * 0.3
    else:
        trade_count_score = max(0.3, 0.7 - (total_trades - 500) / 500.0 * 0.4)
    
    if max_consec_losses <= 3:
        consistency_score = 1.0
    elif max_consec_losses <= 5:
        consistency_score = 0.8
    elif max_consec_losses <= 7:
        consistency_score = 0.6
    elif max_consec_losses <= 10:
        consistency_score = 0.4
    else:
        consistency_score = max(0.2, 0.4 - (max_consec_losses - 10) / 20.0 * 0.2)
    
    quality_score = (
        pf_score * 0.10 +
        sharpe_score * 0.20 +
        sortino_score * 0.15 +
        drawdown_score * 0.20 +
        pl_ratio_score * 0.03 +
        calmar_score * 0.25 +
        win_rate_score * 0.01 +
        trade_count_score * 0.05 +
        consistency_score * 0.01
    )
    
    result = np.zeros(17, dtype=np.float64)
    result[0] = float(total_trades)
    result[1] = win_rate
    result[2] = profit_factor
    result[3] = max_drawdown
    result[4] = sharpe_ratio
    result[5] = sortino_ratio
    result[6] = calmar_ratio
    result[7] = quality_score
    result[8] = avg_profit_pct
    result[9] = avg_loss_pct
    result[10] = profit_loss_ratio
    result[11] = recovery_factor
    result[12] = expectancy
    result[13] = float(max_consec_losses)
    result[14] = float(max_consec_wins)
    result[15] = avg_trade_duration
    result[16] = total_return_pct
    
    return result

def warmup_numba():
    print("🔥 Numba derleniyor...")
    
    size = 500
    dummy_high = np.random.random(size) + 100
    dummy_low = dummy_high * 0.99
    dummy_close = (dummy_high + dummy_low) / 2
    dummy_volume = np.random.random(size) * 1000
    dummy_prices = dummy_close.copy()
    dummy_scores = np.random.random(size) * 100
    dummy_cmf = np.random.random(size) * 0.2 - 0.1
    
    _ = calculate_cmf(dummy_high, dummy_low, dummy_close, dummy_volume, 20)
    _ = calculate_pine_script_scores_correct(dummy_prices, 20, 5.0, 5.0)
    result_tuple = run_backtest_with_cmf(dummy_scores, dummy_cmf, dummy_prices, 
                                         80.0, 20.0, -0.1, 0.1, 10000.0)
    _ = calculate_metrics_fast(result_tuple[0], result_tuple[1], result_tuple[2],
                               result_tuple[3], 10000.0, 1, 1)
    
    print("✅ Numba hazır!\n")

# ==================== PRE-COMPUTATION ENGINE ====================

class PreComputationEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.metadata = {}
        
    def load_data(self):
        print(f"📂 Veri yükleniyor: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"✅ {len(self.data)} bar yüklendi")
        self.metadata['n_bars'] = len(self.data)
        self.metadata['data_path'] = self.data_path
        
    def precompute_all_cmf(self, cmf_lengths):
        print("\n" + "="*70)
        print("📊 CMF ÖN-HESAPLAMA")
        print("="*70)
        
        high = self.data['high'].values
        low = self.data['low'].values
        close = self.data['close'].values
        volume = self.data['volume'].values
        
        start_time = datetime.now()
        
        with h5py.File(CMF_CACHE_FILE, 'w') as hf:
            cmf_group = hf.create_group('cmf_values')
            
            for idx, cmf_len in enumerate(cmf_lengths):
                print(f"⚙️  [{idx+1}/{len(cmf_lengths)}] CMF Length={cmf_len}...", end="", flush=True)
                
                cmf_array = calculate_cmf(high, low, close, volume, cmf_len)
                # SIKIŞTIRILMADAN kaydet - hız için!
                cmf_group.create_dataset(f'length_{cmf_len}', data=cmf_array, compression=None)
                print(" ✓")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        file_size_mb = os.path.getsize(CMF_CACHE_FILE) / (1024 * 1024)
        print(f"\n✅ Tamamlandı! Süre: {elapsed:.1f}s | Boyut: {file_size_mb:.1f} MB")
        
        self.metadata['cmf_lengths'] = cmf_lengths
        
    def precompute_all_channel_combos(self, channel_search_space, num_channels=4, std_dev_mult=5.0):
        """TÜM KANAL KOMBİNASYONLARINI ÖN-HESAPLA VE KAYDET"""
        print("\n" + "="*70)
        print("📊 KANAL KOMBİNASYONLARI ÖN-HESAPLAMA (SÜPER HIZ!)")
        print("="*70)
        
        close_prices = self.data['close'].values
        all_channel_combos = list(combinations_with_replacement(channel_search_space, num_channels))
        
        print(f"🔢 Toplam kombinasyon: {len(all_channel_combos):,}")
        print(f"⚠️  Bu adım biraz uzun sürebilir ama sonrası ÇOK HIZLI olacak!")
        
        start_time = datetime.now()
        
        # Önce individual skorları hesapla
        print("\n1️⃣  Individual kanal skorları hesaplanıyor...")
        individual_scores = {}
        unique_lengths = sorted(set(channel_search_space))
        
        for idx, length in enumerate(unique_lengths):
            print(f"   [{idx+1}/{len(unique_lengths)}] Length={length}...", end="", flush=True)
            scores = calculate_pine_script_scores_correct(close_prices, length, std_dev_mult, std_dev_mult)
            individual_scores[length] = scores
            print(" ✓")
        
        # Şimdi tüm kombinasyonları hesapla ve kaydet
        print(f"\n2️⃣  {len(all_channel_combos):,} kombinasyon hesaplanıyor ve kaydediliyor...")
        
        with h5py.File(CHANNEL_COMBOS_FILE, 'w') as hf:
            combos_group = hf.create_group('channel_combinations')
            
            for idx, channels in enumerate(all_channel_combos):
                if (idx + 1) % 10000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = (idx + 1) / elapsed
                    remaining = (len(all_channel_combos) - idx - 1) / rate
                    print(f"   [{idx+1}/{len(all_channel_combos)}] - {rate:.0f} combo/s - Kalan: ~{remaining/60:.1f}dk")
                
                # Weighted score hesapla
                all_scores = [individual_scores[length] for length in channels]
                all_scores_array = np.column_stack(all_scores)
                weights = np.array(channels, dtype=np.float64)
                weights /= np.sum(weights)
                
                channel_scores = np.dot(all_scores_array, weights)
                channel_scores = np.clip(channel_scores, 0.0, 100.0)
                
                # Kombinasyon ID'si
                combo_id = '_'.join(map(str, channels))
                
                # SIKIŞTIRILMADAN kaydet - maksimum hız!
                combos_group.create_dataset(combo_id, data=channel_scores, compression=None)
            
            # Metadata
            hf.attrs['num_combinations'] = len(all_channel_combos)
            hf.attrs['std_dev_mult'] = std_dev_mult
            hf.attrs['num_channels'] = num_channels
        
        elapsed = (datetime.now() - start_time).total_seconds()
        file_size_gb = os.path.getsize(CHANNEL_COMBOS_FILE) / (1024 * 1024 * 1024)
        
        print(f"\n✅ TÜM KOMBİNASYONLAR HAZIR!")
        print(f"⏱️  Süre: {elapsed/60:.1f} dakika")
        print(f"💾 Dosya boyutu: {file_size_gb:.2f} GB")
        print(f"🚀 Runtime'da HİÇ HESAPLAMA YAPILMAYACAK, sadece okuma!")
        
        self.metadata['channel_combinations_count'] = len(all_channel_combos)
        self.metadata['std_dev_mult'] = std_dev_mult
        self.metadata['num_channels'] = num_channels
        
    def save_metadata(self):
        self.metadata['timestamp'] = datetime.now().isoformat()
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)

# ==================== CHECKPOINT & CSV ====================

def save_checkpoint(completed_hashes, best_result, best_score):
    checkpoint_data = {
        'completed_hashes': completed_hashes,
        'best_overall_score': best_score,
        'best_overall_result': best_result
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                data = pickle.load(f)
            completed_hashes = data.get('completed_hashes', set())
            best_score = data.get('best_overall_score', -1)
            best_result = data.get('best_overall_result')
            print(f"📂 Checkpoint: {len(completed_hashes):,} tamamlanmış")
            return completed_hashes, best_score, best_result
        except:
            return set(), -1, None
    return set(), -1, None

class CSVBatchWriter:
    def __init__(self, filename, batch_size=100):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []
        self.file_exists = os.path.exists(filename)
        
    def add(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        
        if not self.file_exists:
            df.to_csv(self.filename, index=False, float_format='%.4f', mode='w')
            self.file_exists = True
        else:
            df.to_csv(self.filename, index=False, float_format='%.4f', mode='a', header=False)
        
        self.buffer = []

# ==================== ULTRA-FAST WORKER ====================

def process_single_test_ultra_fast(args):
    """HİÇ HESAPLAMA YOK - Sadece H5PY'den oku ve backtest yap!"""
    (combo_id, cmf_len, cmf_buy_thresh, cmf_sell_thresh,
     channel_file_path, cmf_file_path, prices,
     buy_threshold_range, sell_threshold_range, threshold_step) = args
    
    try:
        best_local_result = None
        best_local_score = -1
        good_results = []
        
        # Dosyaları aç (her worker kendi handle'ını kullanıyor)
        with h5py.File(channel_file_path, 'r') as channel_file, \
             h5py.File(cmf_file_path, 'r') as cmf_file:
            
            # SADECE OKU - hesaplama yok!
            channel_scores = channel_file[f'channel_combinations/{combo_id}'][:]
            cmf_values = cmf_file[f'cmf_values/length_{cmf_len}'][:]
            
            # Contiguous yap
            channel_scores = np.ascontiguousarray(channel_scores, dtype=np.float64)
            cmf_values = np.ascontiguousarray(cmf_values, dtype=np.float64)
            prices = np.ascontiguousarray(prices, dtype=np.float64)
            
            for buy_thresh in range(buy_threshold_range[0], buy_threshold_range[1] + 1, threshold_step):
                for sell_thresh in range(sell_threshold_range[0], sell_threshold_range[1] + 1, threshold_step):
                    if buy_thresh <= sell_thresh:
                        continue
                    
                    result_tuple = run_backtest_with_cmf(
                        channel_scores, cmf_values, prices, 
                        float(buy_thresh), float(sell_thresh),
                        float(cmf_buy_thresh), float(cmf_sell_thresh),
                        10000.0
                    )
                    
                    trade_profits = result_tuple[0]
                    trade_profit_pcts = result_tuple[1]
                    trade_durations = result_tuple[2]
                    final_portfolio_value = result_tuple[3]
                    max_consecutive_wins = result_tuple[6]
                    max_consecutive_losses = result_tuple[7]
                    
                    total_trades = len(trade_profits)
                    
                    if total_trades > 10:
                        metrics_array = calculate_metrics_fast(
                            trade_profits, trade_profit_pcts, trade_durations,
                            final_portfolio_value, 10000.0,
                            max_consecutive_wins, max_consecutive_losses
                        )
                        
                        quality_score = metrics_array[7]
                        
                        if quality_score >= SCORE_THRESHOLD:
                            channels_tuple = tuple(map(int, combo_id.split('_')))
                            result_info = {
                                'channels': channels_tuple,
                                'cmf_length': cmf_len,
                                'cmf_buy_threshold': cmf_buy_thresh,
                                'cmf_sell_threshold': cmf_sell_thresh,
                                'buy_threshold': buy_thresh,
                                'sell_threshold': sell_thresh,
                                'metrics': metrics_array
                            }
                            good_results.append(result_info)
                        
                        if quality_score > best_local_score:
                            best_local_score = quality_score
                            channels_tuple = tuple(map(int, combo_id.split('_')))
                            best_local_result = {
                                'channels': channels_tuple,
                                'cmf_length': cmf_len,
                                'cmf_buy_threshold': cmf_buy_thresh,
                                'cmf_sell_threshold': cmf_sell_thresh,
                                'buy_threshold': buy_thresh,
                                'sell_threshold': sell_thresh,
                                'metrics': metrics_array
                            }
        
        return best_local_result, good_results
        
    except Exception as e:
        return None, []

# ==================== MAIN OPTIMIZATION ====================

def run_ultra_fast_optimization(data_path='ETHBTC_5m_50000_bars.csv', force_recompute=False):
    print("\n" + "="*70)
    print("🚀 ULTRA-FAST OPTİMİZASYON - SIFIR HESAPLAMA MOD")
    print("="*70)
    
    channel_search_space = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 624, 688, 752, 816, 880, 944, 1008, 1072, 1136, 1264, 1392, 1520, 1648, 1776, 1904, 2032, 2160, 2288]
    
    cmf_lengths = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 624, 688, 752, 816, 880, 944, 1008, 1072, 1136, 1264, 1392, 1520, 1648, 1776, 1904, 2032, 2160, 2288]
    
    cmf_buy_thresholds = [-0.05, -0.10, -0.15, -0.20, -0.25]
    cmf_sell_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    num_channels = 4
    buy_threshold_range = (60, 100)
    sell_threshold_range = (0, 40)
    threshold_step = 5
    std_dev_mult = 5.0
    
    # PHASE 1: Pre-computation
    if force_recompute or not os.path.exists(CMF_CACHE_FILE) or not os.path.exists(CHANNEL_COMBOS_FILE):
        print("\n🔥 ÖN-HESAPLAMA GEREKLİ!")
        print("⚠️  Bu işlem bir kez yapılacak, sonrası ÇOK HIZLI!")
        
        precomp = PreComputationEngine(data_path)
        precomp.load_data()
        
        if force_recompute or not os.path.exists(CMF_CACHE_FILE):
            precomp.precompute_all_cmf(cmf_lengths)
        
        if force_recompute or not os.path.exists(CHANNEL_COMBOS_FILE):
            precomp.precompute_all_channel_combos(channel_search_space, num_channels, std_dev_mult)
        
        precomp.save_metadata()
        del precomp
        gc.collect()
    else:
        print("\n✅ Tüm ön-hesaplamalar mevcut - ULTRA HIZ MODU!")
    
    # PHASE 2: Ultra-Fast Optimization
    print("\n" + "="*70)
    print("🔥 ULTRA-FAST OPTİMİZASYON BAŞLIYOR")
    print("="*70)
    
    print("📂 Veri yükleniyor...")
    data = pd.read_csv(data_path)
    prices = data['close'].values.astype(np.float64)
    print(f"✅ {len(data)} bar yüklendi")
    
    warmup_numba()
    
    # Metadata oku
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    all_channel_combos = list(combinations_with_replacement(channel_search_space, num_channels))
    
    threshold_combinations = sum(1 for b in range(buy_threshold_range[0], buy_threshold_range[1] + 1, threshold_step)
                                   for s in range(sell_threshold_range[0], sell_threshold_range[1] + 1, threshold_step) if b > s)
    
    total_tasks = (len(all_channel_combos) * len(cmf_lengths) * 
                   len(cmf_buy_thresholds) * len(cmf_sell_thresholds))
    
    total_combinations = total_tasks * threshold_combinations
    
    print(f"🔢 Kanal Kombinasyonları: {len(all_channel_combos):,}")
    print(f"🔢 CMF Kombinasyonları: {len(cmf_lengths)} × {len(cmf_buy_thresholds)} × {len(cmf_sell_thresholds)}")
    print(f"📦 Toplam Paralel Görev: {total_tasks:,}")
    print(f"🎯 TOPLAM TEST: {total_combinations:,}")
    print(f"⚡ Worker: {NUM_CORES}")
    print(f"🚀 MOD: SIFIR HESAPLAMA - Sadece H5PY okuma!")
    print("="*70 + "\n")
    
    completed_hashes, best_overall_score, best_overall_result = load_checkpoint()
    
    total_saved_results = 0
    if os.path.exists(RESULTS_CSV):
        try:
            existing_df = pd.read_csv(RESULTS_CSV)
            total_saved_results = len(existing_df)
            print(f"📊 CSV'de {total_saved_results} kayıt")
        except:
            pass
    
    total_best_score_saves = 0
    if os.path.exists(BEST_SCORES_CSV):
        try:
            best_scores_df = pd.read_csv(BEST_SCORES_CSV)
            total_best_score_saves = len(best_scores_df)
            print(f"🌟 En iyi skorlar: {total_best_score_saves} kayıt")
        except:
            pass
    print()
    
    results_writer = CSVBatchWriter(RESULTS_CSV, CSV_BATCH_SIZE)
    best_scores_writer = CSVBatchWriter(BEST_SCORES_CSV, CSV_BATCH_SIZE)
    
    def task_generator():
        """Görev üreteci - sadece ID'ler, hesaplama yok!"""
        for channels in all_channel_combos:
            combo_id = '_'.join(map(str, channels))
            
            for cmf_len in cmf_lengths:
                for cmf_buy in cmf_buy_thresholds:
                    for cmf_sell in cmf_sell_thresholds:
                        test_hash = hash((combo_id, cmf_len, cmf_buy, cmf_sell))
                        
                        if test_hash not in completed_hashes:
                            # Sadece ID'ler ve dosya yolları - veri yok!
                            yield (combo_id, cmf_len, cmf_buy, cmf_sell,
                                  CHANNEL_COMBOS_FILE, CMF_CACHE_FILE, prices.copy(),
                                  buy_threshold_range, sell_threshold_range, threshold_step)
    
    tasks_to_run_gen = task_generator()
    remaining_tasks_count = total_tasks - len(completed_hashes)
    
    if remaining_tasks_count <= 0:
        print("✅ Tüm görevler tamamlanmış!")
        return
    
    print(f"⏳ Kalan: {remaining_tasks_count:,}")
    print(f"📊 İlerleme: {(len(completed_hashes) * 100 / total_tasks):.1f}%\n")
    print("🔥 ULTRA-FAST işlem başlatılıyor...\n")
    
    start_time = datetime.now()
    test_counter = len(completed_hashes)
    completed_in_this_session = 0
    new_saves_in_session = 0
    new_best_score_saves = 0
    last_checkpoint_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        future_to_task = {}
        
        # İlk batch
        for _ in range(NUM_CORES * 3):
            try:
                task = next(tasks_to_run_gen)
                future = executor.submit(process_single_test_ultra_fast, task)
                future_to_task[future] = task
            except StopIteration:
                break
        
        while future_to_task:
            done_futures = as_completed(future_to_task)
            future = next(done_futures)
            
            task = future_to_task.pop(future)
            combo_id, cmf_len, cmf_buy, cmf_sell = task[0], task[1], task[2], task[3]
            
            try:
                result, good_results = future.result()
                
                test_hash = hash((combo_id, cmf_len, cmf_buy, cmf_sell))
                completed_hashes.add(test_hash)
                
                test_counter += 1
                completed_in_this_session += 1
                
                if good_results:
                    for good_result in good_results:
                        metrics = good_result['metrics']
                        result_row = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'quality_score': metrics[7],
                            'channels': str(good_result['channels']),
                            'std_dev_mult': std_dev_mult,
                            'cmf_length': good_result['cmf_length'],
                            'cmf_buy_threshold': good_result['cmf_buy_threshold'],
                            'cmf_sell_threshold': good_result['cmf_sell_threshold'],
                            'buy_threshold': good_result['buy_threshold'],
                            'sell_threshold': good_result['sell_threshold'],
                            'total_return_pct': metrics[16],
                            'total_trades': int(metrics[0]),
                            'win_rate_pct': metrics[1],
                            'profit_factor': metrics[2],
                            'sharpe_ratio': metrics[4],
                            'sortino_ratio': metrics[5],
                            'calmar_ratio': metrics[6],
                            'max_drawdown_pct': metrics[3],
                            'avg_profit_pct': metrics[8],
                            'avg_loss_pct': metrics[9],
                            'profit_loss_ratio': metrics[10],
                            'max_consecutive_losses': int(metrics[13]),
                            'max_consecutive_wins': int(metrics[14]),
                            'avg_trade_duration': metrics[15],
                            'expectancy': metrics[12]
                        }
                        results_writer.add(result_row)
                        new_saves_in_session += 1
                        total_saved_results += 1
                
                if result is not None and 'metrics' in result:
                    quality_score = result['metrics'][7]
                    
                    if quality_score > best_overall_score:
                        old_best_score = best_overall_score
                        best_overall_score = quality_score
                        best_overall_result = result
                        
                        metrics = result['metrics']
                        
                        best_score_row = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'quality_score': metrics[7],
                            'channels': str(result['channels']),
                            'std_dev_mult': std_dev_mult,
                            'cmf_length': result['cmf_length'],
                            'cmf_buy_threshold': result['cmf_buy_threshold'],
                            'cmf_sell_threshold': result['cmf_sell_threshold'],
                            'buy_threshold': result['buy_threshold'],
                            'sell_threshold': result['sell_threshold'],
                            'total_return_pct': metrics[16],
                            'total_trades': int(metrics[0]),
                            'win_rate_pct': metrics[1],
                            'profit_factor': metrics[2],
                            'sharpe_ratio': metrics[4],
                            'sortino_ratio': metrics[5],
                            'calmar_ratio': metrics[6],
                            'max_drawdown_pct': metrics[3],
                            'avg_profit_pct': metrics[8],
                            'avg_loss_pct': metrics[9],
                            'profit_loss_ratio': metrics[10],
                            'max_consecutive_losses': int(metrics[13]),
                            'max_consecutive_wins': int(metrics[14]),
                            'avg_trade_duration': metrics[15],
                            'expectancy': metrics[12]
                        }
                        best_scores_writer.add(best_score_row)
                        new_best_score_saves += 1
                        total_best_score_saves += 1
                        
                        print(f"\n🏆 [{test_counter}/{total_tasks}] YENİ EN İYİ! {best_overall_score:.4f} (önceki: {old_best_score:.4f})")
                        print(f"   📍 {result['channels']} | CMF: {cmf_len}, {cmf_buy:.2f}, {cmf_sell:.2f}")
                        print(f"   💰 Getiri: {metrics[16]:.2f}% | İşlem: {int(metrics[0])}")
                
                current_time = datetime.now()
                if (current_time - last_checkpoint_time).total_seconds() > CHECKPOINT_INTERVAL:
                    results_writer.flush()
                    best_scores_writer.flush()
                    
                    save_checkpoint(completed_hashes, best_overall_result, best_overall_score)
                    last_checkpoint_time = current_time
                    
                    elapsed = (current_time - start_time).total_seconds()
                    
                    if completed_in_this_session > 0:
                        avg_time = elapsed / completed_in_this_session
                        remaining = remaining_tasks_count - completed_in_this_session
                        est_time = avg_time * remaining
                        
                        print(f"\n   💾 Checkpoint")
                        print(f"   ⏱️  {test_counter}/{total_tasks} ({test_counter*100/total_tasks:.1f}%)")
                        print(f"   ⌛ Geçen: {elapsed/60:.1f}dk | Kalan: ~{est_time/60:.0f}dk")
                        print(f"   ⚡ Hız: {completed_in_this_session/(elapsed/60):.1f} görev/dk 🚀")
                        print(f"   📊 Kayıt: {total_saved_results:,} | Best: {total_best_score_saves:,}\n")
                
                if completed_in_this_session % 500 == 0:
                    print(f".", end="", flush=True)
            
            except Exception as e:
                pass
            
            try:
                new_task = next(tasks_to_run_gen)
                new_future = executor.submit(process_single_test_ultra_fast, new_task)
                future_to_task[new_future] = new_task
            except StopIteration:
                pass
    
    # Cleanup
    results_writer.flush()
    best_scores_writer.flush()
    save_checkpoint(completed_hashes, best_overall_result, best_overall_score)
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n\n" + "="*70)
    print("✅ OPTİMİZASYON TAMAMLANDI!")
    print("="*70)
    print(f"⏱️  Süre: {total_elapsed/60:.1f} dk ({total_elapsed/3600:.2f} saat)")
    print(f"📊 İşlenen: {completed_in_this_session:,}")
    print(f"💾 Kaydedilen: {new_saves_in_session:,}")
    print(f"🌟 Yeni en iyi: {new_best_score_saves:,}")
    if total_elapsed > 0:
        print(f"⚡ Ortalama Hız: {completed_in_this_session/(total_elapsed/60):.1f} görev/dk 🚀🚀🚀")
    
    if os.path.exists(RESULTS_CSV):
        results_df = pd.read_csv(RESULTS_CSV)
        results_df = results_df.sort_values('quality_score', ascending=False).reset_index(drop=True)
        results_df.to_csv(RESULTS_CSV, index=False, float_format='%.4f')
        
        print(f"\n📈 CSV: {len(results_df)} sonuç")
        print(f"\n📊 İSTATİSTİKLER:")
        print(f"   • En yüksek: {results_df['quality_score'].max():.4f}")
        print(f"   • Ortalama: {results_df['quality_score'].mean():.4f}")
        print(f"   • 0.80+: {len(results_df[results_df['quality_score'] >= 0.80])}")
        print(f"   • 0.85+: {len(results_df[results_df['quality_score'] >= 0.85])}")
        print(f"   • 0.90+: {len(results_df[results_df['quality_score'] >= 0.90])}")
        
        print("\n🏆 EN İYİ 5:")
        for idx, row in results_df.head(5).iterrows():
            print(f"   #{idx+1} Skor: {row['quality_score']:.4f} | {row['channels']} | Getiri: {row['total_return_pct']:.1f}%")
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"\n🗑️  Checkpoint temizlendi")
    
    if best_overall_result and 'metrics' in best_overall_result:
        res = best_overall_result
        metrics = res['metrics']
        print("\n" + "="*70)
        print("🏆 EN İYİ STRATEJİ")
        print("="*70)
        print(f"📊 Skor: {metrics[7]:.4f}")
        print(f"📈 Kanallar: {res['channels']}")
        print(f"⚙️  Eşik: {res['buy_threshold']}/{res['sell_threshold']}")
        print(f"📊 CMF: {res['cmf_length']}, {res['cmf_buy_threshold']}, {res['cmf_sell_threshold']}")
        print(f"💰 Getiri: {metrics[16]:.2f}% | İşlem: {int(metrics[0])} | WR: {metrics[1]:.1f}%")
        print("="*70)

# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Fast CMF Optimizer')
    parser.add_argument('--data', type=str, default='ETHBTC_5m_50000_bars.csv')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Tüm ön-hesaplamaları yeniden yap')
    parser.add_argument('--precompute-only', action='store_true',
                       help='Sadece ön-hesaplama yap, optimizasyon yapma')
    
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ '{args.data}' bulunamadı")
        exit(1)
    
    if args.precompute_only:
        print("\n🔥 SADECE ÖN-HESAPLAMA MODU")
        print("⚠️  Bu işlem ~30-60 dakika sürebilir ama sonrası ÇOK HIZLI olacak!\n")
        
        channel_search_space = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 624, 688, 752, 816, 880, 944, 1008, 1072, 1136, 1264, 1392, 1520, 1648, 1776, 1904, 2032, 2160, 2288]
        
        cmf_lengths = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 624, 688, 752, 816, 880, 944, 1008, 1072, 1136, 1264, 1392, 1520, 1648, 1776, 1904, 2032, 2160, 2288]
        
        precomp = PreComputationEngine(args.data)
        precomp.load_data()
        precomp.precompute_all_cmf(cmf_lengths)
        precomp.precompute_all_channel_combos(channel_search_space, num_channels=4, std_dev_mult=5.0)
        precomp.save_metadata()
        
        print("\n✅ Ön-hesaplama tamamlandı!")
        print("🚀 Artık optimizasyonu çalıştırabilirsiniz:")
        print(f"   python {__file__} --data {args.data}")
    else:
        run_ultra_fast_optimization(args.data, args.force_recompute)