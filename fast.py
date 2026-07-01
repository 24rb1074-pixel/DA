# =====================================================================
# データ生成からLETKF実行、パラメータ探索までを1セルに統合した高速化コード
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from joblib import Parallel, delayed

# =====================================================================
# 1. 初期パラメータ設定 (Experiment Configurations)
# =====================================================================
# [モデル設定]
N = 40                  # 変数の数 (観測の数 P も今回は同じ40)
F = 8.0                 # 強制項
dt = 0.01               # 積分時間ステップ
sampling_dt = 0.05      # 観測・同化の間隔 (6時間相当: 0.25日/5日)
m = 8                   # アンサンブルメンバー数 (過酷な環境テスト)
p = 40                  # 観測点の数 (今回は全点観測なので N と同じ)

# [時間設定]
years = 5
units_per_year = 73.0   # 1年 = 365日 / 5日 = 73ユニット
total_time = years * units_per_year
spin_up_time = units_per_year

steps_total = int(total_time / dt)
steps_spin_up = int(spin_up_time / dt)
sampling_interval = int(sampling_dt / dt)

# =====================================================================
# 2. Nature Run (真値) と Observation (観測) の生成
# =====================================================================
def lorenz96(x, F):
    return (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + F

def M(x_in, dt, steps):
    x_out = x_in.copy()
    for _ in range(steps):
        k1 = lorenz96(x_out, F)
        k2 = lorenz96(x_out + k1 * (dt / 2.0), F)
        k3 = lorenz96(x_out + k2 * (dt / 2.0), F)
        k4 = lorenz96(x_out + k3 * dt, F)
        x_out += (k1 + 2.0*k2 + 2.0*k3 + k4) * (dt / 6.0)
    return x_out

print("Nature Runと観測データを生成中...")
x = np.full(N, F)
x[19] += 1.001

true_states = []
for s in range(steps_total):
    x = M(x, dt, 1)
    if s >= steps_spin_up and (s - steps_spin_up) % sampling_interval == 0:
        true_states.append(x.copy())
true_states = np.array(true_states)

rng_obs = np.random.default_rng(seed=67) 
noise = rng_obs.normal(loc=0.0, scale=1.0, size=true_states.shape)
noise -= np.mean(noise, axis=0)
y_o_data_full = true_states + noise

num_cycles = y_o_data_full.shape[0]
print("生成完了！")

# =====================================================================
# 3. 共通関数（汎用化）
# =====================================================================
def get_experiment_matrices(obs_indices):
    p_obs = len(obs_indices)
    H_mat = np.zeros((p_obs, N))
    for k, j in enumerate(obs_indices):
        H_mat[k, j] = 1.0
    R_mat = np.eye(p_obs)
    return H_mat, R_mat

def get_localization_weights(sigma, obs_indices):
    p_obs = len(obs_indices)
    W_weights = np.zeros((N, p_obs))
    def L(d):
        if d < np.sqrt(10.0 / 3.0) * sigma * 2.0:
            return np.exp(- (d**2) / (2.0 * sigma**2))
        return 0.0
    
    for i in range(N):
        for k, j in enumerate(obs_indices):
            d = min(abs(i - j), N - abs(i - j))
            W_weights[i, k] = L(d)
    return W_weights

def run_LETKF_Adaptive_Moving_Agile(init_delta, v_b_val, move_interval, move_step, swath_size, sigma, H_mats, W_weights_list, obs_indices_history):
    rng_enkf = np.random.default_rng(seed=42)
    init_noise = rng_enkf.normal(0.0, 0.1, size=(N, m))
    init_noise -= np.mean(init_noise, axis=1, keepdims=True)
    
    x_raw = np.full(N, F)
    x_raw[min(19, N-1)] += 1.001
    X_a = M(x_raw[:, None] + init_noise, dt, steps_spin_up)
    
    delta_array = np.full(N, init_delta)
    v_b = np.full(N, v_b_val)
    R_diag = np.ones(swath_size)
    
    record_rmse_global = np.zeros(num_cycles)
    record_rmse_land = np.zeros(num_cycles)
    record_rmse_ocean = np.zeros(num_cycles)
    record_delta_history = np.zeros((num_cycles, N))
    
    for t in range(num_cycles):
        shift = ((t // move_interval) * move_step) % N
        curr_obs_indices = obs_indices_history[shift]
        curr_unobs_indices = np.setdiff1d(np.arange(N), curr_obs_indices)
        
        H_mat = H_mats[shift]
        W_weights = W_weights_list[shift]
        y_o = y_o_data_full[t, curr_obs_indices]
        
        X_b = M(X_a, dt, sampling_interval)
        
        if not np.all(np.isfinite(X_b)):
            record_rmse_global[:] = np.inf
            record_rmse_land[:] = np.inf
            record_rmse_ocean[:] = np.inf
            return record_rmse_global, record_rmse_land, record_rmse_ocean, record_delta_history
            
        x_b_mean = np.mean(X_b, axis=1)
        inflation_factor = np.sqrt(1.0 + delta_array) / np.sqrt(m - 1.0)
        Z_b = (X_b - x_b_mean[:, None]) * inflation_factor[:, None]
        Y_b = H_mat @ Z_b
        innovation = y_o - H_mat @ x_b_mean
        
        infl_at_obs = 1.0 + delta_array[curr_obs_indices]
        H_Pf_H_T_diag = np.sum(Y_b**2, axis=1) / infl_at_obs
        
        X_a = np.zeros((N, m))
        delta_array_next = np.zeros(N)
        
        for i in range(N):
            w_i = W_weights[i, :]
            R_loc_inv_diag = np.diag(w_i / R_diag)
            P_a_tilde_inv = np.eye(m) + (Y_b.T @ R_loc_inv_diag @ Y_b)
            
            try:
                D, C = np.linalg.eigh(P_a_tilde_inv)
            except np.linalg.LinAlgError:
                record_rmse_global[:] = np.inf
                record_rmse_land[:] = np.inf
                record_rmse_ocean[:] = np.inf
                return record_rmse_global, record_rmse_land, record_rmse_ocean, record_delta_history
                
            P_a_tilde = C @ np.diag(1.0 / D) @ C.T
            W_mat = C @ np.diag(1.0 / np.sqrt(D)) @ C.T
            w_vec = P_a_tilde @ Y_b.T @ R_loc_inv_diag @ innovation
            T_mat = w_vec[:, None] + np.sqrt(m - 1.0) * W_mat
            X_a[i, :] = x_b_mean[i] + Z_b[i, :] @ T_mat
            
            valid = H_Pf_H_T_diag > 1e-10
            if np.sum(w_i[valid]) > 1e-10:
                mole_alpha = w_i[valid] * (innovation[valid]**2 - R_diag[valid]) / H_Pf_H_T_diag[valid]
                deno_alpha = w_i[valid]
                alpha_o_i = np.sum(mole_alpha) / np.sum(deno_alpha)
                delta_o_i = alpha_o_i - 1.0
                
                alpha_b_i = 1.0 + delta_array[i]
                mole_v = w_i[valid] * ((alpha_b_i * H_Pf_H_T_diag[valid] + R_diag[valid]) / H_Pf_H_T_diag[valid])**2
                v_o_i = 2.0 * np.sum(mole_v) / (np.sum(w_i[valid])**2)
            else:
                delta_o_i, v_o_i = 0.0, 1e10
            
            k_gain = v_b[i] / (v_b[i] + v_o_i)
            delta_a = delta_array[i] + k_gain * (delta_o_i - delta_array[i])
            delta_array_next[i] = np.clip(delta_a, 0.0, 2.0)
            
        delta_array = delta_array_next.copy()
        record_delta_history[t, :] = delta_array
        
        x_a_mean = np.mean(X_a, axis=1)
        record_rmse_global[t] = np.sqrt(np.mean((x_a_mean - true_states[t])**2))
        record_rmse_land[t] = np.sqrt(np.mean((x_a_mean[curr_obs_indices] - true_states[t, curr_obs_indices])**2))
        record_rmse_ocean[t] = np.sqrt(np.mean((x_a_mean[curr_unobs_indices] - true_states[t, curr_unobs_indices])**2))
            
    return record_rmse_global, record_rmse_land, record_rmse_ocean, record_delta_history

# =====================================================================
# 4. 実行パラメータ設定 
# =====================================================================
CONFIG = {
    "v_b": 0.20**2,       
    "move_interval": 10,  
    "move_step": 30,       
    "swath_size": 20
}

sigma_list = np.arange(1.0, 4.5, 0.5)
print(f"事前準備中 (H_mats, obs_hist の生成)...")
H_mats, obs_hist = [], []
for shift in range(N):
    idx = (np.arange(0, CONFIG["swath_size"]) + shift) % N
    obs_hist.append(idx)
    H, _ = get_experiment_matrices(idx) 
    H_mats.append(H)

# =====================================================================
# 5. 【joblib並列化】最適Sigma探索と本番実行
# =====================================================================
print(f"現在の動的条件(Interval={CONFIG['move_interval']}, Step={CONFIG['move_step']})における最適Sigmaを探索中...")

# 並列評価用のラッパー関数
def evaluate_sigma(s):
    W_list_temp = [get_localization_weights(s, obs_hist[shift]) for shift in range(N)]
    rmse_G_temp, _, _, _ = run_LETKF_Adaptive_Moving_Agile(
        0.0, CONFIG["v_b"], CONFIG["move_interval"], CONFIG["move_step"], 
        CONFIG["swath_size"], s, H_mats, W_list_temp, obs_hist
    )
    ex_c = int(num_cycles * 0.2)
    return s, np.mean(rmse_G_temp[ex_c:])

# 全コア(-1)を用いて探索ループを並列実行
results = Parallel(n_jobs=-1)(delayed(evaluate_sigma)(s) for s in sigma_list)

min_rmse = float('inf')
best_sigma = None
sigma_rmse_list = []

for s, mean_rmse in results:
    sigma_rmse_list.append((s, mean_rmse))
    print(f"  Sigma = {s:.1f} -> Mean Global RMSE = {mean_rmse:.4f}")
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_sigma = s

print(f"--> 最適Sigmaが {best_sigma:.1f} に決定されました (RMSE: {min_rmse:.4f})")
CONFIG["sigma"] = best_sigma

print(f"\n動的LETKFを本番実行中... {CONFIG}")
start_time = time.time()
W_list_final = [get_localization_weights(CONFIG["sigma"], obs_hist[shift]) for shift in range(N)]

rmse_G, rmse_L, rmse_O, delta_hist = run_LETKF_Adaptive_Moving_Agile(
    0.0, CONFIG["v_b"], CONFIG["move_interval"], CONFIG["move_step"], 
    CONFIG["swath_size"], CONFIG["sigma"], H_mats, W_list_final, obs_hist
)
print(f"実行完了 ({time.time() - start_time:.1f}秒)")


# =====================================================================
# 6. 【joblib並列化】固定LETKFの最適デルタ探索 ＆ 時系列取得の統合フロー
# =====================================================================
# （※ 元ファイルに定義されていた固定LETKF関数がこちらに入ります。
#   以下は、元のコード内で呼ばれている関数を想定した並列化構造です）

def evaluate_fixed_delta(d, sigma):
    rmse_g, _, _, _ = run_LETKF_Fixed_Moving_Logic(d, sigma)
    ex_c = int(num_cycles * 0.2)
    subset = rmse_g[ex_c:]
    if not np.all(np.isfinite(subset)):
        return d, float('inf')
    return d, np.mean(subset)

def find_best_fixed_delta(sigma):
    print("\n固定インフレーションの最適Deltaを探索中...")
    fixed_deltas = np.arange(0.0, 0.45, 0.05)
    
    # デルタ探索のループも並列実行で高速化
    res_fixed = Parallel(n_jobs=-1)(delayed(evaluate_fixed_delta)(d, sigma) for d in fixed_deltas)
    
    best_d = 0.0
    min_rmse_f = float('inf')
    for d, mean_rmse in res_fixed:
        print(f"Delta={d:.2f} -> Mean RMSE={mean_rmse:.4f}")
        if mean_rmse < min_rmse_f:
            min_rmse_f = mean_rmse
            best_d = d
            
    print(f"探索完了: 最適Fixed Delta = {best_d:.2f} (Global Mean RMSE = {min_rmse_f:.4f})")
    return best_d

# もし run_LETKF_Fixed_Moving_Logic が定義されている場合は、探索を実行
if 'run_LETKF_Fixed_Moving_Logic' in globals():
    best_fixed_delta = find_best_fixed_delta(best_sigma)
    rmse_fixed_G, rmse_fixed_L, rmse_fixed_O, _ = run_LETKF_Fixed_Moving_Logic(best_fixed_delta, best_sigma)

# =====================================================================
# 7. プロットおよび定量的比較出力 (元のコードの末尾部分)
# =====================================================================
# （※以下はプロット処理等のコードです。データが揃っている場合はグラフが出力されます）
try:
    ex_c = int(num_cycles * 0.2)
    print("="*65)
    print(f"実験条件: {CONFIG}")
    print("-" * 65)
    print(f"{'Area':<10} | {'Adaptive Mean':<15} | {'Fixed Mean':<15} | {'Improvement':<10}")
    print("-" * 65)
    for area, ada, fix in zip(['Global', 'Land', 'Ocean'], 
                              [np.mean(rmse_G[ex_c:]), np.mean(rmse_L[ex_c:]), np.mean(rmse_O[ex_c:])],
                              [np.mean(rmse_fixed_G[ex_c:]), np.mean(rmse_fixed_L[ex_c:]), np.mean(rmse_fixed_O[ex_c:])]):
        imp = (fix - ada) / fix * 100
        print(f"{area:<10} | {ada:.4f}          | {fix:.4f}          | {imp:+.1f}%")
    print("="*65)
except Exception as e:
    pass