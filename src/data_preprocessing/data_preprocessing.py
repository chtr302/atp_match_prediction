import pandas as pd
import glob, os, traceback, sys
import numpy as np
from model.custom_elo_model import TennisEloModel

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../model'))

def load_data(data_path):
    path_pattern = os.path.join(data_path, 'atp_matches_*.csv')
    all_files = glob.glob(path_pattern)
    if not all_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV tại: {data_path}")
    
    li = [pd.read_csv(f, index_col=None, header=0) for f in all_files]
    df = pd.concat(li, axis=0, ignore_index=True)
    
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

    df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True) # Sắp xếp theo thời gian để tính Rolling Stars và Elo
    return df

def add_rolling_stats(df):
    """
    Tính toán phong độ 5 trận gần nhất và tỷ lệ thắng trên mặt sân.
    Thực hiện trên dữ liệu gốc (Winner/Loser).
    """
    df_feat = df.copy()
    
    last_5_stats = {} 
    surface_stats = {} 
    
    w_recent_form, l_recent_form = [], []
    w_surf_form, l_surf_form = [], []
    
    for idx, row in df_feat.iterrows():
        wid, lid = row['winner_id'], row['loser_id']
        surf = row['surface']
        
        # Lấy stats TRƯỚC trận
        w_recent = np.mean(last_5_stats.get(wid, [0])) if wid in last_5_stats else 0
        w_surf_data = surface_stats.get(wid, {}).get(surf, [0, 0])
        w_surf = w_surf_data[0] / w_surf_data[1] if w_surf_data[1] > 0 else 0
        
        l_recent = np.mean(last_5_stats.get(lid, [0])) if lid in last_5_stats else 0
        l_surf_data = surface_stats.get(lid, {}).get(surf, [0, 0])
        l_surf = l_surf_data[0] / l_surf_data[1] if l_surf_data[1] > 0 else 0
        
        w_recent_form.append(w_recent)
        l_recent_form.append(l_recent)
        w_surf_form.append(w_surf)
        l_surf_form.append(l_surf)
        
        # Cập nhật stats SAU trận
        # Winner
        if wid not in last_5_stats: last_5_stats[wid] = []
        last_5_stats[wid].append(1)
        if len(last_5_stats[wid]) > 5: last_5_stats[wid].pop(0)
        
        if wid not in surface_stats: surface_stats[wid] = {}
        if surf not in surface_stats[wid]: surface_stats[wid][surf] = [0, 0]
        surface_stats[wid][surf][0] += 1
        surface_stats[wid][surf][1] += 1
        
        # Loser
        if lid not in last_5_stats: last_5_stats[lid] = []
        last_5_stats[lid].append(0)
        if len(last_5_stats[lid]) > 5: last_5_stats[lid].pop(0)
        
        if lid not in surface_stats: surface_stats[lid] = {}
        if surf not in surface_stats[lid]: surface_stats[lid][surf] = [0, 0]
        surface_stats[lid][surf][1] += 1

    df_feat['winner_recent_form'] = w_recent_form
    df_feat['loser_recent_form'] = l_recent_form
    df_feat['winner_surface_win_pct'] = w_surf_form
    df_feat['loser_surface_win_pct'] = l_surf_form
    
    return df_feat

def restructure_data(df):
    """
    Tráo đổi ngẫu nhiên Winner/Loser -> P1/P2.
    """
    common_cols = ['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 'match_num', 'best_of', 'round']
    # Danh sách các feature cần mang theo khi swap
    p_feats = ['id', 'seed', 'entry', 'name', 'hand', 'ht', 'ioc', 'age', 'rank', 'rank_points', 'recent_form', 'surface_win_pct']
    
    np.random.seed(42)
    swap_mask = np.random.rand(len(df)) < 0.5
    
    new_df = df[common_cols].copy()
    
    for feat in p_feats:
        win_col = f'winner_{feat}'
        lose_col = f'loser_{feat}'
        if win_col in df.columns and lose_col in df.columns:
            new_df[f'p1_{feat}'] = np.where(swap_mask, df[lose_col], df[win_col])
            new_df[f'p2_{feat}'] = np.where(swap_mask, df[win_col], df[lose_col])
            
    new_df['target'] = np.where(swap_mask, 0, 1)
    return new_df

def add_elo_features(df):
    """
    Tích hợp hệ thống Elo Hybrid.
    """
    # K=20 (ổn định), Surface weight=0.5 (cân bằng giữa phong độ chung và mặt sân)
    elo_engine = TennisEloModel(k_factor=20, surface_weight=0.5)
    df_elo = elo_engine.fit_transform(df)
    return df_elo

def finalize_features(df):
    """
    Làm sạch giá trị thiếu và tạo các feature chênh lệch (Diff).
    """
    
    # Xóa hàng thiếu dữ liệu cốt lõi
    crit_cols = ['p1_rank', 'p2_rank', 'p1_ht', 'p2_ht', 'p1_age', 'p2_age']
    df_clean = df.dropna(subset=crit_cols).copy()
    
    # Xử lý Seed
    for col in ['p1_seed', 'p2_seed']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(100)
        
    # Tạo các cột hiệu số (Difference) - Đây là cái model học tốt nhất
    df_clean['rank_diff'] = df_clean['p1_rank'] - df_clean['p2_rank']
    df_clean['age_diff'] = df_clean['p1_age'] - df_clean['p2_age']
    df_clean['ht_diff'] = df_clean['p1_ht'] - df_clean['p2_ht']
    df_clean['form_diff'] = df_clean['p1_recent_form'] - df_clean['p2_recent_form']
    df_clean['surf_pct_diff'] = df_clean['p1_surface_win_pct'] - df_clean['p2_surface_win_pct']
    # elo_diff đã được tạo trong bước add_elo_features
    
    # Mã hóa biến phân loại (Surface, Hand)
    cols_dummy = ['surface', 'p1_hand', 'p2_hand']
    df_final = pd.get_dummies(df_clean, columns=cols_dummy, drop_first=True)
    
    # Chỉ giữ lại các cột số để đưa vào training
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    if 'tourney_date' in df_final.columns:
        numeric_cols.append('tourney_date')
        
    df_final = df_final[numeric_cols]
    return df_final

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data")
    
    try:
        # Pipeline thực thi
        raw_df = load_data(data_path)
        rolling_df = add_rolling_stats(raw_df)
        struct_df = restructure_data(rolling_df)
        elo_df = add_elo_features(struct_df)
        final_df = finalize_features(elo_df)
        
        # Lưu file
        output_path = os.path.join(current_dir, "../../processed_atp_data.csv")
        final_df.to_csv(output_path, index=False)
        
    except Exception as e:
        print(f"\n[LỖI] Quy trình thất bại: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()