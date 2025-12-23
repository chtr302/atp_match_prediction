import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data_path):
    path_pattern = os.path.join(data_path, 'atp_matches_*.csv')
    all_files = glob.glob(path_pattern)
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    print(f"--- THỐNG KÊ TỔNG QUAN ---")
    print(f"Tổng số trận đấu: {len(df)}")
    
    # 1. Tỷ lệ thắng của người có thứ hạng cao hơn (Higher Rank)
    # Lưu ý: Rank thấp (số nhỏ) là hạng cao hơn.
    higher_rank_won = (df['winner_rank'] < df['loser_rank']).sum()
    print(f"Tỷ lệ người hạng cao hơn thắng: {higher_rank_won/len(df):.2%}")

    # 2. So sánh các chỉ số trung bình giữa Winner và Loser
    stat_cols = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']
    
    summary = {}
    for col in stat_cols:
        w_col = 'w_' + col
        l_col = 'l_' + col
        if w_col in df.columns and l_col in df.columns:
            summary[col] = {
                'Winner Mean': df[w_col].mean(),
                'Loser Mean': df[l_col].mean(),
                'Diff (%)': ((df[w_col].mean() - df[l_col].mean()) / df[l_col].mean()) * 100
            }
    
    print("\n--- SO SÁNH CHỈ SỐ TRONG TRẬN (WINNER VS LOSER) ---")
    summary_df = pd.DataFrame(summary).T
    print(summary_df)

    # 3. Phân tích theo mặt sân (Surface)
    print("\n--- SỐ TRẬN THEO MẶT SÂN ---")
    print(df['surface'].value_counts())

    # 4. Đặc điểm thể chất
    print("\n--- CHIỀU CAO VÀ TUỔI TÁC ---")
    print(f"Tuổi trung bình người thắng: {df['winner_age'].mean():.1f}")
    print(f"Tuổi trung bình người thua: {df['loser_age'].mean():.1f}")
    print(f"Chiều cao trung bình người thắng: {df['winner_ht'].mean():.1f} cm")
    
    return df

if __name__ == "__main__":
    data_path = "data"
    perform_eda(data_path)
