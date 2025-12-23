import pandas as pd
import glob
import os

def load_and_preprocess_data(data_path):
    """
    Loads ATP tennis match data from CSV files, consolidates them,
    and performs initial data cleaning and preprocessing.

    Args:
        data_path (str): The path where the CSV files are located.

    Returns:
        pd.DataFrame: The cleaned and preprocessed DataFrame.
    """
    print("--- BƯỚC 1: TẢI VÀ HỢP NHẤT DỮ LIỆU ---")
    path_pattern = os.path.join(data_path, 'atp_matches_*.csv')
    all_files = glob.glob(path_pattern)

    if not all_files:
        print(f"Không tìm thấy file CSV nào tại đường dẫn: {data_path}")
        return pd.DataFrame()

    li = []
    for filename in all_files:
        df_temp = pd.read_csv(filename, index_col=None, header=0)
        li.append(df_temp)

    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.sort_values(by='tourney_date', ascending=True).reset_index(drop=True)

    print(f"Đã hợp nhất {len(all_files)} files.")
    print(f"Tổng số trận đấu: {df.shape[0]}")

    print("\n--- BƯỚC 2: LÀM SẠCH VÀ TIỀN XỬ LÝ DỮ LIỆU ---")

    # 1. Xóa các cột có độ thưa thớt cao
    cols_to_drop = ['winner_seed', 'loser_seed', 'winner_entry', 'loser_entry']
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Đã xóa các cột: {cols_to_drop}")

    # 2. Chuyển đổi kiểu dữ liệu cho cột ngày tháng
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    print("Đã chuyển đổi kiểu dữ liệu cho cột 'tourney_date'.")

    # 3. Điền giá trị thiếu cho các cột thống kê trận đấu bằng median
    stat_cols = ['minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
                 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 
                 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']

    for col in stat_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    print(f"Đã điền giá trị thiếu cho các cột thống kê bằng median.")

    # 4. Xóa các hàng có giá trị thiếu ở các cột quan trọng còn lại
    cols_to_check_na = ['surface', 'winner_ht', 'loser_ht', 'winner_rank', 'loser_rank', 
                        'winner_rank_points', 'loser_rank_points', 'winner_age', 'loser_age']
    
    initial_rows = len(df)
    df.dropna(subset=cols_to_check_na, inplace=True)
    rows_dropped = initial_rows - len(df)
    print(f"Đã xóa {rows_dropped} hàng có giá trị thiếu ở các cột quan trọng.")

    # Kiểm tra lại toàn bộ DataFrame
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("Thành công! Không còn giá trị thiếu nào trong bộ dữ liệu.")
    else:
        print(f"Vẫn còn {remaining_missing} giá trị thiếu. Cần kiểm tra lại.")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n--- BƯỚC 3: KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING) ---")

    # 1. Tạo các đặc trưng chênh lệch (Difference Features)
    print("Tạo các đặc trưng chênh lệch (rank, age, height, points)...")

    df['rank_diff'] = df['winner_rank'] - df['loser_rank']
    df['age_diff'] = df['winner_age'] - df['loser_age']
    df['ht_diff'] = df['winner_ht'] - df['loser_ht']
    df['rank_points_diff'] = df['winner_rank_points'] - df['loser_rank_points']

    # 2. Tạo các đặc trưng về tỷ lệ (Ratio/Percentage Features)
    print("Tạo các đặc trưng về tỷ lệ (giao bóng, break points)...")

    # Tỷ lệ cho người thắng (winner)
    # Xử lý trường hợp mẫu số bằng 0 để tránh lỗi chia cho 0
    df['w_1st_serve_pct'] = (df['w_1stIn'] / df['w_svpt']).fillna(0) * 100
    df['w_1st_serve_win_pct'] = (df['w_1stWon'] / df['w_1stIn']).fillna(0) * 100
    df['w_2nd_serve_win_pct'] = (df['w_2ndWon'] / (df['w_svpt'] - df['w_1stIn'])).fillna(0) * 100
    df['w_bp_saved_pct'] = (df['w_bpSaved'] / df['w_bpFaced']).fillna(0) * 100

    # Tỷ lệ cho người thua (loser)
    df['l_1st_serve_pct'] = (df['l_1stIn'] / df['l_svpt']).fillna(0) * 100
    df['l_1st_serve_win_pct'] = (df['l_1stWon'] / df['l_1stIn']).fillna(0) * 100
    df['l_2nd_serve_win_pct'] = (df['l_2ndWon'] / (df['l_svpt'] - df['l_1stIn'])).fillna(0) * 100
    df['l_bp_saved_pct'] = (df['l_bpSaved'] / df['l_bpFaced']).fillna(0) * 100

    print("\nCác đặc trưng mới đã được tạo.")
    
    # Cập nhật thông tin dữ liệu sau khi làm sạch và tạo đặc trưng
    print("\nThông tin dữ liệu sau khi làm sạch và tạo đặc trưng:")
    df.info()
    print("\n5 dòng đầu tiên của dữ liệu với các đặc trưng mới:")
    print(df.head())
    
    return df

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_source_path = os.path.join(current_script_dir, 'data')
    
    print(f"Đang cố gắng tải và tiền xử lý dữ liệu từ: {data_source_path}")
    df_cleaned = load_and_preprocess_data(data_source_path)
    
    if not df_cleaned.empty:
        print("\nQuá trình tiền xử lý hoàn tất. DataFrame đã được làm sạch.")
