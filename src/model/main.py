import pandas as pd
import os
import sys
from model_training import ModelTrainer
from ensemble import EnsembleTrainer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def load_processed_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {filepath}")
    return pd.read_csv(filepath)

def main():
    # 1. Load dữ liệu đã xử lý
    data_path = os.path.join(current_dir, "../../data/processed_atp_data.csv")
    
    try:
        df = load_processed_data(data_path)
    except Exception as e:
        print(e)
        return

    # 2. Chia tập dữ liệu (Time-series Split)
    # Sắp xếp theo thời gian để đảm bảo tính quá khứ -> tương lai
    if 'tourney_date' in df.columns:
        df = df.sort_values('tourney_date').reset_index(drop=True)
        # Xóa cột ngày tháng sau khi sort xong vì model không dùng
        dates = df['tourney_date'] # Lưu lại để debug nếu cần
        df_model = df.drop(columns=['tourney_date'])
    else:
        df_model = df
        
    # Chia 80-20
    split_idx = int(len(df_model) * 0.8)
    
    train_data = df_model.iloc[:split_idx]
    test_data = df_model.iloc[split_idx:]
    
    print(f"Kích thước tập Train: {train_data.shape}")
    print(f"Kích thước tập Test: {test_data.shape}")
    
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target'].astype(int)
    
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target'].astype(int)
    
    # 3. Huấn luyện Model cơ sở (Base Models)
    trainer = ModelTrainer()
    base_results = trainer.train_evaluate(X_train, y_train, X_test, y_test)
    
    # 4. Huấn luyện Ensemble
    trained_models = trainer.get_trained_models()
    ensemble = EnsembleTrainer(trained_models)
    
    ensemble.soft_voting(X_test, y_test)
    ensemble.stacking(X_train, y_train, X_test, y_test)
    
    ensemble_results = ensemble.get_results()
    
    # 5. Tổng hợp kết quả
    final_results = pd.concat([base_results, ensemble_results], ignore_index=True)
    final_results = final_results.sort_values(by='Accuracy', ascending=False)
    
    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(final_results)
    
    # Lưu kết quả ra file CSV để báo cáo
    result_path = os.path.join(current_dir, "../../model_results.csv")
    final_results.to_csv(result_path, index=False)
    print(f"\nĐã lưu bảng kết quả tại: {result_path}")

if __name__ == "__main__":
    main()
