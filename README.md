# Dự đoán Kết quả Tennis ATP và Hệ thống Elo Lai

Dự án học máy dự đoán kết quả trận đấu tennis ATP (2020-2024) dựa trên dữ liệu lịch sử, thống kê phong độ gần nhất và hệ thống xếp hạng Elo lai tự xây dựng.

## Cấu trúc thư mục

Danh sách dưới đây bao gồm cả các thư mục đã được cấu hình trong .gitignore.

```text
├── data/                         # [IGNORED] Dữ liệu gốc ATP CSV
├── plots/                        # Các biểu đồ phân tích và đánh giá
├── src/
│   ├── data_preprocessing/
│   │   └── data_preprocessing.py # Tiền xử lý, chống rò rỉ dữ liệu và tạo đặc trưng
│   └── model/
│       ├── main.py               # Script chạy huấn luyện mô hình chính
│       ├── model_training.py     # Cấu hình các mô hình cơ sở (LR, RF, XGB, SVM)
│       ├── ensemble.py           # Logic thuật toán gộp (Soft Voting, Stacking)
│       └── custom_elo_model.py   # [IGNORED] Thuật toán Elo lai tự xây dựng
├── diagrams_mermaid.md           # Mã nguồn vẽ sơ đồ quy trình
├── processed_atp_data.csv        # Dữ liệu sạch sau khi xử lý
├── model_results.csv             # Kết quả đánh giá các mô hình
├── pyproject.toml                # Cấu hình dự án và thư viện
├── generate_report_plots.py      # Script sinh biểu đồ cho báo cáo
└── README.md                     # Tài liệu hướng dẫn
```

## Phương pháp cốt lõi

1.  **Chống rò rỉ dữ liệu (Anti-Leakage):** Tái cấu trúc ngẫu nhiên cột Winner/Loser thành Player 1/Player 2 để loại bỏ thiên kiến vị trí và bảo vệ tính trung thực của mô hình.
2.  **Hệ thống Elo lai (Hybrid Elo):** Kết hợp điểm số đẳng cấp tổng quát và hiệu suất trên từng mặt sân cụ thể để nắm bắt xu hướng phong độ động.
3.  **Đánh giá Ensemble:** So sánh và kết hợp nhiều thuật toán (Logistic Regression, Random Forest, XGBoost, SVM) thông qua các chiến lược Soft Voting và Stacking.

## Hướng dẫn sử dụng

Dự án sử dụng công cụ `uv` để quản lý môi trường và dependencies.
Hoặc có thể sử dụng `pip` nhưng bạn phải cái các dependencies trong `pyproject.toml`

### 1. Cài đặt môi trường
Yêu cầu Python >= 3.12.
```bash
uv sync
```

### 2. Tiền xử lý dữ liệu
Thực hiện làm sạch dữ liệu, tính toán điểm Elo và tạo các thống kê phong độ trượt.
```bash
uv run src/data_preprocessing/data_preprocessing.py
```

### 3. Huấn luyện và Đánh giá
Huấn luyện các mô hình cơ sở, thực hiện logic ensemble và xuất các chỉ số đánh giá.
```bash
uv run src/model/main.py
```

### 4. Sinh biểu đồ báo cáo
Tạo các biểu đồ Feature Importance, ROC Curve và Confusion Matrix phục vụ báo cáo.
```bash
uv run generate_report_plots.py
```
