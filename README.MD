# DỰ ÁN DỰ ĐOÁN GIÁ VÉ MÁY BAY TẠI VIỆT NAM

## 📋 Tổng quan
Dự án này phân tích và dự đoán giá vé máy bay nội địa tại Việt Nam dựa trên các yếu tố như hãng bay, thời gian bay, điểm đi/đến, số điểm dừng, và hạng vé. Dữ liệu được thu thập từ trang web Traveloka thông qua kỹ thuật web crawling, sau đó được tiền xử lý, phân tích và xây dựng mô hình dự đoán.

## 👥 Thành viên nhóm
- **Phan Văn Toàn** - LightGBM
- **Trương Xuân Phúc** - Random Forest
- **Ngô Thị Kim Ly** - Linear Regression

## 🧰 Công nghệ sử dụng
- **Ngôn ngữ**: Python
- **Thư viện chính**:
  - Selenium (thu thập dữ liệu)
  - Pandas, NumPy (xử lý dữ liệu)
  - Matplotlib, Seaborn, Plotly (trực quan hóa)
  - Scikit-learn (tiền xử lý, đánh giá mô hình)
  - LightGBM, RandomForest (xây dựng mô hình)

## 📊 Dữ liệu
- **Nguồn dữ liệu**: Thu thập từ trang web Traveloka
- **Số lượng mẫu**: ~12.800 bản ghi
- **Các biến chính**:
  - `Airline`: Hãng hàng không (VietJet Air, Vietnam Airlines, Bamboo Airways, v.v.)
  - `Departure_Time`, `Arrival_Time`: Thời gian khởi hành và đến
  - `Duration`: Thời gian bay (phút)
  - `Departure`, `Destination`: Mã sân bay đi và đến
  - `Date`: Ngày bay
  - `Total_Stops`: Số điểm dừng
  - `SClass`: Hạng vé (ECONOMY, PREMIUM_ECONOMY, BUSINESS)
  - `Price`: Giá vé (VND) - biến mục tiêu

## 🔍 Quy trình thực hiện
1. **Thu thập dữ liệu**: Sử dụng Selenium để crawl dữ liệu từ Traveloka
2. **Tiền xử lý dữ liệu**:
   - Chuyển đổi thời gian thành giờ và phút
   - Mã hóa one-hot cho biến phân loại
   - Chuẩn hóa dữ liệu số
   - Chia tập dữ liệu thành train/val/test
3. **Phân tích khám phá dữ liệu (EDA)**:
   - Phân tích phân phối giá vé
   - Tương quan giữa các biến
   - Ảnh hưởng của từng yếu tố đến giá vé
4. **Xây dựng mô hình**:
   - **LightGBM**: Mô hình cây quyết định nâng cao gradient (Gradient Boosting) tối ưu hóa tốc độ và hiệu suất. Ứng dụng để xử lý dữ liệu có nhiều đặc trưng phân loại và số.
   - **Random Forest**: Mô hình tổng hợp dựa trên nhiều cây quyết định, giúp giảm overfitting và tăng độ chính xác khi dự đoán giá vé.
   - **Linear Regression**: Mô hình cơ sở làm mốc so sánh, giúp đánh giá mối quan hệ tuyến tính giữa các biến độc lập và giá vé.

## 🔬 So sánh hiệu suất các mô hình
Dự án thử nghiệm cả ba mô hình trên cùng bộ dữ liệu để đánh giá hiệu quả trong việc giải quyết bài toán dự đoán giá vé máy bay:

| Mô hình | MAE | RMSE | R² | Ưu điểm | Nhược điểm |
|---------|-----|------|---|---------|-----------|
| LightGBM | Thấp nhất | Thấp nhất | ~0.95 | Tốc độ nhanh, xử lý tốt đặc trưng phân loại | Khó giải thích với người dùng cuối |
| Random Forest | Trung bình | Trung bình | ~0.93 | Ổn định, ít overfitting | Chậm hơn khi dữ liệu lớn |
| Linear Regression | Cao nhất | Cao nhất | ~0.80 | Dễ hiểu, dễ giải thích | Không bắt được mối quan hệ phi tuyến |

5. **Đánh giá và tinh chỉnh mô hình**:
   - Grid Search và Random Search cho siêu tham số
   - Cross-validation
   - Đánh giá trên nhiều chỉ số (MAE, RMSE, R²)

## 📈 Kết quả chính
- Mô hình LightGBM đạt hiệu suất tốt nhất với R² ≈ 0.95
- Các yếu tố ảnh hưởng nhiều nhất đến giá vé:
  - Thời lượng bay (Duration)
  - Ngày bay (Date)
  - Thời gian khởi hành và đến (Departure_Hour, Arrival_Hour)
  - Hạng vé (SClass)

## 📁 Cấu trúc dự án
```
├── raw_data/              # Dữ liệu thô thu thập được
├── clean_data/            # Dữ liệu đã được tiền xử lý
├── models/                # Mô hình đã huấn luyện và thông tin đánh giá
├── code/                  # Notebooks phân tích
│   ├── crawl.ipynb        # Thu thập dữ liệu
│   ├── clean_data.ipynb   # Tiền xử lý dữ liệu
│   ├── Preprocessing.ipynb # Xử lý dữ liệu chi tiết
│   ├── EDA.ipynb          # Phân tích khám phá dữ liệu
│   ├── LightGBM.ipynb     # Mô hình LightGBM
│   ├── random_forest.ipynb # Mô hình Random Forest
│   └── linear_regression.ipynb # Mô hình Linear Regression
├── requirements.txt       # Các thư viện yêu cầu
└── README.md              # Tài liệu dự án
```

## 👨‍💻 Hướng dẫn sử dụng
1. Clone repository
   ```
   git clone https://github.com/givoxxs/Airfare_KHDL.git
   ```
2. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```
3. Chạy các notebook theo thứ tự để tái tạo kết quả

## 🔮 Hướng phát triển
- Thu thập dữ liệu từ nhiều nguồn khác nhau
- Thêm các đặc trưng mới (thời tiết, ngày lễ, v.v.)
- Thử nghiệm thêm các mô hình khác (XGBoost, Neural Networks)
- Xây dựng ứng dụng web để người dùng có thể dự đoán giá vé