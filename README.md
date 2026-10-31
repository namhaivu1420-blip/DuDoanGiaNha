# 🏡 Nhóm 2: Ứng dụng dự đoán giá nhà bằng SVM và ANN

Ứng dụng web được xây dựng với **Streamlit**, phục vụ báo cáo học thuật "Ứng dụng hai thuật toán học máy SVM và ANN để dự đoán giá nhà".

## 1. Chức năng

- Nhập diện tích (m²), số phòng ngủ và khoảng cách đến trung tâm Seattle (km).
- Dự đoán giá nhà bằng **SVM (SVR)** và **ANN (MLPRegressor)**.
- So sánh giá dự đoán và tải kết quả CSV.
- Xem MAE, MSE, RMSE, R² trên **cùng tập Test**.
- Xem biểu đồ, bảng dự đoán 5 dòng đầu và 30 dòng Test; tải bảng CSV.
- Xem dữ liệu, các bước tiền xử lý và phương pháp học máy.

## 2. Chạy trên máy tính Windows / macOS

Điều kiện: đã cài **Python 3.11 trở lên** và có kết nối internet để cài thư viện.

1. Giải nén gói ZIP này.
2. Mở Terminal / PowerShell ở đúng thư mục chứa `app.py`.
3. **Trên Windows**, bạn có thể nhấp đúp `chay_app_windows.bat` (nếu đã cài Python và lệnh `py` hoạt động). Hoặc chạy các lệnh sau trong Terminal:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

4. Mở liên kết nội bộ Terminal hiện ra (thường là **http://localhost:8501**).
5. Để dừng app: bấm `Ctrl+C` trong Terminal.

> Nếu Windows không nhận lệnh `python`, bạn có thể thử `py -m pip install -r requirements.txt` và `py -m streamlit run app.py`.

## 3. Đăng lên web để có đường link chia sẻ

1. Vào **https://github.com/**, tạo repository mới (ví dụ `nhom2-du-doan-gia-nha`).
2. Giải nén ZIP và tải **toàn bộ nội dung bên trong thư mục app** lên repository, giữ nguyên `app.py`, `ml_core.py`, `requirements.txt`, thư mục `data` và `.streamlit`.
3. Vào **https://share.streamlit.io/**, đăng nhập bằng GitHub và chọn tạo ứng dụng mới (các nút trên giao diện có thể thay đổi theo thời điểm).
4. Chọn repository vừa tạo, nhánh `main`, **main file path: `app.py`** rồi triển khai.
5. Khi ứng dụng khởi động xong, chia sẻ URL Streamlit Cloud được cấp.

**Lưu ý:** File dữ liệu nằm trong `data/`. Không xóa thư mục này khi đăng lên GitHub. Thư mục `.streamlit/` là cấu hình giao diện, có thể bỏ qua nếu trang tải file ẩn.

## 4. Cấu trúc thư mục

```text
Nhom2_App_DuDoanGiaNha/
├── app.py                  # Giao diện Streamlit
├── ml_core.py              # Dữ liệu, huấn luyện, dự đoán và đánh giá
├── requirements.txt        # Thư viện cần cài
├── chay_app_windows.bat     # Chạy tự động trên Windows
├── chay_app_mac_linux.sh   # Chạy trên macOS / Linux
├── README.md               # Hướng dẫn sử dụng
├── .streamlit/
│   └── config.toml         # Cấu hình giao diện
└── data/
    ├── Nhom2_GiaNha.xlsx
    ├── Nhom2_GiaNha_train.xlsx
    └── Nhom2_GiaNha_test.xlsx
```

## 5. Cách hoạt động và giới hạn

- Sử dụng **149 bản ghi đầu** từ dữ liệu giao dịch nhà ở King County, bang Washington, Hoa Kỳ (2014–2015), không phải dữ liệu nhà ở Việt Nam.
- Biến đầu vào: `DienTich`, `SoPhongNgu`, `KhoangCach`; biến mục tiêu `GiaNha` (USD).
- Dữ liệu Train có 119 dòng, Test có 30 dòng, không trùng mã.
- Quy trình theo notebook Colab: StandardScaler cho X và y, 5-fold CV trên Train để chọn tham số, sau đó đánh giá Test.
- Tùy phiên bản thư viện hoặc môi trường máy chủ, kết quả ANN có thể khác đôi chút so với số liệu trong PDF. App tính lại toàn bộ chỉ số từ chính mô hình đang chạy.
- Kết quả không dùng cho tư vấn giá bất động sản thật và chỉ có giá trị minh họa học thuật.

## 6. Nguồn dữ liệu

- [GeoDa Center – King County House Sales](https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/)
- [File housing.csv được sử dụng trong Excel của nhóm](https://raw.githubusercontent.com/46n/king-county-house-sales-analysis/main/data/housing.csv)
