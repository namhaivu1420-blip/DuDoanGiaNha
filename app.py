"""Ứng dụng Streamlit dự đoán giá nhà bằng SVM (SVR) và ANN - Nhóm 2."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ml_core import FEATURES, TARGET, build_models, check_outside_training_range, predict_one, read_data

st.set_page_config(
    page_title="Nhóm 2 | Dự đoán giá nhà AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1180px;}
.hero {padding: 28px 32px; border-radius: 20px; background: linear-gradient(115deg, #083344, #0b7285); color: #fff; margin-bottom: 16px;}
.hero h1 {color: #fff; font-size: 2rem; line-height: 1.25; margin: 0 0 7px 0;}
.hero p {color: #dcf8fd; font-size: 1rem; margin: 0;}
.section-caption {color: #4c6371; font-size: 0.92rem;}
.pill {display:inline-block; padding:5px 11px; margin:3px 5px 3px 0; color:#defaff; background:#135a6b; border-radius:99px; font-weight:600; font-size:0.79rem;}
[data-testid="stMetric"] {border:1px solid #d6e5eb; border-radius:14px; padding:14px 18px; background:#ffffff;}
div[data-testid="stMetricLabel"] {font-size:0.92rem;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def usd(amount: float) -> str:
    return f"${amount:,.0f}"


def fmt(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}"


@st.cache_data(show_spinner=False)
def load_dataset():
    return read_data()


@st.cache_resource(show_spinner=False)
def load_models_cached():
    _, train, test = read_data()
    return build_models(train, test)


try:
    full_data, train_data, test_data = load_dataset()
    with st.spinner("Đang huấn luyện và đánh giá SVM, ANN trên bộ dữ liệu Nhóm 2…"):
        models, metrics, predictions, tuning = load_models_cached()
except Exception as exc:
    st.error("Không thể tải dữ liệu hoặc xây dựng mô hình. Hãy kiểm tra các file Excel trong thư mục data/.")
    st.exception(exc)
    st.stop()

st.markdown("""
<div class="hero">
    <h1>🏡 AI House Price Lab</h1>
    <p>Dự đoán giá nhà bằng SVM (SVR) &amp; ANN · Đồ án học thuật Nhóm 2</p>
    <div style="margin-top:14px">
        <span class="pill">King County, Mỹ</span>
        <span class="pill">Dữ liệu 2014–2015</span>
        <span class="pill">Giá bằng USD</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🏘️ Nhóm 2")
    st.caption("Công cụ minh họa quá trình huấn luyện AI và dự đoán giá nhà")
    page = st.radio(
        "Chức năng",
        ["🏠 Dự đoán giá nhà", "📊 Đánh giá & so sánh", "🧪 Dữ liệu & phương pháp", "📖 Hướng dẫn sử dụng"],
        label_visibility="collapsed",
    )
    st.divider()
    st.write("**Dữ liệu hiện dùng**")
    s1, s2 = st.columns(2)
    s1.metric("Train", f"{len(train_data)}")
    s2.metric("Test", f"{len(test_data)}")
    st.caption(f"Tổng cộng {len(full_data)} căn nhà; 3 thuộc tính đầu vào.")
    st.info("Mô hình phản ánh mẫu giao dịch tại King County 2014–2015, không phản ánh giá nhà hiện nay hoặc giá bất động sản Việt Nam.", icon="ℹ️")

if page == "🏠 Dự đoán giá nhà":
    st.subheader("Dự đoán giá cho một căn nhà mới")
    st.write("Nhập diện tích sinh hoạt, số phòng ngủ và khoảng cách đường chim bay đến trung tâm Seattle. Hai mô hình sẽ dự đoán độc lập để bạn đối chiếu.")

    with st.form("house_input", clear_on_submit=False):
        a, b, c = st.columns(3, gap="large")
        with a:
            area = st.number_input("📐 Diện tích sinh hoạt (m²)", min_value=20.0, max_value=1000.0, value=150.0, step=5.0, format="%.1f", help="Đơn vị m², đã chuyển đổi từ sqft_living của dữ liệu gốc.")
        with b:
            bedrooms = st.number_input("🛏️ Số phòng ngủ", min_value=0, max_value=15, value=3, step=1, help="Tổng số phòng ngủ của căn nhà.")
        with c:
            distance = st.number_input("📍 Cách trung tâm Seattle (km)", min_value=0.0, max_value=120.0, value=15.0, step=1.0, format="%.1f", help="Khoảng cách đường chim bay, không phải quãng đường di chuyển thực tế.")
        submitted = st.form_submit_button("✨ Dự đoán ngay", type="primary", use_container_width=True)

    if submitted:
        features_entered = {"DienTich": area, "SoPhongNgu": bedrooms, "KhoangCach": distance}
        outputs = predict_one(models, area, bedrooms, distance)
        st.session_state["last_result"] = {"inputs": features_entered, "predictions": outputs}

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        values, outputs = res["inputs"], res["predictions"]
        st.divider()
        st.markdown("#### Kết quả dự đoán")
        left, right = st.columns(2, gap="large")
        left.metric("🔷 SVM (SVR)", usd(outputs["SVM (SVR)"]))
        right.metric("🟠 ANN", usd(outputs["ANN"]))
        gap = abs(outputs["SVM (SVR)"] - outputs["ANN"])
        st.caption(f"Chênh lệch giữa 2 mô hình: **{usd(gap)}**. Kết quả là hai ước lượng độc lập, không phải khoảng tin cậy.")
        st.bar_chart(pd.DataFrame({"Giá dự đoán (USD)": outputs}), horizontal=True, height=210)
        outside = check_outside_training_range(train_data, values)
        if outside:
            st.warning("Một hoặc nhiều thông số nằm ngoài phạm vi dữ liệu huấn luyện. Khi đó độ tin cậy của dự đoán có thể giảm:\n\n" + "\n\n".join(f"• {x}" for x in outside))
        st.info("Giá dự đoán chỉ phục vụ học tập trên dữ liệu lịch sử tại King County, không dùng để định giá giao dịch thực tế.")
        out_df = pd.DataFrame([{
            "DienTich_m2": values["DienTich"],
            "SoPhongNgu": values["SoPhongNgu"],
            "KhoangCach_km": values["KhoangCach"],
            "SVR_USD": outputs["SVM (SVR)"],
            "ANN_USD": outputs["ANN"],
        }])
        st.download_button("⬇️ Lưu kết quả vừa dự đoán (.csv)", data=out_df.to_csv(index=False).encode("utf-8-sig"), file_name="DuDoan_GiaNha_Nhom2.csv", mime="text/csv")
    else:
        st.caption("💡 Ví dụ: nhập 150 m², 3 phòng ngủ, cách Seattle 15 km rồi bấm **Dự đoán ngay**.")

elif page == "📊 Đánh giá & so sánh":
    st.subheader("Đánh giá hai thuật toán trên cùng 30 căn nhà Test")
    st.caption("Chỉ số được tính trực tiếp từ các mô hình được huấn luyện trên 119 căn nhà Train. Cùng 5-fold CV để lựa chọn siêu tham số.")
    view_metrics = metrics[["MAE (USD)", "MSE (USD²)", "RMSE (USD)", "R²"]].copy()
    view_metrics.index.name = "Thuật toán"
    st.dataframe(
        view_metrics.style.format({"MAE (USD)": "{:,.0f}", "MSE (USD²)": "{:,.0f}", "RMSE (USD)": "{:,.0f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )
    rmse_winner = metrics["RMSE (USD)"].idxmin()
    mae_winner = metrics["MAE (USD)"].idxmin()
    r2_winner = metrics["R²"].idxmax()
    p1, p2, p3 = st.columns(3)
    p1.metric("RMSE thấp nhất", rmse_winner)
    p2.metric("MAE thấp nhất", mae_winner)
    p3.metric("R² cao nhất", r2_winner)
    st.caption("MAE, MSE, RMSE càng thấp càng tốt; R² càng gần 1 càng tốt. Đây là đánh giá hồi quy, nên không sử dụng Accuracy/Precision/Recall/F1-score.")

    st.markdown("#### So sánh RMSE")
    st.bar_chart(metrics[["RMSE (USD)"]], horizontal=True, height=210)

    st.markdown("#### Giá thực tế và giá dự đoán")
    st.caption("Các điểm càng gần đường chéo giá thực = giá dự đoán thì sai số càng nhỏ.")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y_actual = predictions[TARGET].to_numpy()
    y_svr = predictions["SVM (SVR)"].to_numpy()
    y_ann = predictions["ANN"].to_numpy()
    min_v = float(min(y_actual.min(), y_svr.min(), y_ann.min()))
    max_v = float(max(y_actual.max(), y_svr.max(), y_ann.max()))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1.5, label="Dự đoán hoàn hảo")
    ax.scatter(y_actual, y_svr, label="SVM (SVR)", alpha=0.68, s=35)
    ax.scatter(y_actual, y_ann, label="ANN", alpha=0.68, s=35)
    ax.set(xlabel="Giá thực tế (USD)", ylabel="Giá dự đoán (USD)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Bảng dự đoán trên tập kiểm tra")
    shown = predictions.rename(columns={
        "MaBanGhi": "Mã căn nhà",
        "DienTich": "Diện tích (m²)",
        "SoPhongNgu": "Phòng ngủ",
        "KhoangCach": "Khoảng cách (km)",
        "GiaNha": "Giá thực tế (USD)",
        "SVM (SVR)": "Dự đoán SVR (USD)",
        "ANN": "Dự đoán ANN (USD)",
    }).reset_index(drop=True)
    shown["Sai số SVR (USD)"] = (shown["Dự đoán SVR (USD)"] - shown["Giá thực tế (USD)"]).abs()
    shown["Sai số ANN (USD)"] = (shown["Dự đoán ANN (USD)"] - shown["Giá thực tế (USD)"]).abs()
    st.dataframe(shown.head(5).style.format(precision=1, thousands=","), use_container_width=True)
    with st.expander("Hiển thị đủ 30 dòng Test"):
        st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Tải toàn bộ kết quả tập Test (.csv)", shown.to_csv(index=False).encode("utf-8-sig"), file_name="Nhom2_KetQua_30CanNha.csv", mime="text/csv")
    with st.expander("Thông số chọn bằng cross-validation"):
        for name, params in tuning.items():
            st.write(f"**{name}** · RMSE CV: {usd(metrics.loc[name, 'RMSE CV (USD)'])}")
            st.json({key.split("__")[-1]: str(value) for key, value in params.items()})
    st.info("Một số giá trị ANN có thể khác đôi chút so với bản báo cáo PDF khi thay đổi phiên bản thư viện hoặc môi trường chạy. App luôn tính lại các chỉ số từ chính dự đoán được hiển thị.")

elif page == "🧪 Dữ liệu & phương pháp":
    st.subheader("Dữ liệu và quy trình huấn luyện")
    st.write("Dữ liệu là **149 dòng đầu** từ bộ dữ liệu giao dịch nhà ở **King County, Washington (2014–2015)**. App sử dụng dữ liệu đã xử lý từ ba file Excel của Nhóm 2, không tự tạo giá giả lập.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tổng dữ liệu", f"{len(full_data)}")
    k2.metric("Huấn luyện", f"{len(train_data)}")
    k3.metric("Kiểm tra", f"{len(test_data)}")
    k4.metric("Biến đầu vào", len(FEATURES))
    st.markdown("#### Các thuộc tính mô hình sử dụng")
    data_desc = pd.DataFrame([
        ["DienTich", "Diện tích sinh hoạt (m²)", "Đầu vào X₁"],
        ["SoPhongNgu", "Số phòng ngủ", "Đầu vào X₂"],
        ["KhoangCach", "Khoảng cách đường chim bay đến trung tâm Seattle (km)", "Đầu vào X₃"],
        ["GiaNha", "Giá giao dịch nhà (USD)", "Đầu ra y"],
    ], columns=["Tên cột", "Ý nghĩa", "Vai trò"])
    st.dataframe(data_desc, use_container_width=True, hide_index=True)
    st.markdown("#### 10 dòng dữ liệu Train đầu tiên")
    st.dataframe(train_data[["MaBanGhi"] + FEATURES + [TARGET]].head(10), hide_index=True, use_container_width=True)
    st.markdown("#### Quy trình")
    st.markdown("""
1. **Thu thập dữ liệu:** Lấy 149 bản ghi từ bộ dữ liệu nhà ở King County.
2. **Tiền xử lý:** Chuyển diện tích từ ft² sang m², tính khoảng cách từ tọa độ nhà đến trung tâm Seattle, lựa chọn 3 biến đầu vào.
3. **Chia dữ liệu:** 119 dòng Train và 30 dòng Test, không trùng mã căn nhà (tỷ lệ xấp xỉ 80/20).
4. **Chuẩn hóa và tìm siêu tham số:** Dùng `StandardScaler` trong `Pipeline`, kết hợp `GridSearchCV` với 5-fold CV trên Train.
5. **Huấn luyện:** SVM sử dụng **SVR kernel RBF**; ANN sử dụng **MLPRegressor** với hai tầng ẩn.
6. **Đánh giá:** Cùng tập Test, dùng MAE, MSE, RMSE và R² để so sánh.
""")
    st.write("**Giá trị đầu vào trong dữ liệu Train:**")
    ranges = pd.DataFrame({
        "Thuộc tính": ["Diện tích (m²)", "Phòng ngủ", "Khoảng cách (km)"],
        "Nhỏ nhất": [train_data[col].min() for col in FEATURES],
        "Lớn nhất": [train_data[col].max() for col in FEATURES],
    })
    st.dataframe(ranges, hide_index=True, use_container_width=True)
    with st.expander("Nguồn tham khảo và lưu ý"):
        st.markdown("""
- Nguồn mô tả dữ liệu: [GeoDa Center, King County House Sales](https://geodacenter.github.io/data-and-lab/KingCounty-HouseSales2015/).
- File dữ liệu gốc mà nhóm đã sử dụng: [housing.csv trên GitHub](https://raw.githubusercontent.com/46n/king-county-house-sales-analysis/main/data/housing.csv).
- Mẫu chỉ gồm 149 bản ghi, không đại diện cho toàn bộ thị trường bất động sản.
- Mục tiêu của ứng dụng là minh họa quá trình triển khai mô hình học máy.
""")

elif page == "📖 Hướng dẫn sử dụng":
    st.subheader("Hướng dẫn sử dụng ứng dụng")
    st.markdown("""
**Dự đoán giá nhà**

1. Chọn **🏠 Dự đoán giá nhà** ở thanh bên trái.
2. Điền **diện tích nhà (m²)**, **số phòng ngủ** và **khoảng cách đến trung tâm Seattle (km)**.
3. Bấm **✨ Dự đoán ngay**. Đọc giá USD do hai mô hình SVR và ANN dự đoán.
4. Có thể tải file CSV chứa thông tin vừa nhập cùng kết quả dự đoán.

**Kiểm tra độ chính xác**

Chọn **📊 Đánh giá & so sánh** để xem chỉ số MAE, MSE, RMSE, R², biểu đồ và tải kết quả 30 căn nhà Test.

**Trình bày báo cáo học thuật**

Chọn **🧪 Dữ liệu & phương pháp** để trình bày nguồn dữ liệu, cách xử lý, phân chia Train/Test và các mô hình sử dụng.
""")
    st.warning("App là sản phẩm thực hành học thuật, không phải công cụ thẩm định hoặc tư vấn đầu tư bất động sản.")

st.divider()
st.caption("Nhóm 2 · SVM & ANN · Đề tài thực hành huấn luyện mô hình AI · Dữ liệu lịch sử 2014–2015 tại King County, Washington")
