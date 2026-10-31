"""Huấn luyện và đánh giá SVR + ANN cho đề tài dự đoán giá nhà Nhóm 2.

Tất cả phép chuẩn hóa được fit bên trong Pipeline bằng dữ liệu Train.
GridSearchCV chỉ dùng tập Train. Tập Test không dùng để chọn siêu tham số.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES = ["DienTich", "SoPhongNgu", "KhoangCach"]
TARGET = "GiaNha"
DISPLAY_NAMES = {
    "DienTich": "Diện tích nhà (m²)",
    "SoPhongNgu": "Số phòng ngủ",
    "KhoangCach": "Khoảng cách đến Seattle (km)",
    "GiaNha": "Giá nhà (USD)",
}


def read_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nạp bộ dữ liệu đã xử lý và xác minh phân chia Train / Test."""
    filenames = ["Nhom2_GiaNha.xlsx", "Nhom2_GiaNha_train.xlsx", "Nhom2_GiaNha_test.xlsx"]
    frames = [pd.read_excel(data_dir / filename, sheet_name="DuLieu") for filename in filenames]
    full, train, test = frames
    expected = FEATURES + [TARGET, "MaBanGhi"]
    for name, df in zip(filenames, frames):
        missing = set(expected) - set(df.columns)
        if missing:
            raise ValueError(f"File {name} thiếu cột: {sorted(missing)}")
        if df[expected].isnull().any().any():
            raise ValueError(f"File {name} có ô trống trong cột cần thiết")
        for column in FEATURES + [TARGET]:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Cột {column} không phải số trong file {name}")
    if len(train) + len(test) != len(full):
        raise ValueError("Tổng số dòng Train và Test không bằng dữ liệu đầy đủ")
    ids_full, ids_train, ids_test = [set(df["MaBanGhi"]) for df in (full, train, test)]
    if ids_train.intersection(ids_test) or ids_full != ids_train.union(ids_test):
        raise ValueError("Phân chia Train / Test sai hoặc có căn nhà bị trùng")
    if (full["DienTich"] <= 0).any() or (full[TARGET] <= 0).any():
        raise ValueError("Diện tích hoặc giá nhà không hợp lệ")
    return full, train, test


def _make_regressor(estimator: object) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("scale", StandardScaler()),
            ("model", estimator),
        ]),
        transformer=StandardScaler(),
    )


def build_models(train: pd.DataFrame, test: pd.DataFrame):
    """Khớp mô hình theo cùng phương pháp với notebook Colab của Nhóm 2."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    specifications = {
        "SVM (SVR)": (
            _make_regressor(SVR(kernel="rbf")),
            {
                "regressor__model__C": [1, 10, 100],
                "regressor__model__epsilon": [0.05, 0.1, 0.2],
            },
        ),
        "ANN": (
            _make_regressor(MLPRegressor(solver="lbfgs", max_iter=2000, random_state=42)),
            {
                "regressor__model__hidden_layer_sizes": [(8, 4), (16, 8)],
                "regressor__model__alpha": [0.1, 1, 10],
            },
        ),
    }
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    models = {}
    metrics = {}
    tuning = {}
    predictions = test[["MaBanGhi"] + FEATURES + [TARGET]].copy()
    for name, (estimator, grid) in specifications.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            search.fit(X_train, y_train)
        model = search.best_estimator_
        y_pred = model.predict(X_test)
        models[name] = model
        predictions[name] = y_pred
        metrics[name] = {
            "MAE (USD)": float(mean_absolute_error(y_test, y_pred)),
            "MSE (USD²)": float(mean_squared_error(y_test, y_pred)),
            "RMSE (USD)": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R²": float(r2_score(y_test, y_pred)),
            "RMSE CV (USD)": float(-search.best_score_),
        }
        tuning[name] = search.best_params_
    return models, pd.DataFrame(metrics).T, predictions, tuning


def predict_one(models: dict, area: float, bedrooms: int, distance: float) -> dict[str, float]:
    X_new = pd.DataFrame([{
        "DienTich": float(area),
        "SoPhongNgu": int(bedrooms),
        "KhoangCach": float(distance),
    }], columns=FEATURES)
    return {name: float(model.predict(X_new)[0]) for name, model in models.items()}


def check_outside_training_range(train: pd.DataFrame, values: dict) -> list[str]:
    outside = []
    for key, value in values.items():
        lo, hi = float(train[key].min()), float(train[key].max())
        if value < lo or value > hi:
            suffix = "" if key == "SoPhongNgu" else ""
            outside.append(f"{DISPLAY_NAMES[key]}: tập Train {lo:g}–{hi:g}, bạn nhập {value:g}{suffix}")
    return outside
