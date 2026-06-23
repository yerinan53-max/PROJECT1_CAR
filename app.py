import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from db import DB_CONFIG, get_recent_predictions, init_database, save_prediction


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mpg.csv"
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
ENABLE_DB = os.getenv("ENABLE_DB", "false").lower() == "true"
MODEL_VERSION = "no-gradient-boosting-v2"

FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]

FEATURE_LABELS = {
    "cylinders": "실린더 수",
    "displacement": "배기량",
    "horsepower": "마력",
    "weight": "무게",
    "acceleration": "가속력",
    "model_year": "연식",
}

DEFAULT_VALUES = {
    "cylinders": 4.0,
    "displacement": 140.0,
    "horsepower": 90.0,
    "weight": 2400.0,
    "acceleration": 15.0,
    "model_year": 82.0,
}


def inject_custom_css():
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f4f7fb;
            --ink: #162033;
            --muted: #5d6f86;
            --line: #dfe7f1;
            --field-line: #cbd7e6;
            --brand: #1769aa;
            --brand-dark: #12588f;
            --teal: #1f7a8c;
            --hero-start: #154c79;
            --hero-end: #1f7a8c;
        }

        .stApp {
            background: var(--page-bg);
            color: var(--ink);
            font-family: Arial, "Malgun Gothic", sans-serif;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 40px;
            padding-bottom: 48px;
        }

        h1, h2, h3, p {
            margin-top: 0;
            letter-spacing: 0;
        }

        h1 {
            margin-bottom: 12px;
            color: white;
            font-size: 36px;
            font-weight: 800;
        }

        h2, h3 {
            color: var(--ink);
            font-weight: 800;
        }

        .hero-band {
            margin-bottom: 24px;
            padding: 32px;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--hero-start), var(--hero-end));
            color: white;
        }

        .hero-eyebrow {
            margin: 0 0 8px;
            color: #c9eff7;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .hero-band {
            display: grid;
            grid-template-columns: 1fr 260px;
            gap: 24px;
            align-items: stretch;
        }

        .hero-copy {
            max-width: 720px;
            margin-bottom: 0;
            color: white;
            line-height: 1.7;
            font-size: 16px;
        }

        .hero-score {
            display: grid;
            place-items: center;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.12);
            text-align: center;
        }

        .hero-score span,
        .hero-score small {
            color: #d9f7ff;
        }

        .hero-score strong {
            display: block;
            color: white;
            font-size: 56px;
            line-height: 1.1;
        }

        div[data-testid="stForm"],
        .result-card,
        div[data-testid="stDataFrame"],
        div[data-testid="stAlert"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: white;
            box-shadow: none;
        }

        div[data-testid="stForm"] {
            padding: 24px;
        }

        div[data-testid="stNumberInput"] label p {
            margin-bottom: 8px;
            color: #4d5f78;
            font-weight: 700;
        }

        div[data-testid="stNumberInput"] input {
            border: 1px solid var(--field-line);
            border-radius: 8px;
            background: white;
            font-size: 16px;
        }

        .stButton > button {
            width: 100%;
            margin-top: 20px;
            padding: 14px 18px;
            border: 0;
            border-radius: 8px;
            background: var(--brand);
            color: white;
            cursor: pointer;
            font-size: 17px;
            font-weight: 700;
            box-shadow: none;
        }

        .stButton > button:hover,
        .stButton > button:focus {
            border: 0;
            background: var(--brand-dark);
            color: white;
        }

        .model-panel,
        .feature-panel {
            padding: 24px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: white;
        }

        .model-panel h3,
        .feature-panel h3 {
            margin-bottom: 18px;
            font-size: 22px;
        }

        .best-model {
            margin-bottom: 12px;
            color: var(--ink);
        }

        .db-pill {
            display: inline-block;
            margin-bottom: 16px;
            padding: 8px 10px;
            border-radius: 999px;
            background: #e7f5ec;
            color: #17623a;
            font-size: 13px;
            font-weight: 700;
        }

        .metric-list {
            display: grid;
            gap: 12px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            padding: 14px;
            border-radius: 8px;
            background: #f4f7fb;
        }

        .metric-row span,
        .feature-row span,
        .note {
            color: var(--muted);
        }

        .metric-row strong {
            color: var(--ink);
        }

        .note {
            margin: 16px 0 0;
            line-height: 1.6;
        }

        div[data-testid="stMetric"] {
            padding: 14px;
            border-radius: 8px;
            background: #f4f7fb;
        }

        div[data-testid="stMetricLabel"] p,
        div[data-testid="stCaptionContainer"],
        .stMarkdown p {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 800;
        }

        div[data-testid="stDataFrame"] {
            overflow: hidden;
        }

        button[data-baseweb="tab"] {
            color: #4d5f78;
            font-weight: 700;
            border-radius: 0;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--brand);
            border-bottom-color: var(--brand);
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 24px;
        }

        hr {
            border-color: #dfe7f1;
            margin: 24px 0;
        }

        .feature-list {
            display: grid;
            gap: 14px;
            margin-top: 18px;
        }

        .feature-row {
            display: grid;
            grid-template-columns: 210px 1fr;
            gap: 18px;
            align-items: center;
        }

        .feature-row strong,
        .feature-row span {
            display: block;
        }

        .bar-wrap {
            height: 14px;
            overflow: hidden;
            border-radius: 999px;
            background: #edf2f7;
        }

        .bar {
            height: 100%;
            min-width: 2px;
            border-radius: 999px;
            background: var(--teal);
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 24px 16px;
            }

            .hero-band,
            .feature-row {
                grid-template-columns: 1fr;
            }

            .hero-band {
                padding: 24px;
            }

            h1 {
                font-size: 30px;
            }

            .result-value {
                font-size: 44px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return df[["mpg", *FEATURES]].dropna()


def build_models():
    return {
        "선형회귀": LinearRegression(),
        "의사결정나무": DecisionTreeRegressor(random_state=42, max_depth=5),
        "랜덤 포레스트": RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=8,
        ),
        "KNN 회귀": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5),
        ),
    }


def get_feature_importance(best_model):
    estimator = best_model
    if hasattr(best_model, "named_steps"):
        estimator = list(best_model.named_steps.values())[-1]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "feature_importances_"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
        importance_type = "coefficient"
    else:
        values = np.zeros(len(FEATURES))
        importance_type = "not_supported"

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "label": [FEATURE_LABELS[item] for item in FEATURES],
            "importance": values,
            "absolute": np.abs(values),
        }
    ).sort_values("absolute", ascending=False)

    max_value = importance_df["absolute"].max()
    importance_df["percent"] = (
        0 if max_value == 0 else (importance_df["absolute"] / max_value * 100).round()
    )

    return importance_df, importance_type


@st.cache_resource
def train_models(model_version=MODEL_VERSION):
    df = load_data()
    X = df[FEATURES]
    y = df["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_models = {}
    model_results = []

    for model_name, current_model in build_models().items():
        current_model.fit(X_train, y_train)
        y_pred = current_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        trained_models[model_name] = current_model
        model_results.append(
            {
                "모델": model_name,
                "R2": round(r2_score(y_test, y_pred), 3),
                "MAE": round(mean_absolute_error(y_test, y_pred), 3),
                "RMSE": float(round(rmse, 3)),
            }
        )

    model_results = sorted(model_results, key=lambda item: item["RMSE"])
    best_model_name = model_results[0]["모델"]
    best_model = trained_models[best_model_name]
    feature_importance, importance_type = get_feature_importance(best_model)

    metrics = {
        **model_results[0],
        "train_count": len(X_train),
        "test_count": len(X_test),
        "best_model_name": best_model_name,
        "importance_type": importance_type,
    }

    return best_model, metrics, feature_importance, pd.DataFrame(model_results)


@st.cache_resource
def setup_database():
    if not ENABLE_DB:
        return "disabled"

    try:
        init_database()
        return None
    except pymysql.MySQLError as exc:
        return str(exc)


def predict_mpg(model, values):
    input_df = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
    return round(float(model.predict(input_df)[0]), 2)


def render_prediction_form(model, db_error):
    st.subheader("자동차 정보 입력")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        values = {}
        for index, feature in enumerate(FEATURES):
            target_col = col1 if index % 2 == 0 else col2
            values[feature] = target_col.number_input(
                FEATURE_LABELS[feature],
                value=DEFAULT_VALUES[feature],
                step=1.0 if feature in {"cylinders", "model_year"} else 0.1,
            )

        submitted = st.form_submit_button("연비 예측하기")

    if not submitted:
        return None

    prediction = predict_mpg(model, values)

    if db_error == "disabled":
        st.info("예측이 완료되었습니다. 클라우드 배포에서는 DB 저장 기능을 사용하지 않습니다.")
    elif db_error:
        st.warning(f"예측은 완료했지만 MariaDB 저장은 건너뛰었습니다: {db_error}")
    else:
        try:
            save_prediction(values, prediction)
            st.success("예측 결과를 MariaDB에 저장했습니다.")
        except pymysql.MySQLError as exc:
            st.warning(f"예측은 완료했지만 MariaDB 저장 중 오류가 발생했습니다: {exc}")

    return prediction


def render_model_summary(metrics, model_results):
    st.markdown(
        f"""
        <div class="model-panel">
            <h3>모델 성능</h3>
            <p class="best-model">선택된 모델: <strong>{metrics['best_model_name']}</strong></p>
            <span class="db-pill">Streamlit Cloud 배포용</span>
            <div class="metric-list">
                <div class="metric-row"><span>R² 점수</span><strong>{metrics['R2']}</strong></div>
                <div class="metric-row"><span>MAE</span><strong>{metrics['MAE']}</strong></div>
                <div class="metric-row"><span>RMSE</span><strong>{metrics['RMSE']}</strong></div>
            </div>
            <p class="note">
                학습 데이터 {metrics['train_count']}개, 테스트 데이터 {metrics['test_count']}개로
                배운 범위의 회귀 모델을 비교 평가했습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_importance(feature_importance):
    st.markdown('<div class="feature-panel"><h3>특성 영향 분석</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="note">최종 선택된 모델 기준의 특성 영향도입니다. 값이 클수록 해당 변수가 연비 예측에 더 크게 반영됩니다.</p>',
        unsafe_allow_html=True,
    )
    max_value = feature_importance["absolute"].max() or 1
    rows = []
    for _, row in feature_importance.iterrows():
        percent = int(round((row["absolute"] / max_value) * 100))
        rows.append(
            f"""
            <div class="feature-row">
                <div>
                    <strong>{row['label']}</strong>
                    <span>중요도 {row['importance']:.4f}</span>
                </div>
                <div class="bar-wrap"><div class="bar" style="width: {percent}%"></div></div>
            </div>
            """
        )
    st.markdown('<div class="feature-list">' + ''.join(rows) + '</div></div>', unsafe_allow_html=True)


def render_recent_predictions(db_error):
    st.subheader("최근 예측 기록")

    if db_error == "disabled":
        st.info("클라우드 배포에서는 DB 저장 기능을 사용하지 않아 최근 기록을 표시하지 않습니다.")
        return

    if db_error:
        st.info("MariaDB 연결 오류가 있어 최근 예측 기록을 불러올 수 없습니다.")
        return

    try:
        rows = get_recent_predictions()
    except pymysql.MySQLError as exc:
        st.warning(f"최근 예측 기록 조회 중 오류가 발생했습니다: {exc}")
        return

    if not rows:
        st.info("아직 저장된 예측 기록이 없습니다.")
        return

    df = pd.DataFrame(rows).rename(
        columns={
            "created_at": "시간",
            "cylinders": "실린더",
            "displacement": "배기량",
            "horsepower": "마력",
            "weight": "무게",
            "acceleration": "가속력",
            "model_year": "연식",
            "predicted_mpg": "예측 MPG",
        }
    )
    st.dataframe(df.drop(columns=["id"], errors="ignore"), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="자동차 연비 예측", page_icon="CAR", layout="wide")
    inject_custom_css()

    model, metrics, feature_importance, model_results = train_models()
    db_error = setup_database()

    last_prediction = st.session_state.get("last_prediction")
    hero_value = "?" if last_prediction is None else last_prediction
    hero_unit = "값을 입력해보세요" if last_prediction is None else "MPG"

    st.markdown(
        f"""
        <section class="hero-band">
            <div>
                <p class="hero-eyebrow">Linear Regression Project</p>
                <h1>자동차 연비 예측</h1>
                <p class="hero-copy">
                    자동차의 실린더 수, 배기량, 마력, 무게, 가속력, 연식을 입력하면
                    여러 회귀 모델 중 가장 성능이 좋은 모델이 예상 연비(MPG)를 계산합니다.
                </p>
            </div>
            <div class="hero-score">
                <span>예상 연비</span>
                <strong>{hero_value}</strong>
                <small>{hero_unit}</small>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if db_error == "disabled":
        st.caption("클라우드 배포용으로 DB 저장 기능은 꺼져 있습니다.")
    elif db_error:
        st.error(f"MariaDB 연결 오류: {db_error}")
    else:
        st.caption(f"MariaDB `{DB_CONFIG['database']}` 데이터베이스에 예측 기록을 저장합니다.")

    left, right = st.columns([1.5, 1])
    with left:
        prediction = render_prediction_form(model, db_error)
        if prediction is not None:
            st.session_state["last_prediction"] = prediction

    with right:
        render_model_summary(metrics, model_results)

    st.divider()

    st.subheader("모델별 성능 비교")
    st.caption("RMSE가 낮을수록 예측 오차가 작고, R² 점수가 높을수록 모델 설명력이 좋습니다.")
    st.dataframe(model_results, use_container_width=True, hide_index=True)

    render_recent_predictions(db_error)
    render_feature_importance(feature_importance)


if __name__ == "__main__":
    main()
