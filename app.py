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
def train_models():
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

    if db_error:
        st.warning(f"예측은 완료했지만 MariaDB 저장은 건너뛰었습니다: {db_error}")
    else:
        try:
            save_prediction(values, prediction)
            st.success("예측 결과를 MariaDB에 저장했습니다.")
        except pymysql.MySQLError as exc:
            st.warning(f"예측은 완료했지만 MariaDB 저장 중 오류가 발생했습니다: {exc}")

    return prediction


def render_model_summary(metrics, model_results):
    st.subheader("모델 성능")
    st.write(f"최종 선택 모델: **{metrics['best_model_name']}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("R2", metrics["R2"])
    col2.metric("MAE", metrics["MAE"])
    col3.metric("RMSE", metrics["RMSE"])

    st.caption(
        f"학습 데이터 {metrics['train_count']}개, 테스트 데이터 {metrics['test_count']}개로 "
        "배운 범위의 회귀 모델을 비교했습니다. RMSE가 가장 낮은 모델을 최종 예측 모델로 사용합니다."
    )

    st.dataframe(model_results, use_container_width=True, hide_index=True)
    st.bar_chart(model_results.set_index("모델")["RMSE"])


def render_feature_importance(feature_importance):
    st.subheader("특성 영향 분석")

    chart_df = feature_importance[["label", "importance"]].rename(
        columns={"label": "특성", "importance": "중요도"}
    )
    st.dataframe(chart_df, use_container_width=True, hide_index=True)
    st.bar_chart(chart_df.set_index("특성")["중요도"])


def render_recent_predictions(db_error):
    st.subheader("최근 예측 기록")

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

    model, metrics, feature_importance, model_results = train_models()
    db_error = setup_database()

    st.title("자동차 연비 예측")
    st.write(
        "자동차의 실린더 수, 배기량, 마력, 무게, 가속력, 연식을 입력하면 "
        "배운 범위의 회귀 모델 중 가장 성능이 좋은 모델로 예상 연비(MPG)를 계산합니다."
    )

    if db_error:
        st.error(f"MariaDB 연결 오류: {db_error}")
    else:
        st.caption(f"MariaDB `{DB_CONFIG['database']}` 데이터베이스에 예측 기록을 저장합니다.")

    left, right = st.columns([1, 1])
    with left:
        prediction = render_prediction_form(model, db_error)

    with right:
        st.subheader("예상 연비")
        if prediction is None:
            st.metric("MPG", "-")
            st.caption("값을 입력하고 예측 버튼을 눌러주세요.")
        else:
            st.metric("MPG", prediction)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["모델 비교", "특성 분석", "최근 기록"])
    with tab1:
        render_model_summary(metrics, model_results)
    with tab2:
        render_feature_importance(feature_importance)
    with tab3:
        render_recent_predictions(db_error)


if __name__ == "__main__":
    main()
