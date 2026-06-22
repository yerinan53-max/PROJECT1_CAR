from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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


def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df = df[["mpg", *FEATURES]].dropna()
    return df


def train_model():
    df = load_data()
    X = df[FEATURES]
    y = df["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics = {
        "r2": round(r2_score(y_test, y_pred), 3),
        "mae": round(mean_absolute_error(y_test, y_pred), 3),
        "rmse": round(rmse, 3),
        "train_count": len(X_train),
        "test_count": len(X_test),
    }

    coefficients = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "label": [FEATURE_LABELS[item] for item in FEATURES],
                "coefficient": model.coef_,
                "absolute": np.abs(model.coef_),
            }
        )
        .sort_values("absolute", ascending=False)
        .to_dict("records")
    )

    return model, metrics, coefficients


app = Flask(__name__)
model, metrics, coefficients = train_model()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    form_values = {
        "cylinders": "4",
        "displacement": "140",
        "horsepower": "90",
        "weight": "2400",
        "acceleration": "15",
        "model_year": "82",
    }

    if request.method == "POST":
        form_values = {feature: request.form.get(feature, "") for feature in FEATURES}

        try:
            input_values = [float(form_values[feature]) for feature in FEATURES]
            input_df = pd.DataFrame([input_values], columns=FEATURES)
            prediction = round(float(model.predict(input_df)[0]), 2)
        except ValueError:
            error = "모든 입력칸에 숫자를 입력해주세요."

    return render_template(
        "index.html",
        features=FEATURES,
        labels=FEATURE_LABELS,
        values=form_values,
        prediction=prediction,
        error=error,
        metrics=metrics,
        coefficients=coefficients,
    )


if __name__ == "__main__":
    app.run(debug=True)
