# 자동차 연비 예측 Flask 앱

자동차 제원 데이터를 이용해 연비(MPG)를 예측하는 선형회귀 과제 예제입니다.

## 실행 방법

```bash
pip install -r requirements.txt
python app.py
```

실행 후 브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:5000/
```

## 데이터

기본 데이터는 Seaborn 예제 데이터인 `mpg.csv`를 사용합니다.
앱 실행 시 `data/mpg.csv` 파일이 없으면 아래 공개 CSV에서 자동으로 내려받습니다.

```text
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
```

원 데이터셋은 UCI Machine Learning Repository의 Auto MPG Dataset입니다.

```text
https://archive.ics.uci.edu/dataset/9/auto+mpg
```

## 입력값

- 실린더 수
- 배기량
- 마력
- 무게
- 가속력
- 연식

## 출력값

- 예상 연비(MPG)
- 모델 성능 지표: R², MAE, RMSE
- 회귀 계수 기반 특성 영향 분석
