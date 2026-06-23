# 자동차 연비 예측 Streamlit 앱

자동차 제원 데이터를 사용해 예상 연비(MPG)를 예측하는 Streamlit 앱입니다.
여러 회귀 모델을 비교한 뒤 RMSE가 가장 낮은 모델로 예측하고, 예측 기록은 MariaDB에 저장합니다.

## 실행 방법

```powershell
pip install -r requirements.txt
streamlit run app.py
```

실행 후 브라우저에서 아래 주소로 접속합니다.

```text
http://localhost:8501
```

같은 와이파이에 있는 다른 컴퓨터에서 접속해야 하면 다음처럼 실행합니다.

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

다른 컴퓨터에서는 브라우저 주소창에 아래처럼 입력합니다.

```text
http://내컴퓨터IP:8501
```

## MariaDB 연결

앱 실행 시 `project1_car` 데이터베이스와 `predictions` 테이블을 자동으로 생성합니다.
기본 연결 정보는 `db.py`에 정의되어 있습니다.

```text
host: 127.0.0.1
port: 3306
user: root
database: project1_car
```

비밀번호를 바꿔야 하면 실행 전에 환경 변수를 설정합니다.

```powershell
$env:DB_PASSWORD="비밀번호"
streamlit run app.py
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
- 모델 성능 지표 R2, MAE, RMSE
- 모델별 성능 비교
- 최종 모델 기준 특성 영향 분석
- 최근 예측 기록
