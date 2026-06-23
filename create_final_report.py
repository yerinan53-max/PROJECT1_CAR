# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

from app import FEATURES, FEATURE_LABELS, load_data, train_models


BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / "자동차_연비_예측_최종보고서.docx"


def set_font(run, size=10.5, bold=False, color=None):
    run.font.name = "맑은 고딕"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def set_paragraph_font(paragraph, size=10.5):
    for run in paragraph.runs:
        set_font(run, size=size)


def add_heading(doc, text, level):
    paragraph = doc.add_heading(level=level)
    run = paragraph.add_run(text)
    set_font(run, size=18 if level == 1 else 14 if level == 2 else 12, bold=True)
    return paragraph


def add_paragraph(doc, text):
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    set_font(run)
    paragraph.paragraph_format.line_spacing = 1.35
    return paragraph


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, fill=None):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(str(text))
    set_font(run, size=9.5, bold=bold)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if fill:
        shade_cell(cell, fill)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for index, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[index], header, bold=True, fill="D9EAF7")

    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            set_cell_text(cells[index], value)

    doc.add_paragraph("")
    return table


def add_bullet(doc, text):
    paragraph = doc.add_paragraph(style="List Bullet")
    run = paragraph.add_run(text)
    set_font(run)


def format_float(value):
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return value


def main():
    df = load_data()
    _, metrics, importance, model_results = train_models()

    doc = Document()
    styles = doc.styles
    styles["Normal"].font.name = "맑은 고딕"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("자동차 연비 예측 프로젝트 최종 보고서")
    set_font(run, size=20, bold=True, color=(21, 76, 121))

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Auto MPG 데이터 기반 회귀 모델 비교 및 Streamlit 앱 구현")
    set_font(run, size=11, color=(77, 95, 120))

    doc.add_paragraph("")

    add_heading(doc, "1. 프로젝트 개요", 1)
    add_paragraph(
        doc,
        "본 프로젝트는 자동차 제원 데이터를 활용하여 예상 연비(MPG)를 예측하는 머신러닝 모델을 학습하고, "
        "최종 선택 모델을 Streamlit 웹 앱으로 구현한 결과물이다. 사용자는 실린더 수, 배기량, 마력, 무게, "
        "가속력, 연식을 입력하여 예상 연비를 확인할 수 있다.",
    )

    add_heading(doc, "2. 제출 산출물 구성", 1)
    add_table(
        doc,
        ["구분", "파일 또는 내용", "설명"],
        [
            ["모델 학습 파일", "car_mpg_training.ipynb", "데이터 로드, 전처리, 모델 학습, 성능 비교 과정을 정리한 Jupyter Notebook"],
            ["Streamlit 앱", "app.py", "사용자 입력값으로 예상 MPG를 계산하는 웹 앱"],
            ["의존성 파일", "requirements.txt", "Streamlit Cloud 배포에 필요한 라이브러리 목록"],
            ["보고서", "자동차_연비_예측_최종보고서.docx", "프로젝트 수행 과정과 결과 정리"],
        ],
    )

    add_heading(doc, "3. 데이터 설명", 1)
    add_paragraph(
        doc,
        "데이터는 Seaborn 예제 데이터인 mpg.csv를 사용하였다. 원 데이터는 UCI Auto MPG Dataset 기반이며, "
        "본 프로젝트에서는 결측치를 제거한 뒤 mpg와 주요 제원 변수만 사용하였다.",
    )
    add_table(
        doc,
        ["항목", "내용"],
        [
            ["데이터셋", "Auto MPG Dataset"],
            ["데이터 개수", f"{len(df)}개"],
            ["예측 대상", "mpg: 자동차 연비"],
            ["입력 변수", ", ".join(FEATURES)],
            ["학습/테스트 분리", f"학습 {metrics['train_count']}개, 테스트 {metrics['test_count']}개"],
        ],
    )

    summary_rows = []
    for column in ["mpg", *FEATURES]:
        label = FEATURE_LABELS.get(column, "연비")
        summary_rows.append(
            [
                label,
                f"{df[column].mean():.2f}",
                f"{df[column].min():.2f}",
                f"{df[column].max():.2f}",
            ]
        )
    add_table(doc, ["변수", "평균", "최소", "최대"], summary_rows)

    add_heading(doc, "4. 모델 학습 과정", 1)
    add_paragraph(
        doc,
        "모델 학습은 car_mpg_training.ipynb에서 수행하였다. 수업에서 다룬 범위에 맞춰 "
        "선형회귀, 의사결정나무, 랜덤 포레스트, KNN 회귀 4개 모델을 비교하였다.",
    )
    add_table(
        doc,
        ["모델", "설명"],
        [
            ["선형회귀", "입력 변수와 연비 사이의 선형 관계를 학습하는 기본 회귀 모델"],
            ["의사결정나무", "조건 분기를 통해 데이터를 나누며 예측하는 트리 기반 모델"],
            ["랜덤 포레스트", "여러 개의 결정나무를 결합하여 예측 안정성을 높인 앙상블 모델"],
            ["KNN 회귀", "입력값과 가까운 이웃 데이터들의 평균을 활용해 예측하는 모델"],
        ],
    )

    add_heading(doc, "5. 모델 성능 비교", 1)
    add_paragraph(
        doc,
        "성능 평가는 R², MAE, RMSE를 기준으로 수행하였다. RMSE는 예측 오차를 나타내므로 값이 낮을수록 좋고, "
        "R²는 모델 설명력을 나타내므로 값이 높을수록 좋다.",
    )

    model_rows = []
    for _, row in model_results.iterrows():
        selected = "최종 모델" if row["모델"] == metrics["best_model_name"] else "비교 모델"
        model_rows.append(
            [
                row["모델"],
                format_float(float(row["R2"])),
                format_float(float(row["MAE"])),
                format_float(float(row["RMSE"])),
                selected,
            ]
        )
    add_table(doc, ["모델", "R²", "MAE", "RMSE", "선택 여부"], model_rows)

    add_paragraph(
        doc,
        f"비교 결과 RMSE가 가장 낮은 모델은 {metrics['best_model_name']}이며, "
        f"테스트 데이터 기준 RMSE {metrics['RMSE']}, R² {metrics['R2']}, MAE {metrics['MAE']}의 성능을 보였다. "
        "따라서 해당 모델을 Streamlit 앱의 최종 예측 모델로 사용하였다.",
    )

    add_heading(doc, "6. 특성 영향 분석", 1)
    add_paragraph(
        doc,
        "최종 선택 모델 기준으로 입력 변수가 예측에 얼마나 영향을 주는지 확인하였다. "
        "랜덤 포레스트는 feature_importances_ 값을 제공하므로 이를 특성 영향도로 사용하였다.",
    )
    importance_rows = []
    for _, row in importance.iterrows():
        importance_rows.append([row["label"], f"{float(row['importance']):.4f}"])
    add_table(doc, ["특성", "중요도"], importance_rows)

    add_heading(doc, "7. Streamlit 앱 구현", 1)
    add_paragraph(
        doc,
        "웹 앱은 Streamlit으로 구현하였다. 사용자가 결과를 쉽게 확인할 수 있도록 상단 히어로 영역, "
        "자동차 정보 입력 폼, 모델 성능 요약, 모델별 성능 비교표, 특성 영향 분석 영역으로 구성하였다.",
    )
    add_bullet(doc, "사용자 입력값: 실린더 수, 배기량, 마력, 무게, 가속력, 연식")
    add_bullet(doc, "출력값: 예상 연비(MPG), 최종 선택 모델, 성능 지표, 특성 영향도")
    add_bullet(doc, "Streamlit Cloud 배포를 고려하여 MariaDB 저장 기능은 기본 비활성화하였다.")
    add_bullet(doc, "로컬 환경에서 ENABLE_DB=true를 설정하면 MariaDB 저장 기능을 사용할 수 있다.")

    add_heading(doc, "8. 결론 및 향후 개선 방향", 1)
    add_paragraph(
        doc,
        "본 프로젝트는 Auto MPG 데이터를 활용하여 회귀 모델을 학습하고, 성능 비교를 통해 최종 모델을 선택한 뒤 "
        "Streamlit 앱으로 배포 가능한 형태까지 구현하였다. 제출 산출물은 모델 학습 노트북, 한글 보고서, Streamlit 앱으로 구성된다.",
    )
    add_bullet(doc, "추가 데이터가 확보되면 제조사, 차량명, 국가(origin) 등 범주형 변수를 포함해 예측 성능을 높일 수 있다.")
    add_bullet(doc, "모델별 하이퍼파라미터 튜닝을 수행하면 랜덤 포레스트나 KNN 회귀의 성능 개선을 기대할 수 있다.")
    add_bullet(doc, "클라우드 DB를 연결하면 Streamlit Cloud에서도 예측 기록 저장 기능을 사용할 수 있다.")

    add_heading(doc, "9. 참고 자료", 1)
    add_bullet(doc, "UCI Machine Learning Repository, Auto MPG Dataset")
    add_bullet(doc, "Seaborn Data Repository, mpg.csv")
    add_bullet(doc, "scikit-learn Documentation")
    add_bullet(doc, "Streamlit Documentation")

    doc.save(REPORT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
