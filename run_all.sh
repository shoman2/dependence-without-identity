#!/usr/bin/env bash
# 전체 재현 파이프라인 (공개 SEC 데이터만 사용). 순서대로 실행. 소요: 수집 단계 수 시간, 분석 단계 약 1~2시간.
# SEC fair-access: src/sec_client.py 의 UA 에 연락처 이메일을 넣고 요청 간격(0.125~0.15초)을 지킨다.
set -euo pipefail
PY=".venv/bin/python -W ignore"
# 1) 수집·추출
$PY src/extract_fsn.py          # 주석 데이터셋 79개 → 주요고객 팩트·연결 매출·제출정보
$PY src/fetch_fundamentals.py   # SEC frames 재무변수 + FF-12 매핑
$PY src/extract_text.py         # 사업개요 텍스트(노트) → 제품 유사도용
$PY src/embed_text.py           # 문장 임베딩 (torch, 별도 프로세스)
# 2) 관계·정답셋
$PY src/build_gt.py             # 관계 테이블, GT1, GT3
$PY src/fetch_tenk_all.py       # 익명 고객 10-K 본문 → 실명 문장
$PY src/build_gt2.py            # 본문 실명 공개율 추이, GT2
# 3) 기술통계·분해
$PY src/tables.py               # 표 1–3
$PY src/robust.py               # 위약·성향 분해·전환율
$PY src/dark_network.py         # 측정 맹점(표 6)
# 4) AI 노출·이중차분
$PY src/fetch_item1.py          # FY2021 10-K Item 1 키워드 노출
$PY src/did_anon.py
$PY src/revision_c1.py          # 균형·사전추세·SIC2×연도·매칭
$PY src/revision_r2.py          # 노출 분위·표 5(본 표본)·5사 LP 상한
$PY src/revision_r3.py          # 구간 차이·매칭 균형·좌측 절단
# 5) 설비투자 공변·상한
$PY src/fetch_quarterly.py
$PY src/fetch_nonai.py
$PY src/capex_transmission.py
$PY src/revision_c2.py          # 비AI 위약·AR 충격·차이 CI
$PY src/bounds.py
$PY src/revision_c345.py        # 진입 유형·공동 상한·공변량 이동·평가 흐름
# 6) 복원 벤치마크 (lightgbm 과 torch 는 같은 프로세스에 올리지 않는다: macOS libomp 교착)
$PY src/exp_lgbm.py; $PY src/exp_lgbm.py --nograph; $PY src/exp_lgbm.py --text
for ys in "2015 2018 2021 2024" "2016 2019 2022 2025" "2017 2020 2023"; do $PY src/exp_gae.py "$ys"; $PY src/exp_gae.py --text "$ys"; done
$PY src/eval_linkpred.py
# 7) 원고 수치 대조
$PY src/check_numbers.py
