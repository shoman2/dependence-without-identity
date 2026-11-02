"""AI 인프라 노출도: 사업개요 임베딩과 'AI 인프라 공급' 설명문의 코사인 유사도.
기준 연도 = FY2021 (ChatGPT 이전, 처치 오염 방지). 연도별 값도 함께 저장(강건성).
torch 사용 → lightgbm 과 분리된 프로세스."""
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INT = Path(__file__).resolve().parents[1] / "data" / "interim"
SEEDS = [
    "We design and sell graphics processing units (GPUs) and AI accelerators used for training and inference of artificial intelligence models in data centers.",
    "We manufacture servers, storage systems and high-performance computing platforms for cloud data centers and AI workloads.",
    "We provide high-speed networking switches, interconnects and optical transceivers for hyperscale data centers.",
    "We supply power distribution, uninterruptible power supply and thermal management cooling equipment for data centers.",
    "We develop semiconductors, memory chips and advanced packaging used in artificial intelligence and high performance computing.",
]
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
q = m.encode(SEEDS, normalize_embeddings=True)
rows = []
for p in sorted((INT / "emb").glob("fy*.npz")):
    z = np.load(p)
    sims = z["E"] @ q.T
    rows.append(pd.DataFrame({"cik": z["ciks"], "fy": int(p.stem[2:]), "ai_exp": sims.max(1), "ai_exp_mean": sims.mean(1)}))
out = pd.concat(rows)
out.to_parquet(INT / "ai_exposure.parquet")
print(out.groupby("fy")["ai_exp"].describe()[["count", "mean", "50%", "max"]].round(3).tail(6))
