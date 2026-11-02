"""회계연도별 사업개요 문장 임베딩(all-MiniLM-L6-v2, 384차원, L2 정규화) → data/interim/emb/fy{연도}.npz
torch 를 쓰므로 lightgbm 프로세스와 분리해 미리 계산한다."""
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from features import text_table  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "interim" / "emb"
OUT.mkdir(parents=True, exist_ok=True)
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")
t = text_table()
for fy, d in t.groupby("fy"):
    if len(d) < 50:
        continue
    E = m.encode(d["value"].str[:1500].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    np.savez(OUT / f"fy{fy}.npz", ciks=d["cik"].to_numpy(), E=E.astype(np.float32))
    print(fy, len(d), flush=True)
