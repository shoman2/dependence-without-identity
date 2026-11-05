# Reproduction guide — "Who is Customer A?"

All inputs are public SEC EDGAR data (Financial Statement and Notes data sets, XBRL frames and company-facts APIs, 10-K filings). No proprietary databases are used.

1. Python 3.12; install: `pandas numpy scipy scikit-learn statsmodels linearmodels lightgbm torch torch_geometric sentence-transformers rapidfuzz lxml pyarrow openpyxl requests matplotlib`.
2. Set your contact in the environment, e.g. `export SEC_USER_AGENT="Your Name you@example.com"` (or put it in a `.sec_user_agent` file at the repository root). The SEC requires a declared User-Agent; keep the built-in rate limit.
3. Run `./run_all.sh`. Raw downloads are cached under `data/raw/` (zips are deleted after extraction); intermediate files go to `data/interim/`, results to `results/`.
4. `src/check_numbers.py` re-derives key numbers from `results/` and checks them against the manuscript (the manuscript itself is not distributed with the code).

Key design notes: fiscal year = calendar year of the period end, minus one if the period ends before July; segment-attributed customer facts are used only when no consolidated fact exists; LightGBM and PyTorch run in separate processes.
