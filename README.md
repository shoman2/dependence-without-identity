# Dependence without identity — code and data pipeline

Replication code for *"Dependence without identity: Anonymous major customers and the blind spot in disclosed supply chains"* (working paper, under preparation for submission).

U.S. GAAP requires firms to disclose revenue from each customer that accounts for 10% or more of sales but not to name the customer. This repository builds a census of every major-customer fact tagged in SEC XBRL filings (2009–2026), classifies each customer as named or anonymous, links named customers to SEC registrants, validates anonymity against the full 10-K text, and reproduces all tables and figures of the paper: trends and decompositions of anonymity, the dollar-weighted "blind spot" of named-only supply-chain networks, partial-identification bounds for key customers, the AI-exposure difference-in-differences, and a link-recovery benchmark (learning-to-rank and graph autoencoders).

**Data.** Only public SEC EDGAR sources are used: Financial Statement and Notes data sets, the XBRL frames and company-facts APIs, and 10-K filings. Nothing proprietary is required. Raw and intermediate data are not committed; the pipeline downloads them.

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export SEC_USER_AGENT="Your Name you@example.com"   # required by the SEC fair-access policy
./run_all.sh
```

See [REPRODUCE.md](REPRODUCE.md) for details. Full collection takes several hours (rate-limited downloads); analyses take one to two hours on a laptop.

## Layout

| Path | Contents |
|---|---|
| `src/extract_fsn.py`, `build_gt.py`, `classify_members.py` | XBRL extraction, named/anonymous classification, registrant linking, ground-truth sets |
| `src/fetch_tenk_all.py`, `text_customers.py`, `build_gt2.py` | 10-K text validation and text-matched ground truth |
| `src/tables.py`, `robust.py`, `dark_network.py` | Descriptive tables, decompositions, blind-spot measures |
| `src/fetch_item1.py`, `did_anon.py`, `revision_*.py` | AI exposure, difference-in-differences, robustness and sensitivity |
| `src/capex_transmission.py`, `bounds.py` | Hyperscaler investment covariation, partial-identification bounds (LP) |
| `src/features.py`, `exp_lgbm.py`, `exp_gae.py`, `eval_linkpred.py` | Link-recovery benchmark |
| `src/check_numbers.py`, `sensitivity_linking.py` | Consistency check and linking-audit sensitivity |
| `results/` | Result tables (CSV/XLSX) and reviewed audit samples |
| `figures/` | Figures |

## Notes

- LightGBM and PyTorch are run in separate processes (on macOS, loading both OpenMP runtimes in one process can deadlock).
- Reviewed audit samples (`results/audit_*.csv`) contain sentences quoted from public 10-K filings.

## License

Code: MIT (see `LICENSE`). Underlying data are public SEC filings.
