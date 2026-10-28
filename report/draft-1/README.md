# Report drafts and supporting analysis

The four `.tex` files are fragments for the main report. The two Method sections
and the Discussion section keep their roles; `results.tex` contains the Results
chapter. Event F1 label efficiency is the primary outcome, with the other metrics
used to explain trade-offs.

## Integration

Copy the five PDFs in `figures/` into the main report's `figures/` directory. Merge
`references.bib` into its bibliography, or add it with
`\addbibresource{references.bib}`. Existing citation keys are retained for papers
still cited. Avoid loading duplicate entries with the same keys.

The fragments use `amsmath`, `graphicx`, `booktabs`, `adjustbox`, and `biblatex`
(with `\autocite` and `\textcite`). Bibliography entries include paper URLs;
ensure the main report's bibliography style displays URLs if these should be
visible in the compiled report. Existing figure/table labels are retained.

No main document or TeX toolchain is available in this repository/environment,
so the fragments have been checked structurally but not compiled as a report.

## Reproduce the analysis and figures

From the repository root:

```sh
python experiments/4_active_learning_runs/report_analysis.py --refresh-learning-curves
```

This reads the existing JSON files without training or model inference. It
validates all 40 runs and 400 rounds, reconstructs each labeled pool from the
initial pool and subsequent queries, verifies checkpoint selection against
validation edit score, and checks all 240 numbers in the report's AULC tables.
It also verifies that the notebook maps strategy IDs to the correct labels.

The command executes `test_results.ipynb`, refreshes its saved outputs and four
learning-curve PDFs, and copies those PDFs into `report_draft/figures/`. The new
acquisition-composition figure is generated directly by the script. Without
`--refresh-learning-curves`, it only regenerates the analysis CSVs and acquisition
figure and performs the same data/table checks.

Dependencies: NumPy, pandas, and Matplotlib; refreshing the notebook additionally
requires nbformat, nbclient, IPython, and a working `python3` Jupyter kernel.

The CSVs in `experiments/4_active_learning_runs/report_analysis/` contain:

- `round_metrics.csv`: all ten metric/granularity combinations at every seed and round.
- `aulc_per_seed.csv` and `aulc_summary.csv`: exact-budget AULCs, sample standard
  deviations, absolute differences, and relative gains over random sampling.
- `paired_differences.csv` and `paired_summary.csv`: same-seed AULC differences
  against random sampling, their variability, and the number of positive differences.
- `threshold_crossings.csv`: first observed mean-curve crossings of 90%, 95%, and
  100% of the full-budget random-sampling mean, with no interpolation or smoothing.
- `early_acquisitions.csv`: composition of the first four query batches for each
  seed, excluding the initial pool. These batches take the labeled pool from
  approximately 10% to 50% of training frames. Counts are pooled within each seed
  before calculating rates; figure bars and error bars are their mean and sample SD.

F1 values in `round_metrics.csv` and the crossing scores use 0–1 units; edit
scores and all AULCs use 0–100 units. Differences in prose are calculated before
rounding, so they can differ by 0.01 from subtracting displayed table cells.
Acquisition density uses original training-frame counts, matching the budget,
not the downsampled inference timeline. Winner/error outcomes mean `winner`,
`forced-err`, and `unforced-err` (all outcomes other than `in`).

## Interpretation and sources

The acquisition analysis is retrospective and does not use held-out test labels.
It provides possible explanations, not causal evidence. No per-outcome prediction
errors or annotation times were measured. Five-seed variability is descriptive;
no statistical-significance claim is made. Budget crossings use the mean test
curve and are not a deployment stopping rule. The three granularities project
the same predictions; they are not separate model-training experiments.

The notebook's three misassigned labels (Fine UM, Fine EM, Grouped Fine EM) were
replaced with an explicit strategy-to-label mapping. Other pre-existing cell
sources were preserved. The report's AULC tables already had the correct numbers
and strategy assignments.

Papers used:

- [F³Set — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c8e2cd11daf2d0a8c68663d504a841d-Abstract-Conference.html): event attributes, granularities, and evaluation context.
- [Spotting Temporally Precise, Fine-Grained Events in Video — ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3387_ECCV_2022_paper.php): precise event spotting.
- [Towards Active Learning for Action Spotting in Association Football Videos — CVPR Workshops 2023](https://arxiv.org/abs/2304.04220): UM/EM acquisition and learning-curve evaluation. This is a workshop paper.
- [An Analysis of Active Learning Strategies for Sequence Labeling Tasks — EMNLP 2008](https://aclanthology.org/D08-1112/): AULC evaluation.
- [Towards Robust and Reproducible Active Learning Using Neural Networks — CVPR 2022](https://arxiv.org/abs/2002.09564): variability of active-learning gains across experimental settings.

The exact formulas and counting rules in the Method sections follow this
repository's query and test-evaluation implementations. They should not be
interpreted as universal definitions of the metrics.
