# ChromaGRT–SMRT

Reproduction code for *When Does Off-Domain Data Help? Pretraining vs. Joint Training for Retention Time Prediction*.

## Setup

Download the ZIP from the [Zenodo data deposit](https://doi.org/10.5281/zenodo.24524265) and extract it. The archive contains the curated inputs, fixed folds, molecular descriptors, SMRT checkpoints and archived predictions. The reserved DOI resolves once the deposit is published.

Run the following from this repository. Set `DATA` to the extracted directory containing `reportrt-random-cc/`, `reportrt-bm/` and `smrt/`.

```bash
export DATA=/absolute/path/to/extracted-data
git clone https://github.com/ceu-biolab/ChromaGRT.git ../ChromaGRT
git -C ../ChromaGRT checkout 3ddee7ff7c5d4225d4636cc7b63ee4ba3d7fd216
conda env create -f ../ChromaGRT/environment.yml
conda activate chromagrt
```

Optionally, use ChromaGRT's RepoRT download and preprocessing workflow to obtain
the original benchmark inputs. See the [ChromaGRT README](https://github.com/ceu-biolab/ChromaGRT/blob/3ddee7ff7c5d4225d4636cc7b63ee4ba3d7fd216/README.md)
for download and preprocessing instructions.

## Train and evaluate

Each case selects a scenario, hidden width, fraction of RepoRT training observations and strategy. List the supported cases with `python experiments/paper_runs.py --help`. The examples below run joint training at 25% availability, molecular transfer at full availability and joint training with scaffold filtering.

```bash
python experiments/paper_runs.py random/h256/fraction_0.25/joint \
  --chromagrt-root ../ChromaGRT --target-dir "$DATA/reportrt-random-cc" \
  --smrt-dir "$DATA/smrt" --output-root outputs

python experiments/paper_runs.py random/h256/fraction_1/transfer \
  --chromagrt-root ../ChromaGRT --target-dir "$DATA/reportrt-random-cc" \
  --checkpoint "$DATA/checkpoints/h256_heads8_layers6_ffn512/model.pt" \
  --output-root outputs

python experiments/paper_runs.py bm/h512/fraction_1/joint \
  --chromagrt-root ../ChromaGRT --target-dir "$DATA/reportrt-bm" \
  --smrt-dir "$DATA/smrt" --output-root outputs
```

- Use `reportrt-random-cc` for random and CC and `reportrt-bm` for BM. Use RDKit 2025.09.6 for BM scaffold filtering.
- For `target-only`, omit the SMRT arguments. For `transfer`, select the architecture-matched `model.pt` and keep its sibling `pretrain_checkpoint.json` in the same directory. To regenerate a checkpoint from the frozen SMRT split, see `python experiments/pretrain_smrt.py --help`.
- Training uses a GPU by default. Use `--folds 0` to run one fold or `--folds 0 1 2 3 4 5 6 7 8 9` to request all ten explicitly. By default the runner uses ten folds except for Joint at width 128, where it uses folds 0–3 to match the archived predictions.
- Each fold writes test predictions and model artifacts under `outputs/<case>/fold_<n>/`. Use a fresh output root for a repeat run.
- The `random/h256/fraction_0.25/matched-updates` case repeats RepoRT training observations under Joint's validation cadence and optimizer-update cap. Early stopping remains active. The planned and observed counts are in [matched-budgets.json](experiments/matched-budgets.json).

## Regenerate metrics and the availability figure

Use the archived predictions to reproduce the analysis without retraining.

```bash
python experiments/paper_results.py \
  --predictions "$DATA/predictions" \
  --random-target "$DATA/reportrt-random-cc" \
  --smrt-train "$DATA/smrt/train_data.tsv" \
  --output outputs/analysis
```

The analysis writes pooled metrics, per-fold differences, the update-budget comparison, SMRT identity overlap and the availability figure in PDF and 600 dpi PNG. The archive also contains `historical-predictions/` for the comparison with the original ChromaGRT study.
