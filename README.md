# ChromaGRT–SMRT (ESANN)

Code for the ESANN paper *When Does Off-Domain Data Help? Pretraining vs. Joint Training for Retention Time Prediction*. The current runner trains one target-only ChromaGRT fold, using retention time divided by the chromatographic program duration as the prediction target.

## Run one fold

Check out the pinned ChromaGRT revision and create its Conda environment:

```bash
git clone https://github.com/ceu-biolab/ChromaGRT.git ../ChromaGRT
git -C ../ChromaGRT checkout 3ddee7ff7c5d4225d4636cc7b63ee4ba3d7fd216
conda env create -f ../ChromaGRT/environment.yml
conda activate chromagrt
```

Run the following from this repository. Use the curated ESANN split directory containing `train_data.tsv`, `val_data.tsv` and `test_data.tsv`, and its matching molecular descriptor table. The Zenodo link will be added when the data are deposited.

```bash
python experiments/target_only.py \
  --chromagrt-root ../ChromaGRT \
  --fold-dir /path/to/esann-data/random/fold_0 \
  --molecular-descriptors /path/to/esann-data/molecular_descriptors.tsv \
  --scenario random \
  --output outputs/random/fold_0
```

The output directory must be new. It will contain `Results.tsv` with test predictions, the trained model and preprocessing metadata. Training uses a GPU by default. Use `--accelerator cpu --max-epochs 1` for a short check.

## Reproduce the paper experiments

The planned ESANN archive uses separate roots for random/CC, BM and the frozen
SMRT split. Each target root contains its descriptors and scenario folders
(\`random\`, \`cc\` or \`bm_scaffold\`). Use RDKit 2025.09.6 for the BM filter.

\`\`\`bash
python experiments/paper_runs.py --help
python experiments/paper_runs.py random/h256/fraction_0.25/joint \\
  --chromagrt-root ../ChromaGRT \\
  --target-dir /path/to/random-cc-data --smrt-dir /path/to/smrt-data \\
  --output-root outputs
\`\`\`

The case name selects the paper configuration and published folds. Use
\`--folds 0\` to run just fold zero. For BM, point \`--target-dir\` to the BM
root. For transfer, pass \`--checkpoint /path/to/model.pt\` with its sibling
\`pretrain_checkpoint.json\`. You can also generate both files using
\`experiments/pretrain_smrt.py\` with \`--chromagrt-root\`, \`--smrt-dir\` and
\`--output\`. The architecture flags are \`--hidden-dim\`, \`--num-heads\`,
\`--num-layers\` and \`--ffn-dim\`. The default is 256/8/6 with FFN 512.

The case \`random/h256/fraction_0.25/matched-updates\` repeats RepoRT examples
to match Joint's validation cadence and the maximum number of optimizer
updates per fold. Early stopping remains active. In the original runs this
control stopped below the cap in all ten folds. The planned and observed
updates are recorded in [matched-budgets.json](experiments/matched-budgets.json).

## Regenerate results without training

\`\`\`bash
python experiments/paper_results.py --predictions /path/to/predictions \\
  --random-target /path/to/random-cc-data \\
  --smrt-train /path/to/smrt-data/train_data.tsv --output outputs/analysis
\`\`\`

The prediction archive uses the same case and fold directory layout as
\`paper_runs.py\`. The analysis writes pooled metrics, per-fold differences,
the update-budget comparison, SMRT identity overlap and the availability
figure in PDF and 600 dpi PNG.
