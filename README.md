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
