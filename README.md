# ASR Related Languages for Low-Resource IPA ASR

This repository contains the code used for a set of experiments on **training-language selection for low-resource cross-lingual IPA-based ASR**. The project asks a practical question: when building a small multilingual ASR model for an unseen low-resource target language, which training languages should be chosen?

The experiments compare four selection strategies:

- **Genealogical relatedness**
- **Geographic proximity**
- **Phonological inventory overlap**
- **Random baseline**

The pipeline combines **Common Voice** audio and **VoxCommunis** TextGrid alignments, builds multilingual training datasets, trains **Wav2Vec2-CTC** IPA ASR models, and evaluates transfer to unseen target languages.

---

## Repository status

This is a research codebase rather than a packaged software release. The scripts reflect the actual experimental workflow used for the project and currently assume:

- local copies of external datasets
- hard-coded filesystem paths in several places
- an HPC-style environment for large-scale downloads and training

The code is still useful for replication and adaptation, but most users will need to edit paths before running it end to end.

---

## What this repository does

The repository contains two connected workflows:

1. **Language selection and metadata preparation**
   - enrich a language list with Glottolog and PHOIBLE IDs
   - merge in genealogical and geographic metadata
   - normalize PHOIBLE inventories
   - compute candidate training-language rankings for each target language

2. **ASR data preparation, training, and evaluation**
   - download Common Voice and VoxCommunis data
   - extract archives
   - generate transcript/audio tables from TextGrids
   - create train/eval/test splits
   - normalize IPA transcripts
   - build Hugging Face datasets
   - create model vocabularies
   - train Wav2Vec2-CTC models
   - evaluate on unseen and seen languages
   - summarize and model the results

---

## Repository layout

A recommended organization for GitHub is:

```text
.
├── README.md
├── scripts/
│   ├── metadata/
│   │   ├── get_ids.py
│   │   ├── data_extraction.py
│   │   ├── collapse_phoible.py
│   │   ├── count_textgrids.py
│   │   ├── get_training_langs_gptedit.py
│   │   └── get_phoible_bib.py
│   ├── data_prep/
│   │   ├── download_data.py
│   │   ├── download_textgrids.py
│   │   ├── unzip_data.py
│   │   ├── gen_transcripts.py
│   │   ├── set_splits_and_transcripts.py
│   │   ├── preprocess_transcripts.py
│   │   ├── create_datasets.py
│   │   └── vocab_builder.py
│   ├── training/
│   │   └── mipa_main_gpt_edit.py
│   ├── evaluation/
│   │   ├── all_model_eval.py
│   │   ├── all_model_eval_seen.py
│   │   └── result_analysis.py
│   └── stats/
│       └── regression_model.R


---

## Setup

### 1. Create an environment

A Python environment with the following kinds of packages is needed:

- `pandas`
- `datasets`
- `transformers`
- `torch`
- `soundfile`
- `scipy`
- `jiwer`
- `huggingface_hub`
- `geopy`
- `tqdm`
- `regex`
- `requests`
- `bibtexparser`
- `datacollective` or equivalent access to the dataset download API

The R analysis script additionally uses:

- `lme4`
- `effects`
- `mgcv`
- `mgcViz`
- `ggplot2`
- `lmerTest`
- `car`
- `sandwich`

### 2. Install Python dependencies

Example with `pip`:

```bash
pip install pandas datasets transformers torch soundfile scipy jiwer huggingface_hub geopy tqdm regex requests bibtexparser
```

If you use the MDC / Data Collective downloader, install its client in the same environment.

### 3. Prepare external resources

You will need local access to:

- a project language list, e.g. `rel_langs.tsv`
- Glottolog metadata
- PHOIBLE CLDF files
- a dataset ID table, e.g. `mdc_ids.tsv`
- Common Voice data
- VoxCommunis TextGrid archives

### 4. Edit paths

Most scripts currently use absolute paths such as:

```text
/scratch/jliebl/ASR_ComputEL/...
```

Before running the pipeline, update those paths to match your environment.

---

## Quick start

This is the shortest practical guide to the full workflow.

### A. Prepare language metadata and select training languages

Run:

```bash
python scripts/metadata/get_ids.py
python scripts/metadata/data_extraction.py
python scripts/metadata/collapse_phoible.py
python scripts/metadata/get_training_langs_gptedit.py
```

This stage enriches the language table with Glottolog and PHOIBLE metadata, normalizes the PHOIBLE inventories, and prints the candidate training languages for each target language and selection strategy.

### B. Download and extract the data

Run:

```bash
python scripts/data_prep/download_data.py
python scripts/data_prep/download_textgrids.py
python scripts/data_prep/unzip_data.py
```

This stage downloads Common Voice audio and VoxCommunis TextGrids, then extracts the archives.

### C. Build transcript/audio tables and create data splits

Run:

```bash
python scripts/data_prep/gen_transcripts.py
python scripts/data_prep/set_splits_and_transcripts.py
python scripts/data_prep/preprocess_transcripts.py
```

This stage parses the TextGrids, creates per-language transcript/audio tables, samples the train/eval/test partitions, and normalizes the IPA transcripts.

### D. Build model datasets and vocabularies

Run:

```bash
python scripts/data_prep/create_datasets.py
python scripts/data_prep/vocab_builder.py
```

This stage creates model-specific Hugging Face datasets and writes one `vocab.json` per model.

### E. Train a model

Example:

```bash
python scripts/training/mipa_main_gpt_edit.py \
  --train_ds_path /path/to/Datasets/hsb_geo_tr \
  --test_ds_path /path/to/Datasets/hsb_geo_te \
  --vocab_path /path/to/Models/hsb_geo/vocab.json \
  --out_dir /path/to/Models/hsb_geo \
  --batch_size 4 \
  --epochs 10 \
  --lr 1e-4
```

Repeat this for each model condition.

### F. Evaluate models

Run:

```bash
python scripts/evaluation/all_model_eval.py
python scripts/evaluation/all_model_eval_seen.py
python scripts/evaluation/result_analysis.py
```

This stage evaluates models on unseen target languages, optionally evaluates them on seen languages, and computes descriptive summaries.

### G. Fit the statistical model

Run in R:

```r
source("scripts/stats/regression_model.R")
```

This stage fits the mixed-effects regression used for the final analysis.

---

## Full pipeline order

The intended order of execution is:

1. `get_ids.py`
2. `data_extraction.py`
3. `collapse_phoible.py`
4. `get_training_langs_gptedit.py`
5. `download_data.py`
6. `download_textgrids.py`
7. `unzip_data.py`
8. `gen_transcripts.py`
9. `set_splits_and_transcripts.py`
10. `preprocess_transcripts.py`
11. `create_datasets.py`
12. `vocab_builder.py`
13. `mipa_main_gpt_edit.py`
14. `all_model_eval.py`
15. `all_model_eval_seen.py` *(optional but recommended sanity check)*
16. `result_analysis.py`
17. `regression_model.R`

Two auxiliary scripts are not required for the core pipeline:

- `count_textgrids.py`
- `get_phoible_bib.py`

---

## Script reference

## Metadata and language-selection scripts

### `get_ids.py`

**Purpose**  
Adds Glottolog and PHOIBLE identifiers to the project language list.

**Inputs**
- `rel_langs.tsv`
- `Phoible_Data/cldf/languages.csv`
- `Glottolog_Data/languages.csv`

**Outputs**
- `rel_langs_w_ids.tsv`

**Role**  
First metadata-enrichment step. Required before geographic, genealogical, and phonological similarity calculations.

---

### `data_extraction.py`

**Purpose**  
Adds Glottolog classification and coordinates to the language table.

**Inputs**
- `rel_langs_w_ids.tsv`
- `Glottolog_Data/values.csv`
- `Glottolog_Data/languages.csv`

**Outputs**
- `rel_langs_w_ids_and_data.tsv`

**Role**  
Builds the enriched metadata table used to compute geographic and genealogical relatedness.

---

### `collapse_phoible.py`

**Purpose**  
Normalizes PHOIBLE segment labels and collapses equivalent inventory columns.

**Inputs**
- `rel_langs_phoible_inventories.tsv`

**Outputs**
- `collapsed_rel_langs_phoible_inventories.tsv`

**Role**  
Creates the cleaned inventory table used for phonological-overlap calculations.

---

### `count_textgrids.py`

**Purpose**  
Counts `.TextGrid` files in VoxCommunis zip archives without fully extracting them.

**Inputs**
- Hugging Face dataset repo `pacscilab/VoxCommunis`

**Outputs**
- printed table
- optional `rel_langs_tg_counts.tsv`

**Role**  
Auxiliary corpus-inspection script for estimating TextGrid coverage.

---

### `get_training_langs_gptedit.py`

**Purpose**  
Computes candidate training languages for each target language under the genealogical, geographic, and phonological strategies.

**Inputs**
- `rel_langs_full_df.tsv`
- `collapsed_rel_langs_phoible_inventories.tsv`

**Outputs**
- printed top-ranked candidate languages by strategy
- printed random-baseline relationship tables

**Role**  
Main language-selection script. Used to determine which languages go into each model condition.

---

### `get_phoible_bib.py`

**Purpose**  
Extracts the bibliography entries corresponding to the PHOIBLE inventories used in the project.

**Inputs**
- `rel_langs_full_df.tsv`
- `Phoible_Data/cldf/values.csv`
- `Phoible_Data/cldf/sources.bib`

**Outputs**
- `inventory_sources.bib`

**Role**  
Paper-support utility. Not needed for training or evaluation.

---

## Data download and preprocessing scripts

### `download_data.py`

**Purpose**  
Downloads Common Voice datasets from the MDC / Data Collective source.

**Inputs**
- `mdc_ids3.tsv`

**Expected columns**
- `Dataset_ID`
- `CV_ID`

**Outputs**
- downloaded Common Voice datasets on disk

**Role**  
First raw-data acquisition step for audio.

---

### `download_textgrids.py`

**Purpose**  
Downloads VoxCommunis TextGrid zip archives for the project languages.

**Inputs**
- `mdc_ids3.tsv`

**Expected columns**
- `CV_ID`
- `VC_ID`

**Outputs**
- downloaded VoxCommunis `.zip` archives

**Role**  
Second raw-data acquisition step, providing aligned phone-level TextGrids.

---

### `unzip_data.py`

**Purpose**  
Extracts Common Voice `.tar.gz` archives and VoxCommunis `.zip` archives.

**Inputs**
- downloaded archive files in the raw data directory

**Outputs**
- extracted CV and VC folders

**Role**  
Makes downloaded data accessible to the downstream parsing scripts.

---

### `gen_transcripts.py`

**Purpose**  
Creates per-language transcript/audio TSV files by parsing TextGrids.

**Inputs**
- extracted VoxCommunis TextGrid folders
- extracted Common Voice audio folders
- `mdc_ids3.tsv`

**Outputs**
- `Dataframes/full/{lang}_data.tsv`

**Output columns**
- `Audio`
- `Transcript`

**Role**  
Core corpus-construction step. Converts TextGrid phone alignments into transcript/audio tables.

---

### `set_splits_and_transcripts.py`

**Purpose**  
Creates train/eval/test splits from the full per-language TSVs.

**Inputs**
- `Dataframes/full/{lang}_data.tsv`

**Outputs**
- `Dataframes/train/{lang}_train.tsv`
- `Dataframes/eval/{lang}_eval.tsv`
- `Dataframes/test/{lang}_test.tsv`
- `char_set.txt`

**Role**  
Defines the data partitions used for training and evaluation.

**Split logic**
- Training languages: sample 2800 rows → 2000 train, 400 test, 400 eval
- Unseen target languages: sample 400 rows → test only

---

### `preprocess_transcripts.py`

**Purpose**  
Normalizes IPA transcripts in the split TSV files.

**Inputs**
- split train/eval/test TSV files

**Outputs**
- same TSV files rewritten in place

**New columns**
- `Unprocessed Transcript`
- `Transcript`

**Role**  
Standardizes transcript format before dataset creation and model training.

---

### `create_datasets.py`

**Purpose**  
Builds model-specific Hugging Face datasets from the cleaned split TSV files.

**Inputs**
- cleaned train TSVs
- cleaned eval TSVs

**Outputs**
Saved Hugging Face datasets such as:
- `rand_tr`, `rand_te`
- `hsb_geo_tr`, `hsb_geo_te`
- `hsb_fam_tr`, `hsb_fam_te`
- `hsb_phon_tr`, `hsb_phon_te`
- `lg_geo_tr`, `lg_geo_te`
- `lg_fam_tr`, `lg_fam_te`
- `lg_phon_tr`, `lg_phon_te`
- `tt_geo_tr`, `tt_geo_te`
- `tt_fam_tr`, `tt_fam_te`
- `tt_phon_tr`, `tt_phon_te`

**Role**  
Packages language-specific splits into strategy-specific model datasets.

---

### `vocab_builder.py`

**Purpose**  
Creates one character vocabulary per model condition.

**Inputs**
- saved HF train datasets
- saved HF eval datasets

**Outputs**
- `Models/{model}/vocab.json`

**Role**  
Prepares the tokenizer vocabulary used during Wav2Vec2-CTC training.

---

## Training and evaluation scripts

### `mipa_main_gpt_edit.py`

**Purpose**  
Trains Wav2Vec2-CTC IPA ASR models.

**Inputs**
Command-line arguments:
- `--train_ds_path`
- `--test_ds_path`
- `--vocab_path`
- `--model_name_or_path`
- `--out_dir`
- `--batch_size`
- `--epochs`
- `--lr`

Optional:
- `--create_vocab`

**Outputs**
- trained model directory
- tokenizer / processor files
- saved model weights and config

**Role**  
Core training script. Loads HF datasets, processes audio, tokenizes transcripts, fine-tunes the model, and saves the trained checkpoint.

---

### `all_model_eval.py`

**Purpose**  
Evaluates trained models on unseen target-language test sets.

**Inputs**
- trained model directories
- target test files:
  - `hsb_test.tsv`
  - `lg_test.tsv`
  - `tt_test.tsv`

**Outputs**
Evaluation result files such as:
- `hsb_fam_eval_results_hsb.tsv`
- `hsb_geo_eval_results_hsb.tsv`
- `hsb_phon_eval_results_hsb.tsv`
- `rand_eval_results_hsb.tsv`

and corresponding files for Luganda and Tatar.

**Added columns**
- `Prediction`
- `CER scores`

**Role**  
Main target-language evaluation step.

---

### `all_model_eval_seen.py`

**Purpose**  
Evaluates trained models on languages seen during training.

**Inputs**
- trained model directories
- seen-language test files

**Outputs**
- `{model}_{lang}_seen_results.tsv`

**Role**  
Auxiliary sanity-check evaluation used to confirm that model behavior on unseen targets is not simply due to failed training.

---

### `result_analysis.py`

**Purpose**  
Computes descriptive summaries from evaluation outputs.

**Inputs**
- target-language evaluation TSVs from `all_model_eval.py`

**Outputs**
- printed mean CERs by strategy
- printed improvement-over-random summaries

**Role**  
Post-evaluation descriptive analysis.

---

## Statistical analysis script

### `regression_model.R`

**Purpose**  
Fits the inferential mixed-effects model for CER outcomes.

**Inputs**
- `merged_results.tsv`

**Expected columns**
- `CER.scores`
- `Utterance_Length`
- `Target_Lang`
- `Strategy`
- `Utterance_ID`

**Outputs**
- saved model object (`m1.Rda`)
- printed summary
- effects plot in the R session

**Role**  
Final inferential analysis used in the paper.

---

## Typical end-to-end workflow

A typical full run looks like this:

```bash
# 1. Metadata and language selection
python scripts/metadata/get_ids.py
python scripts/metadata/data_extraction.py
python scripts/metadata/collapse_phoible.py
python scripts/metadata/get_training_langs_gptedit.py

# 2. Download and extract data
python scripts/data_prep/download_data.py
python scripts/data_prep/download_textgrids.py
python scripts/data_prep/unzip_data.py

# 3. Build transcript/audio tables and splits
python scripts/data_prep/gen_transcripts.py
python scripts/data_prep/set_splits_and_transcripts.py
python scripts/data_prep/preprocess_transcripts.py

# 4. Build model datasets and vocabularies
python scripts/data_prep/create_datasets.py
python scripts/data_prep/vocab_builder.py

# 5. Train models
python scripts/training/mipa_main_gpt_edit.py --train_ds_path ... --test_ds_path ... --vocab_path ... --out_dir ...

# 6. Evaluate
python scripts/evaluation/all_model_eval.py
python scripts/evaluation/all_model_eval_seen.py
python scripts/evaluation/result_analysis.py
```

Then run the R model:

```r
source("scripts/stats/regression_model.R")
```

---

## Reproducibility notes

To improve portability, the code would benefit from:

- replacing hard-coded paths with CLI arguments or config files
- centralizing path and dataset definitions
- writing more intermediate outputs to disk in a standardized way
- separating reusable functions from script entry points
- adding a single orchestration script, `Makefile`, or workflow file

At present, the safest approach is to treat the scripts as an explicit research workflow and run them in the documented order.

---

## Acknowledgements

Parts of the training code were adapted from the MultIPA project. If you use it, please cite:

@misc{taguchi2023universal,
      title={Universal Automatic Phonetic Transcription into the International Phonetic Alphabet}, 
      author={Chihiro Taguchi and Yusuke Sakai and Parisa Haghani and David Chiang},
      year={2023},
      eprint={2308.03917},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


