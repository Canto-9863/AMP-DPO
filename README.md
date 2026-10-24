# AMP-DPO

Code and supporting data for antimicrobial peptide generation using a discrete diffusion model and direct preference optimization (DPO).

The workflow comprises protein-sequence pretraining, preference-based fine-tuning, and peptide generation with physicochemical evaluation and candidate selection. Pretrained and DPO model checkpoints are included, so generation can be performed without retraining.

## Repository structure

```text
AMP-DPO-main/
├── pretrain_discrete.py          # Protein-sequence pretraining
├── train_discrete.py             # DPO fine-tuning
├── generate_discrete.py          # Peptide generation and evaluation
├── config.py                    # Additional argument definitions
├── requirements.txt
├── 1_pretrain.bat                # Original Windows pretraining command
├── 1_train_discrete_gpu.bat      # Original Windows DPO training command
├── 1_generate_discrete_gpu.bat   # Original Windows generation command
├── models/                      # Diffusion, DPO, and Transformer modules
├── utils/                       # Sequence properties and evaluation
├── data/                        # Tokenizer, dataset loader, and AMP data
├── pretrain_data/
│   ├── *.py                     # UniProt download and preprocessing scripts
│   ├── config.json
│   └── data/
│       ├── processed_uniprot/   # Processed protein sequences
│       └── training_data/       # Train/validation/test batches and indices
└── checkpoints/
    ├── discrete_pretrain/       # Pretraining checkpoints and training history
    └── discrete_dpo/            # DPO checkpoint and training history
```

## Installation

Use a Python environment with PyTorch and the packages listed in `requirements.txt`. Run the following commands from the repository root:

```shell
python -m pip install -r requirements.txt
python -m pip install seaborn requests
```

`seaborn` is imported by the evaluation module, and `requests` is used by the UniProt download script. These packages supplement the original dependency list.

For GPU execution, install a PyTorch build compatible with the local CUDA environment. The main scripts select CUDA when available and otherwise use the CPU; `--device cpu` can be supplied explicitly. Training batch sizes in the supplied commands may require adjustment for available memory.

The dependency list specifies minimum versions rather than a locked environment. An exact Python/PyTorch/CUDA environment has not been established or tested for this submission package.

## Generate peptides using the included checkpoint

Run this command from the repository root. It uses the same generation settings as the supplied Windows batch file:

```shell
python generate_discrete.py --checkpoint checkpoints/discrete_dpo/discrete_amp_dpo_best.pt --num_samples 300 --num_strategies 10 --iterations 3 --batch_size 10 --output_dir 8generated_results --output_prefix discrete_amps --top_k 100 --seq_len_min 2 --seq_len_max 11
```

On Windows PowerShell, the supplied launcher can also be used:

```powershell
.\1_generate_discrete_gpu.bat
```

The number of saved sequences can differ from `--num_samples` because sampling is distributed across strategies and iterations, and generated sequences undergo processing. `--top_k` is an upper limit on selected candidates; fewer candidates may satisfy the selection criteria.

When results are available, the output directory contains:

| File | Contents |
| --- | --- |
| `discrete_amps_iteration_*.csv` | Results from individual generation iterations |
| `discrete_amps_all.csv` | Combined generated sequences and calculated properties |
| `discrete_amps_strategies.csv` | Generation strategy summaries |
| `discrete_amps_top.csv` | Selected candidates, when available |
| `discrete_amps_*.png` | Evaluation plots, when successfully generated |

The script also writes `discrete_amp_generate.log` in the working directory.

## Train the models

The commands below reproduce the argument settings in the supplied batch files. They are provided as training entry points; they do not establish exact reproduction of every manuscript result. Training writes checkpoints and histories to the specified directories, so use different `--checkpoint_dir` values if the included checkpoints should be preserved.

### 1. Protein-sequence pretraining

```shell
python pretrain_discrete.py --train_data pretrain_data/data/training_data/train --val_data pretrain_data/data/training_data/val --embedding_dim 128 --hidden_dim 512 --transformer_layers 6 --transformer_heads 8 --num_timesteps 1000 --beta_schedule linear --batch_size 512 --num_epochs 60 --learning_rate 1e-4 --checkpoint_dir checkpoints/discrete_pretrain
```

### 2. DPO fine-tuning

```shell
python train_discrete.py --data_path data/amp_preference_train.csv --val_data_path data/amp_preference_val.csv --pretrained_model checkpoints/discrete_pretrain/discrete_diffusion_pretrained_latest.pt --num_epochs 60 --batch_size 128 --learning_rate 5e-3 --patience 20 --progressive_beta --initial_beta 0.1 --beta 1.0
```

The default DPO checkpoint directory is `checkpoints/discrete_dpo`. The training loader retains preference pairs whose preferred MIC is lower than the non-preferred MIC.

Windows PowerShell launchers:

```powershell
.\1_pretrain.bat
.\1_train_discrete_gpu.bat
```

The pretraining and DPO batch files set `CUDA_VISIBLE_DEVICES=0`. Both training scripts expose `--seed` with a default of `42`. The generation script does not expose a seed argument, so repeated runs can produce different sequences.

For the full argument lists:

```shell
python pretrain_discrete.py --help
python train_discrete.py --help
python generate_discrete.py --help
```

The three main scripts define their own command-line arguments. Refer to those scripts and their launchers for the settings used by each workflow.

## Included data

| Path | Description |
| --- | --- |
| `data/grampa.csv` | Original AMP data table supplied with the project |
| `data/processed_grampa.csv` | Processed AMP data table |
| `data/amp_preference_pairs.csv` | AMP preference-pair table |
| `data/amp_preference_train.csv` | DPO training split |
| `data/amp_preference_val.csv` | DPO validation split |
| `pretrain_data/data/processed_uniprot/protein_sequences.csv` | Processed protein-sequence table |
| `pretrain_data/data/training_data/` | Pretraining train, validation, and test batches, with `index.csv` files |

Preference-pair CSV files contain `seq_preferred`, `seq_non_preferred`, `mic_preferred`, and `mic_non_preferred`, together with paired property columns and a bacterium column. Pretraining batch files contain a `sequence` column.

The included data allow training from the supplied splits without downloading UniProt again. Download and preprocessing scripts are retained in `pretrain_data/`; their configuration paths are relative to the working directory. To invoke the preparation workflow with its existing configuration, run it from that directory:

```shell
cd pretrain_data
python prepare_uniprot.py --config config.json
```

This command downloads and processes data and may replace existing processed files. The large raw UniProt FASTA download, its compressed copy, and intermediate preprocessing chunks are excluded. A new database download may differ from the historical source data.

## Included checkpoints

| Path under `checkpoints/` | Role |
| --- | --- |
| `discrete_pretrain/discrete_diffusion_pretrained_best.pt` | Saved best pretraining checkpoint |
| `discrete_pretrain/discrete_diffusion_pretrained_latest.pt` | Latest pretraining checkpoint referenced by the DPO launcher |
| `discrete_dpo/discrete_amp_dpo_best.pt` | Saved best DPO checkpoint referenced by the generation launcher |

Training-history JSON files and curve images accompany the checkpoints. Intermediate epoch/step checkpoints and historical experiment backups are omitted.

## Evaluation and reproducibility notes

Sequence properties and ranking scores are implemented in `utils/protein_utils.py` and `utils/evaluation.py`. The AMP and toxicity scores are sequence-property-based heuristic scores; they should be interpreted according to those implementations, rather than as experimental MIC measurements or toxicity measurements. Candidate selection also applies stability, amphipathicity, and molecular-weight criteria.

This package covers the supplied model training, generation, and built-in sequence evaluation code. Separate downstream screening, molecular simulation, and experimental workflows are not included here.

The original source files, data, configuration, and launchers were copied without modification. Package checks verified copied-file hashes and the presence of pretraining batches referenced by the supplied indices. Training and inference were not executed during packaging, and correspondence between the included checkpoints and every manuscript experiment has not been independently verified.
