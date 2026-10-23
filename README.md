# PPG-Bench - A Unified Evaluation Framework for PPG Foundation Models

Official code for PPG-Bench. Benchmark your PPG encoder on nine datasets with linear probes, random forests,
frozen MLP probes and end-to-end finetuning.

[Paper](https://arxiv.org/abs/2610.13435)
## Install

Use Python 3.11 or newer. Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch inside this environment for your CPU/CUDA setup using the
[PyTorch installer](https://pytorch.org/get-started/locally/), then:

```bash
pip install ppg-bench
ppg-bench list
ppg-bench init
```

`ppg-bench init` creates `configs/` and `examples/` in your working directory.
It preserves existing directories; use `--output another-directory` if needed.

For the random-forest backend, a supported NVIDIA GPU and RAPIDS cuML
are required:

```bash
pip install --extra-index-url https://pypi.nvidia.com 'ppg-bench[rf]'
```

Check [RAPIDS installation requirements](https://docs.rapids.ai/install/) for
your CUDA environment. Linear probing, MLP probing and finetuning can also run
on CPU. Model-specific dependencies belong to your encoder's environment.

## 1. Download datasets

```bash
ppg-bench download ppgbp bidmc
ppg-bench download all --physionet-user YOUR_USERNAME
```

Downloads go to `data/raw/<dataset>`. Interrupted downloads can be resumed by rerunning the command.

| Name | Source/version | Raw layout recognized by preprocessing |
| --- | --- | --- |
| `dalia` | [UCI PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | `S1/S1.pkl`, … |
| `dreamt` | [PhysioNet 2.1.0](https://physionet.org/content/dreamt/2.1.0/) | `data_64Hz/S*_whole_df.csv` |
| `wesad` | [WESAD authors](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) | `S2/S2.pkl`, … |
| `ecsmp` | [Mendeley v2](https://data.mendeley.com/datasets/vn5nknh3mn/2) | `E4/`, `EEG_downsample/`, `scale.xlsx` |
| `ppgbp` | [Figshare v5](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299/5) | `PPG-BP dataset.xlsx`, signal text files |
| `bidmc` | [PhysioNet 1.0.0](https://physionet.org/content/bidmc/1.0.0/) | `bidmc_csv/` |
| `butppg` | [PhysioNet 2.0.0](https://physionet.org/content/butppg/2.0.0/) | `subject-info.csv`, `quality-hr-ann.csv`, record directories |
| `realworldppg` | [Mendeley v1](https://data.mendeley.com/datasets/yynb8t9x3d/1) | `train8.xlsx`, `test8.xlsx` |
| `wildppg` | [WildPPG authors](https://siplab.org/projects/WildPPG) | `WildPPG_Part_*.mat` |

DREAMT requires an approved PhysioNet account and signed data-use agreement.

## 2. Quickstart: your own model

After downloading a dataset, preprocess it at your encoder's sampling rate:

```bash
ppg-bench preprocess ppgbp --fs 125
```

Use `preprocess all --fs 125` to prepare the full benchmark. For existing raw
files, add `--raw-root /path/to/dataset` when preprocessing one dataset.

Edit `examples/model.py`, created by `ppg-bench init`, to implement two methods:

```python
import torch
from ppg_bench.models import ModelPort
from your_model import Encoder  # Your model's architecture.


class MyModel(ModelPort):
    def load_encoder(self):
        encoder = Encoder()
        weights = torch.load(
            self.model_init_cfg["ckpt_path"], map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(weights)
        return encoder

    def prepare_batch(self, batch, model_cfg, device):
        x = batch["x"].float().to(device)  # [batch, samples]
        # Apply your model's filtering and normalization here.
        return x.unsqueeze(1)  # Example: [batch, 1, samples]
```

`load_encoder()` must return a fresh encoder with pretrained weights whose
forward pass produces `[batch, embedding_dim]`. Adapt checkpoint loading to your
model's format. `prepare_batch()` applies the input formatting, filtering and
normalization used during pretraining; the benchmark supplies resampled PPG
windows. For finetuning, keep gradients enabled through the encoder.

Set `examples/model.yaml` to match your model:

```yaml
name: MyModel
port: model.py:MyModel
fs: 125
embedding_dim: 512
init:
  ckpt_path: ../checkpoints/my_encoder.pt
```

Paths are relative to the YAML file. Set `fs` and `embedding_dim` to your encoder's
sampling rate and output size. The benchmark handles downstream heads, training,
hyperparameter selection and scoring.

Configs for the paper's external models are also created in `configs/`. Download their
weights separately; [AnyPPG](https://github.com/Ngk03/AnyPPG) and
[SigmaPPG](https://github.com/ZonghengGuo/SigmaPPG) also require the authors' source
code, with `init.source_root` pointing to the downloaded repository.

## 3. Run a task, a dataset, or everything

Start with a single fold and seed to check your adapter:

```bash
ppg-bench run --model examples/model.yaml --datasets ppgbp-sysbp \
  --methods linear --seeds 42 --folds 0 --device cpu --output results/check
```

Evaluate every task on one dataset with the full seeds and folds:

```bash
ppg-bench run --model examples/model.yaml --datasets dalia \
  --methods linear --output results/dalia
```

Run the full benchmark:

```bash
ppg-bench run --model examples/model.yaml --output results/my-model
```

The full suite evaluates 17 primary tasks with all four methods and nine
secondary tasks with linear and RF probes: **2,150 outer evaluations per model**.
Models and folds run sequentially to keep execution simple (it is recommended to add parallelization if possible on your local machine). This command is interrupt-resume-friendly. **Remember to change the output dir between runs to avoid overwriting results**. Results run with this command will be directly comparable to those in the paper, so you do not need to rerun any external models.

## Protocol and results

The protocol in `ppg_bench.protocol` specifies:

- Five outer folds and four inner folds for exhaustive linear/RF grid search.
- A 20% group-disjoint validation holdout for neural methods, repeated for
  subject-level tasks, with repeat-specific seeds.
- 48 TPE trials, eight startup trials and Hyperband pruning: 25 maximum epochs
  for MLP probes and 15 for finetuning.
- Mean subject aggregation for regression; mean, median and top-10% aggregation
  candidates for subject-level classification.

Groups are subjects, except closed-set RealWorldPPG identification, where groups
are individual records and participant IDs remain shared class labels. PPG-BP
targets are scored at subject level. Selection uses MAE for regression, AUROC
for binary linear/RF/MLP tasks, and balanced accuracy for multiclass tasks and
finetuning classification. Secondary ECSMP emotion estimates six questionnaire
scores and reports balanced accuracy of their argmax.

Each run saves:

```text
results/my-model/
  run.json          # models, data identity, protocol, planned jobs
  summary.csv       # task/method means and Student-t 95% CIs
  status.json       # completed/planned counts
  jobs/             # per-fold metrics, splits, predictions and HPO records
  features/         # reusable frozen embeddings
```

## License

PPG-Bench's own code is licensed under the GNU General Public License v3.0
only (`GPL-3.0-only`); see `LICENSE`. Bundled PaPaGei and PulsePPG code retains
its BSD-3-Clause and MIT licenses, respectively, with the full notices in
`ppg_bench/encoders/NOTICE`. External model code, weights and datasets remain
subject to their authors' terms.
