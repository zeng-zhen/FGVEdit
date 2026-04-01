<a name="readme-top"></a>

<br />
<div align="center">
  <h1 align="center">FGVEdit</h1>
  <p align="center">
    Official code and data release for
    <br />
    <strong>Visual-Oriented Fine-Grained Knowledge Editing for Multimodal Large Language Models</strong>
  </p>
  <p align="center">
    <a href="https://huggingface.co/datasets/ZhenZeng/FGVEdit">Dataset</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ul>
    <li><a href="#-about-this-project">🛠️ About This Project</a></li>
    <li><a href="#-getting-started">🚀 Getting Started</a>
      <ul>
        <li><a href="#download-data">Download Data</a></li>
        <li><a href="#environment-setup">Environment Setup</a></li>
        <li><a href="#download-pre-trained-models">Download Pre-trained Models</a></li>
      </ul>
    </li>
    <li><a href="#-usage">🧪 Usage</a></li>
    <li><a href="#-acknowledgments">🎉 Acknowledgments</a></li>
  </ul>
</details>


## 🛠️ About This Project

FGVEdit is a benchmark and codebase for fine-grained multimodal knowledge editing. The current release contains data processing and experiment entrypoints for four editing methods:

- IKE
- MEND
- SERAC
- MSCKE

The released code currently supports two multimodal backbones:

- BLIP-2
- MiniGPT-4

The default data split used by this repository is:

- `train_data.json`: 8334 samples
- `test_data.json`: 2778 samples

Each sample contains the edit target, rephrase query, textual locality query, fine-grained generality query, fine-grained locality query, and the associated image path.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🚀 Getting Started

### Download Data

The released dataset is available at [ZhenZeng/FGVEdit](https://huggingface.co/datasets/ZhenZeng/FGVEdit).

You can download it with Hugging Face CLI:

```bash
huggingface-cli download ZhenZeng/FGVEdit --repo-type dataset --local-dir ./tmp/FGVEdit_download
```

After download, arrange the files into the layout expected by the current code:

```bash
data/
├── FGVEdit/
│   ├── train_data.json
│   └── test_data.json
└── images/
    ├── train2014/
    └── val2014/
```

The dataset loader reads:

- `data/FGVEdit/train_data.json`
- `data/FGVEdit/test_data.json`
- image paths under `data/images/`

The `image` field inside each JSON record is a relative path such as `train2014/xxx.jpg` or `val2014/xxx.jpg`, so the image root must be `data/images`.

### Environment Setup

This repository currently provides a single Python dependency file: `requirements.txt`.

We recommend using `conda`:

```bash
conda create -n fgvedit python=3.10 -y
conda activate fgvedit
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Download Pre-trained Models

The current `hparams/` files reference pre-trained model caches and checkpoints with the following layout:

```bash
hugging_cache/
├── all-MiniLM-L6-v2/
├── bert-base-uncased/
├── clip-vit-large-patch14/
├── distilbert-base-cased/
├── opt-2.7b/
├── opt-125m/
├── Vicuna/
│
├── blip2_pretrained_flant5xxl.pth
├── blip2_pretrained_opt2.7b.pth
├── eva_vit_g.pth
└── pretrained_minigpt4_7b.pth
```

Links are in the following:
<table>
    <tr>
        <td><a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2</a></td>
        <td><a href="https://huggingface.co/google-bert/bert-base-uncased">bert-base-uncased</a></td>
        <td><a href="https://huggingface.co/distilbert/distilbert-base-cased">distilbert-base-cased</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/facebook/opt-2.7b">opt-2.7b</a></td>
        <td><a href="https://huggingface.co/facebook/opt-125m">opt-125m</a></td>
        <td><a href="https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main">vicuna-7b</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/blip2_pretrained_flant5xxl.pth">blip2_pretrained_flant5xxl.pth</a></td>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt2.7b.pth">blip2_pretrained_opt2.7b.pth</a></td>
        <td><a href="https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/prerained_minigpt4_7b.pth">prerained_minigpt4_7b.pth</a></td>
    </tr>
    <tr>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth">eva_vit_g.pth</a></td>
        <td><a href="https://huggingface.co/openai/clip-vit-large-patch14">clip-vit-large-patch14</a></td>
        <td></td>
    </tr>
</table>



<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🧪 Usage

All experiment entrypoints are at the repository root:

- `edit_IKE.py`
- `edit_MEND.py`
- `edit_SERAC.py`
- `edit_MSCKE.py`

All hyper-parameters are stored in `hparams/`.

### IKE

Configs:

- `hparams/IKE/blip2.yaml`
- `hparams/IKE/minigpt4.yaml`

Run embedding generation only:

```bash
python edit_IKE.py --model blip2 --mode embed
python edit_IKE.py --model minigpt4 --mode embed
```

Run evaluation:

```bash
python edit_IKE.py --model blip2 --mode eval
python edit_IKE.py --model minigpt4 --mode eval
```

The script will build train-set embeddings and then evaluate on `data/FGVEdit/test_data.json`.

### MEND

Training configs:

- `hparams/TRAINING/MEND/blip2.yaml`
- `hparams/TRAINING/MEND/minigpt4.yaml`

Evaluation configs:

- `hparams/MEND/blip2.yaml`
- `hparams/MEND/minigpt4.yaml`

Run training:

```bash
python edit_MEND.py --model blip2 --mode train
python edit_MEND.py --model minigpt4 --mode train
```

Run evaluation:

```bash
python edit_MEND.py --model blip2 --mode eval
python edit_MEND.py --model minigpt4 --mode eval
```

Important:

- `--mode eval` requires a trained checkpoint.
- Before evaluation, update the `archive` field in the chosen evaluation YAML so that it points to a concrete `.pt` checkpoint file produced during training.
- Training outputs are saved under `results_dir/models/<ALG>/`.

### SERAC

Training configs:

- `hparams/TRAINING/SERAC/blip2.yaml`
- `hparams/TRAINING/SERAC/minigpt4.yaml`

Evaluation configs:

- `hparams/SERAC/blip2.yaml`
- `hparams/SERAC/minigpt4.yaml`

Run training:

```bash
python edit_SERAC.py --model blip2 --mode train
python edit_SERAC.py --model minigpt4 --mode train
```

Run evaluation:

```bash
python edit_SERAC.py --model blip2 --mode eval
python edit_SERAC.py --model minigpt4 --mode eval
```

Important:

- `--mode eval` requires a trained checkpoint.
- Before evaluation, update the `archive` field in the chosen evaluation YAML to the actual checkpoint file path.

### MSCKE

Training configs:

- `hparams/TRAINING/MSCKE/blip2.yaml`
- `hparams/TRAINING/MSCKE/minigpt4.yaml`

Evaluation configs:

- `hparams/MSCKE/blip2.yaml`
- `hparams/MSCKE/minigpt4.yaml`

Run training:

```bash
python edit_MSCKE.py --model blip2 --mode train
python edit_MSCKE.py --model minigpt4 --mode train
```

Run evaluation:

```bash
python edit_MSCKE.py --model blip2 --mode eval
python edit_MSCKE.py --model minigpt4 --mode eval
```

Important:

- `--mode eval` requires a trained checkpoint.
- Before evaluation, update the `archive` field in the chosen evaluation YAML to the actual checkpoint file path produced during training.

### Practical Notes

- The current scripts read `device` from the selected YAML file, so change that field before running on your machine.
- If you move data or checkpoints, update the corresponding paths in the YAML file instead of relying on implicit defaults.
- MEND, SERAC, and MSCKE evaluation configs currently contain placeholder `archive` paths. They must be replaced with real checkpoint filenames.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 🎉 Acknowledgments

This repository builds on the multimodal editing ecosystem around [EasyEdit](https://github.com/zjunlp/EasyEdit), and uses pretrained components or model implementations from [LAVIS / BLIP-2](https://github.com/salesforce/LAVIS), [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [Transformers](https://github.com/huggingface/transformers), [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers), and [CLIP](https://github.com/openai/CLIP).

We thank the authors and maintainers of these projects for making their code and models publicly available.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
