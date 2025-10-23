# HuBERT-ECG as a Self-Supervised Foundation Model for Broad and Scalable Cardiac Application

[![arXiv](https://img.shields.io/badge/medRxiv-2024.11.14.24317328-B31B1B?logo=arxiv)](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v2)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface)](https://huggingface.co/Edoardo-BS)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)

A self-supervised foundation model for electrocardiogram (ECG) analysis, pre-trained on 9.1 million 12-lead ECGs covering 164 cardiovascular conditions.

<br>

## 🔍 Overview

HuBERT-ECG addresses the challenge of limited ECG datasets by leveraging self-supervised learning on a large-scale corpus. The model achieves:

- **AUROCs 0.843-0.99** across diverse downstream tasks
- **140+ diseases** detected with AUROC > 0.9
- **Mortality prediction** with AUROC up to 0.91 (2-year follow-up)

### Architecture

Built on the HuBERT (Hidden-Unit BERT) framework, adapted for multi-lead ECG signals with:
- 12-lead ECG input processing
- Self-supervised pre-training objectives
- Task-agnostic feature extraction
- Flexible fine-tuning heads for downstream tasks

<br>

## 🛠️ Installation

### Conda Environment

```bash
# Create environment
conda create -n hubert-ecg python=3.11
conda activate hubert-ecg

# Clone repository
git clone https://github.com/Setsu-Kuhaku/HuBERT-ECG-CGMH.git
cd HuBERT-ECG-CGMH
```

**Linux/Mac:**
```bash
chmod +x ./scripts/install_dependencies.sh
./scripts/install_dependencies.sh
```

**Windows:**
```bash
./scripts/install_dependencies.bat
```

<br>

## Reproducibility
In the `reproducibility` folder you can find all train, validation, and test splits we used in our work as .csv files. You simply have to follow the instructions in the `reproducibility/README.md` to reproduce our results.
As an example, you can easily fine-tune and evaluate an instance of HuBERT-ECG on PTB-XL All dataset, as shown in .sh scripts `finetune.sh` and `test.sh`.
Thw forward pass on a single instance takes less than 1 second on an A100 GPU node, which is also the machine we ran our experiments and evaluations on.
The splits were used in cross-validation experiments/evaluations to also mitigate the performance difference that can be be observed when using different hardware and machiens.

**Remember to pre-process your data before feeding HuBERT-ECG. Take a look at Data and Preprocessing section in the paper**

<br>

## News
- [06/2025] A new medrxiv version has been updated with new results, findings and insights!
- [12/2024] Reproducibility has never been easier! Training, validation, and test splits ready to use in the reproducibility folder!
- [12/2024] Pre-trained models are easily downloadable from Hugging Face using `AutoModel.from_pretrained`
- [11/2024] Pre-trained models are freely available on HuggingFace
- [11/2024] This repository has been made public!

<br>

## 📚 Citation
If you use our models or find our work useful, please consider citing us:
```bibtex
@article {Coppola2024.11.14.24317328,
	author = {Coppola, Edoardo and Savardi, Mattia and Massussi, Mauro and Adamo, Marianna and Metra, Marco and Signoroni, Alberto},
	title = {HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications},
	elocation-id = {2024.11.14.24317328},
	year = {2025},
	doi = {10.1101/2024.11.14.24317328},
	publisher = {Cold Spring Harbor Laboratory Press},
	journal = {medRxiv}
}

```


