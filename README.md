# MELP

<b>From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining</b>, ICML 2025.
<br><em>Fuying Wang, Jiacheng Xu, and Lequan Yu</em></br>

[Arxiv](https://arxiv.org/abs/2506.21803) | [Cite](#acknowledgements) | [HuggingFace](https://huggingface.co/fuyingw/MELP_Encoder)

**Abstract**: Electrocardiograms (ECGs) play a vital role in monitoring cardiac health and diagnosing heart diseases. However, traditional deep learning approaches for ECG analysis rely heavily on large-scale manual annotations, which are both time-consuming and resource-intensive to obtain. To overcome this limitation, self-supervised learning (SSL) has emerged as a promising alternative, enabling the extraction of robust ECG representations that can be efficiently transferred to various downstream tasks. While previous studies have explored SSL for ECG pretraining and multi-modal ECG-language alignment, they often fail to capture the multi-scale nature of ECG signals. As a result, these methods struggle to learn generalized representations due to their inability to model the hierarchical structure of ECG data. To address this gap, we introduce MELP, a novel Multi-scale ECG-Language Pretraining (MELP) model that fully leverages hierarchical supervision from ECG-text pairs. MELP first pretrains a cardiology-specific language model to enhance its understanding of clinical text. It then applies three levels of cross-modal supervision—at the token, beat, and rhythm levels—to align ECG signals with textual reports, capturing structured information across different time scales. We evaluate MELP on three public ECG datasets across multiple tasks, including zero-shot ECG classification, linear probing, and transfer learning. Experimental results demonstrate that MELP outperforms existing SSL methods, underscoring its effectiveness and adaptability across diverse clinical applications.

![](docs/framework.png)

## Updates
- 29/05/2025: The first version of MELP code base is now alive.

## Installation 

```
conda create -n melp python=3.10
conda activate melp
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

Every filesystem root resolves from the environment (see `src/melp/paths.py`; run
`python src/melp/paths.py` to print what currently resolves and whether it exists).

```
MELP_RAW_DATA_PATH            # raw datasets, one directory per dataset
|- mimic-iv-ecg
|- ptbxl
|- icbeb
|- chapman

MELP_PROCESSED_DATA_PATH      # output of scripts/preprocess/*; defaults next to the raw root
MELP_SPLIT_DIR                # data splits; defaults to the ones shipped under src/melp/data_split
MELP_ECGFM_PATH               # ECG-FM checkpoint used to initialise the ECG encoder (optional)
```

The MIMIC-IV-ECG splits are not shipped in this repo — download them (link below) and either
drop them at `src/melp/data_split/mimic-iv-ecg/` or point `MELP_SPLIT_DIR` at where they live.
ECG: 
- [MIMIC-IV](https://physionet.org/content/mimic-iv-ecg/1.0/)
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- [Code-15%](https://zenodo.org/records/4916206)
- [CPSC 2018](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/)
- [CSN](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- [G12E](https://physionet.org/content/challenge-2020/1.0.2/training/georgia/)

**Please download the splits of MIMIC-IV-ECG from [this link](https://drive.google.com/drive/folders/1RsLRFGoDwakC8smhUmVoBSkjq46To7vI?usp=sharing)**

## Walkthrough of MELP

## Pretraining Stage

```
cd scripts/pretrain
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain.py --num_devices 4 --train_data_pct 1 \
    --text_encoder_name ncbi/MedCPT-Query-Encoder \
    --lr 2e-4 --model_name melp --batch_size 64 --max_epochs 100 \
    --ecg_encoder_name ecgfm \
    --clip_loss_weight 1.0 --caption_loss_weight 2.0 --local_loss_weight 0.2
```

`--ecg_source` selects where the pretraining waveforms are read from: `raw` (default) reads
the wfdb records under `MELP_RAW_DATA_PATH/mimic-iv-ecg/`, `processed` reads the denoised
`.npy` store written by `scripts/preprocess/preprocess_mimic_iv_ecg.py` under
`MELP_PROCESSED_DATA_PATH/mimic-iv-ecg/records/`. The two are not interchangeable -- pick one
and keep it fixed across a set of runs.

## Smoke test

```
python scripts/smoke_test.py      # datasets, model forward/loss/backward, zero-shot, linear probe, pretraining
```

## Evaluation 

### Linear Probing

```
cd scripts/finetune
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
    --model_name melp --dataset_name icbeb \
    --train_data_pct 0.01 \
    --ckpt_path CKPT_PATH \
    --num_devices 1
```

### Zero-shot Classification
```
cd scripts/zeroshot
python test_zeroshot.py
```

## Acknowledgements
If you find our work useful in your research or if you use parts of our code, please cite our paper:
```
@article{wang2025token,
  title={From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining},
  author={Wang, Fuying and Xu, Jiacheng and Yu, Lequan},
  journal={arXiv preprint arXiv:2506.21803},
  year={2025}
}
```
