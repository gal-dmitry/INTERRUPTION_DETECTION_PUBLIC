# INTERRUPTION_DETECTION_PUBLIC



### Description

This is a public repository containing code for paper ["Conversational RuBERT for Detecting Competitive Interruptions in ASR-Transcribed Dialogues"](https://aircconline.com/csit/papers/vol14/csit141306.pdf). This is our second work dedicated to the task of contact center monitoring (see also ["Text-Based Detection of On-Hold Scripts in Contact Center Calls"](https://arxiv.org/abs/2407.09849)).



### Experiments


#### Scripts

- Text + TextPair:
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_01.yml`
- Text only:
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_02.yml`
- Class weights:
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_03.yml`
- Learning Rate:
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_04.yml`
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_05.yml`
        - best: lr=7e-6; roc_auc_binary=0.8870; F1_macro=0.8404
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_06.yml`
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_07.yml`
- Synthetic samples / wider context:
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_08.yml`
        - \+ synthetic samples of class 0
    - `python train.py --config_path configs/train/comp_nocomp/comp_nocomp_09.yml`
        - wider context


#### Notebook

- [EXPERIMENTS_01.ipynb](./EXPERIMENTS_01.ipynb)


#### Metric Table 

| Config | Metadata folder | Notebook section name | Hyperparameter | ROC AUC Binary | Best threshold | Recall macro | Precision Macro | Balanced Accuracy | F1 Macro |
|:-------|:----------------|:----------------------|---------------:|---------------:|---------------:|-------------:|----------------:|-------------------:|---------:|
| comp_nocomp_01.yml | [2024_05_22__09_20_49](./mlruns/comp_nocomp/2024_05_22__09_20_49) | `comp_nocomp_01 / 2024_05_22__09_20_49` | Input: Speaker + Listener | 0.8508 | 0.5303 | 0.7571 | 0.7599 | 0.7571 | 0.7582 |
| comp_nocomp_02.yml | [2024_05_22__12_04_55](./mlruns/comp_nocomp/2024_05_22__12_04_55) | `comp_nocomp_02 / 2024_05_22__12_04_55` | Input: Only Listener | 0.8118 | 0.5249 | 0.7120 | 0.7301 | 0.7120 | 0.7151 |
| comp_nocomp_03.yml | [2024_05_22__10_48_23](./mlruns/comp_nocomp/2024_05_22__10_48_23) | `comp_nocomp_03 / 2024_05_22__10_48_23` | Class weights: `[1.0, 0.75]` | 0.8197 | 0.4579 | 0.7299 | 0.7394 | 0.7299 | 0.7320 |
| comp_nocomp_04.yml | [2024_05_22__12_32_53](./mlruns/comp_nocomp/2024_05_22__12_32_53) | `comp_nocomp_04 / 2024_05_22__12_32_53` | LR = 5.e-6 | 0.8818 | 0.4374 | 0.8221 | 0.8483 | 0.8221 | 0.8286 
| **comp_nocomp_05.yml** | [2024_05_22__13_03_23](./mlruns/comp_nocomp/2024_05_22__13_03_23) | `comp_nocomp_05 / 2024_05_22__13_03_23` | LR = 7.e-6 | 0.8870 | 0.3858 | **0.8325** | **0.8671** | **0.8325** | **0.8404** |
| comp_nocomp_06.yml | [2024_05_22__13_27_47](./mlruns/comp_nocomp/2024_05_22__13_27_47) | `comp_nocomp_06 / 2024_05_22__13_27_47` | LR = 9.e-6 | **0.8891** | 0.3983 | 0.8260 | 0.8517 | 0.8260 | 0.8325 |
| comp_nocomp_07.yml | [2024_05_22__14_11_45](./mlruns/comp_nocomp/2024_05_22__14_11_45) | `comp_nocomp_07 / 2024_05_22__14_11_45` | LR = 1.e-6 | 0.5566 | 0.5035 | 0.5334 | 0.5348 | 0.5334 | 0.5566 |
| comp_nocomp_08.yml | [2024_05_22__14_58_17](./mlruns/comp_nocomp/2024_05_22__14_58_17) | `comp_nocomp_08 / 2024_05_22__14_58_17` | + Synthetic data: class 0 | 0.8874 | 0.4338 | 0.8121 | 0.8239 | 0.8121 | 0.8155 |
| comp_nocomp_09.yml | [2024_05_22__15_43_48](./mlruns/comp_nocomp/2024_05_22__15_43_48) | `comp_nocomp_09 / 2024_05_22__15_43_48` | Context window: 8 phrases | 0.7049 | 0.3949 | 0.6542 | 0.6572 | 0.6542 | 0.6534 |


#### Extended Metric Table

Best model: [2024_05_22__13_03_23 / predictions / test_metrics_f1.csv](./mlruns/comp_nocomp/2024_05_22__13_03_23/predictions/test_metrics_f1.csv)

