# vehiclex-classification

## Contents
- [vehiclex-classification](#vehiclex-classification)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Requirements and Dependencies](#requirements-and-dependencies)
  - [Data Preparation](#data-preparation)
  - [Code Execution](#code-execution)
    - [Visualization Reproduction](#visualization-reproduction)
    - [Experiment Results Reproduction](#experiment-results-reproduction)

## Overview
We use the DARTS [(Liu *et al.*, 2019)](https://arxiv.org/abs/1806.09055) neural architecture search strategy to search for an effective and efficient architecture for the fine-grained vehicle classification task on the [VehicleX](https://github.com/yorkeyao/VehicleX) dataset [(Yao *et al.*, 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_46).

## Requirements and Dependencies
To run the code, the machine **MUST** have at least four GPUs with 11 GiB video memory and CUDA support. The code has been tested on Ubuntu 18.04.3 with Python 3.8.18 and CUDA 11.6.

To install the required Python libraries, execute the code below.
```
pip3 install -r requirements.txt
```

## Data Preparation
We are using the VehicleX-v2 dataset. The dataset should be downloaded to the `~/Data/` folder. The `~/Data/` folder should have the following structure.
```
~/Data/
├── vehicle-x_v2/
│   ├─ Classification Task/
│   │  ├─ train/
│   │  ├─ val/
│   │  └─ test/
│   ├─ finegrained_label.xml
│   ├─ list_color.txt
│   ├─ list_type.txt
│   ├─ ReadMe ass 2.txt
│   ├─ Simulating Content Consistent Vehicle Datasets with Attribute Descent.pdf
│   └─ reference paper.txt
└─ ...
```

## Code Execution
### Visualization Reproduction
Run the script below to generate the label distribution plots and related statistics. The plots will be saved in the `./logs/` folder.
```
python3 label_dist_analysis.py
```

To visualize the normal and reduce cells of the automatically searched architecture, run the following command to generate the plots for each variant. Two files, `normal.pdf` and `reduce.pdf`, will be generated in `./logs/darts_a/` and `./logs/darts_b/` directories.
```
python3 ./src/nas/visualize.py --arch_path ./cfg/exported_arch_darts_a.json --save_path ./logs/darts_a/ && python3 ./src/nas/visualize.py --arch_path ./cfg/exported_arch_darts_b.json --save_path ./logs/darts_b/
```

### Experiment Results Reproduction
To run the baseline experiments and parse the evaluation results, run the command below. The log for each experiment will be stored in the `./logs/baseline/` folder, and the parsed results will be in the `./out/baseline/` directory.
```
python3 run.py --cfg_path ./cfg/benchmark.json && python3 results_parser.py --logdir ./logs/baseline/ --outdir ./out/baseline/
```

Run the following command to run the DARTS searching experiment twice. Results will be saved in `./logs/darts_a/` and `./logs/darts_b/`, respectively. **(Optional)**
```
bash nas_search.sh
```

Run the command below to retrain the automatically searched architecture and conduct ablation studies on various hyperparameter setups. The log for each experiment will be stored in the `./logs/darts_a/` and `./logs/darts_b/` folders, and the parsed results will be in the `./out/darts_a/` and `./out/darts_b/` directories.
```
python3 run.py --cfg_path ./cfg/darts_retrain.json && python3 results_parser.py --logdir ./logs/darts_a/ --outdir ./out/darts_a/ && python3 results_parser.py --logdir ./logs/darts_b/ --outdir ./out/darts_b/
```
