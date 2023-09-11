# vehicle-x-classification
Implementing Constructive Cascade Network technique [(Khoo *et al*., 2009)](https://link.springer.com/chapter/10.1007/978-3-642-03040-6_29) for vehicle classification on [Vehicle-X](https://github.com/yorkeyao/VehicleX) dataset [(Yao *et al.*, 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_46).

## Prerequisites
To run the code, the machine **MUST** have at least one GPU with CUDA support. The code has been tested on Ubuntu 18.04.3 with Python 3.8.17 and CUDA 11.6.

To install the required Python libraries, execute the code below.
```bash
pip3 install -r requirements.txt
```

The dataset should be downloaded to the `~/Data/` folder. The `~/Data/` folder should have the following structure.
```
~/Data/
├── vehicle-x/
│   ├─ train
│   ├─ val
│   ├─ test
│   ├─ finegrained_label.xml
│   ├─ list_color.txt
│   ├─ list_type.txt
│   ├─ ReadMe.txt
│   └─ reference paper.txt
└─ ...
```

## Code Execution
Run the script below to generate the label distribution plots and related statistics. The plots will be saved in the `./logs` folder.
```bash
python3 label_dist_analysis.py
```

Run the script below to execute the code. The results will be saved in the `./logs` folder. The `--cfg_path` argument specifies the path to the configuration file.
```bash
python3 run.py --cfg_path ./cfg/benchmark.json && python3 run.py --cfg_path ./cfg/exploratory.json
```
Run the script below to parse the results. The results will be saved in the `./out` folder. The `--partition` argument specifies the partition of the logs to be parsed.
```bash
python3 results_parser.py --partition benchmark && python3 results_parser.py --partition exploratory
```

## Results
Benchmark and exploratory *average* results (in %) are shown below. Corresponding configs can be found in the `./out/<benchmark/exploratory>/configs.json` files after running the code.

### Benchmark
| Configs | acc    | prec   | rec    | f1     |
|---------|--------|--------|--------|--------|
|Prototype| 9.3343 | 9.2757 | 9.4832 | 8.9593 |
|$L_2$ - 5e-5 \& $L_3$ - 1e-5| 9.1850 | 9.1585 | 9.3796 | 8.8217 |
|Threshold_decay - 0.9| 9.1415 | 9.0966 | 9.3128 | 8.7762 |
|Threshold - 1.5| 9.1071 | 8.9222 | 9.2493 | 8.6658 |
|Cascade_hidden - 32| 9.0543 | 8.9687 | 9.2578 | 8.6988 |
|Threshold_decay - 0.8| 8.9790 | 8.8445 | 9.1238 | 8.5728 |
|Threshold - 2.3| 8.8997 | 8.7515 | 9.0820 | 8.5028 |
|Baseline| 7.6727 | 7.3501 | 7.8120 | 7.2219 |
|Dropout - 0.0| 5.0509 | 5.2948 | 5.1422 | 4.9430 |

### Exploratory
| Configs | acc     | prec    | rec     | f1      |
|---------|---------|---------|---------|---------|
|Adam_b1024_lr5e-4_focal| 13.7472 | 14.1927 | 14.0398 | 12.8774 |
|Adam_b1024_lr5e-4_ce| 13.4711 | 13.8425 | 13.8380 | 12.4137 |
|Adam_b1024_lr1e-3_ce| 13.2981 | 13.6498 | 13.6013 | 11.9654 |
|Adam_b1024_lr1e-4_focal| 13.2043 | 13.1773 | 13.4505 | 12.1683 |
|Adam_b1024_lr1e-4_ce| 12.8319 | 12.5054 | 13.1305 | 11.5614 |
|Adam_b0_lr5e-4_ce| 12.6034 | 12.3407 | 12.8986 | 11.3786 |
|RPROP_b0_lr5e-4_ce| 7.8391  | 7.5973  | 7.9222  | 7.4113  |
|RPROP_b0_lr1e-2_ce| 7.7929  | 7.5402  | 7.9048  | 7.3551  |
|RPROP_b0_lr1e-2_focal| 7.3042  | 7.2535  | 7.4053  | 6.9875  |
|RPROP_b1024_lr1e-2_ce| 2.1001  | 0.9679  | 2.1682  | 1.0327  |
|SGD_b1024_lr5e-4_ce| 0.1242  | 0.0795  | 0.1258  | 0.0602  |
