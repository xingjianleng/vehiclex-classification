# COMP4660-assignment
Implementing Constructive Cascade Network technique [(Khoo *et al*., 2009)](https://link.springer.com/chapter/10.1007/978-3-642-03040-6_29) for vehicle classification on [Vehicle-X](https://github.com/yorkeyao/VehicleX) dataset [(Yao *et al.*, 2020)](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_46).

## Prerequisites
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

Run the script below to execute the code. The results will be saved in the `./logs` folder. The `--cfg_path` argument specifies the path to the configuration file. The `--gpu_ids` argument specifies the GPU IDs to be used. The `--procs_per_gpu` argument specifies the number of processes to be run on each GPU.
```bash
python3 run.py --cfg_path ./cfg/benchmark.json --gpu_ids <gpu_ids> && python3 run.py --cfg_path ./cfg/exploratory.json --gpu_ids <gpu_ids> --procs_per_gpu 2
```
Run the script below to parse the results. The results will be saved in the `./out` folder. The `--partition` argument specifies the partition of the logs to be parsed.
```bash
python3 results_parser.py --partition benchmark && python3 results_parser.py --partition exploratory
```

## Results
Benchmark and exploratory results (in %) are shown below. Corresponding configs can be found in the `./out/<benchmark/exploratory>/configs.json` files after running the code.

### Benchmark
| Configs | acc    | prec   | rec    | f1     |
|---------|--------|--------|--------|--------|
| 0       | 9.3343 | 9.2757 | 9.4832 | 8.9593 |
| 1       | 9.1850 | 9.1585 | 9.3796 | 8.8217 |
| 2       | 9.1415 | 9.0966 | 9.3128 | 8.7762 |
| 3       | 9.1071 | 8.9222 | 9.2493 | 8.6658 |
| 4       | 9.0543 | 8.9687 | 9.2578 | 8.6988 |
| 5       | 8.9790 | 8.8445 | 9.1238 | 8.5728 |
| 6       | 8.8997 | 8.7515 | 9.0820 | 8.5028 |
| 7       | 7.6727 | 7.3501 | 7.8120 | 7.2219 |
| 8       | 5.0509 | 5.2948 | 5.1422 | 4.9430 |

### Exploratory
| Configs | acc     | prec    | rec     | f1      |
|---------|---------|---------|---------|---------|
| 0       | 13.7472 | 14.1927 | 14.0398 | 12.8774 |
| 1       | 13.4711 | 13.8425 | 13.8380 | 12.4137 |
| 2       | 13.2981 | 13.6498 | 13.6013 | 11.9654 |
| 3       | 13.2043 | 13.1773 | 13.4505 | 12.1683 |
| 4       | 12.8319 | 12.5054 | 13.1305 | 11.5614 |
| 5       | 12.6034 | 12.3407 | 12.8986 | 11.3786 |
| 6       | 7.8391  | 7.5973  | 7.9222  | 7.4113  |
| 7       | 7.7929  | 7.5402  | 7.9048  | 7.3551  |
| 8       | 7.3042  | 7.2535  | 7.4053  | 6.9875  |
| 9       | 2.1001  | 0.9679  | 2.1682  | 1.0327  |
| 10      | 0.1242  | 0.0795  | 0.1258  | 0.0602  |
