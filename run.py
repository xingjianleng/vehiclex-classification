import argparse
from itertools import product
import os
import queue
import subprocess
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser('main.py runner')
    parser.add_argument('--cascade', action='store_true', help='Run cascade')
    parser.add_argument('--hyperparameters', type=str, help='Hyperparameter to vary')
    args = parser.parse_args()

    total_hyperparameter_space = {
        'hidden_dims': [''],
        # 'hidden_dims': ['2048'],
        'lr': [1e-3, 5e-4, 1e-4],
        'loss': ['ce', 'focal'],
        'scheduler': [None, 'lambda'],
        'batch_size': [0, 1024],
        'optim': ['adam', 'sgd', 'rprop']
    }

    if args.hyperparameters:
        hyperparameters = args.hyperparameters.split(',')
    else:
        hyperparameters = list(total_hyperparameter_space.keys())        

    # create a list of processes
    processes = []
    hyperparameter_space = {k: v for k, v in total_hyperparameter_space.items() if k in hyperparameters}

    # worker queue
    worker_queue = queue.Queue()
    for i in range(3):
        worker_queue.put(i)

    # queue to hold scripts
    script_arg_queue = queue.Queue()

    combinations = list(product(*total_hyperparameter_space.values()))
    for combination in combinations:
        if args.cascade:
            raise NotImplementedError('Cascade not implemented')
        else:
            arguments = ['main_baseline.py']
        for key, value in zip(total_hyperparameter_space.keys(), combination):
            if value is not None:
                arguments.extend([f'--{key}', str(value)])
        script_arg_queue.put(arguments)

    total = script_arg_queue.qsize()
    while not script_arg_queue.empty():
        if worker_queue.empty():
            # No GPU is available, wait for a process to finish
            for process in processes:
                if process.poll() is not None:  # A None value indicates that the process is still running
                    processes.remove(process)
                    worker_queue.put(process.worker_id)
                    break
            else:
                # No process has finished, wait a bit before checking again
                time.sleep(1)
                continue

        worker_id = worker_queue.get()
        script_arg = script_arg_queue.get()

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '1'

        process = subprocess.Popen(['python'] + script_arg, env=env)
        process.worker_id = worker_id
        processes.append(process)
        print(f'[{total - script_arg_queue.qsize()} / {total}]')

    # Wait for all processes to finish
    for process in processes:
        process.wait()
