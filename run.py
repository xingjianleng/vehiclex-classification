import argparse
from itertools import product
import json
import os
import queue
import subprocess
import time


def main(args):
    with open(args.cfg_path, 'r') as fp:
        cfg = json.load(fp)
        constr_casc_mode = bool(cfg['constr_casc_mode'])
        hyperparameters = cfg['hyperparameters']

    # create a list of processes
    processes = []

    # worker queue, each gpu runs 3 processes
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')] * 3
    assert len(gpu_ids) > 0, 'the script needs to use gpu'
    worker_queue = queue.Queue()
    for id in gpu_ids:
        worker_queue.put(id)

    # queue to hold scripts
    script_arg_queue = queue.Queue()

    for hyperparameter in hyperparameters:
        if constr_casc_mode:
            arguments = ['main.py', '--constr_casc']
        else:
            arguments = ['main.py']
        arguments.extend(hyperparameter)
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
        env['CUDA_VISIBLE_DEVICES'] = str(worker_id)

        process = subprocess.Popen(['python'] + script_arg, env=env)
        process.worker_id = worker_id
        processes.append(process)
        print(f'[{total - script_arg_queue.qsize()} / {total}]')

    # Wait for all processes to finish
    for process in processes:
        process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('main.py runner')
    parser.add_argument('--cfg_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs to use, separated by commas')
    args = parser.parse_args()

    main(args)
