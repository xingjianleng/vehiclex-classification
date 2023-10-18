'''
This script is used to run multiple **training** experiments in parallel, not for NAS search experiments.
'''
import argparse
import json
import os
import queue
import subprocess
import time


def main(args):
    with open(args.cfg_path, 'r') as fp:
        configs = json.load(fp)

    # create a list of processes
    processes = []

    # set random seeds
    seeds = [int(seed) for seed in args.random_seeds.split(',')]
    assert len(seeds) > 0, 'the script needs to use random seeds'

    # NOTE: gpu queue, change according to your setup
    gpu_ids = [0, 1, 2, 3]
    worker_queue = queue.Queue()
    for id in gpu_ids:
        worker_queue.put(id)

    # queue to hold scripts
    script_arg_queue = queue.Queue()

    for config in configs:
        arguments = ['main.py', *config]
        for seed in seeds:
            script_arg_queue.put(arguments + ['--seed', str(seed)])

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

        # Start a new process
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
    parser.add_argument('--random_seeds', type=str, default='0,13,21,42,389', help='Random seeds to use, separated by commas')
    args = parser.parse_args()

    main(args)
