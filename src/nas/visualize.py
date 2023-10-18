'''
Code copied and modified from https://nni.readthedocs.io/en/stable/tutorials/darts.html,
which was initially copied and modified from https://github.com/quark0/darts/blob/master/cnn/visualize.py
This version fixes HUGE bugs in all previous implementations...
'''
import argparse
import json
import os

import graphviz


def plot_single_cell(arch_dict, cell_name, save_path, debug=False):
    g = graphviz.Digraph(
        node_attr=dict(style='filled', shape='rect', align='center'),
        format='pdf',
        engine='dot'
    )
    g.body.extend(['rankdir=TB'])

    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')
    assert len(arch_dict) % 2 == 0

    for i in range(2, 6):
        g.node(str(i), fillcolor='lightblue')

    for i in range(2, 6):
        for j in range(2):
            op = arch_dict[f'{cell_name}/op_{i}_{j}']
            # cast the list to the index of node
            from_ = arch_dict[f'{cell_name}/input_{i}_{j}'][0]
            if from_ == 0:
                u = 'c_{k-2}'
            elif from_ == 1:
                u = 'c_{k-1}'
            else:
                u = str(from_)
            v = str(i)
            g.edge(u, v, label=op, fillcolor='gray')

    g.node('c_{k}', fillcolor='palegoldenrod')
    for i in range(2, 6):
        g.edge(str(i), 'c_{k}', fillcolor='gray')

    g.attr(label=f'{cell_name.capitalize()} cell')

    g.render(save_path, cleanup=not debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_path', type=str, default='./exported_arch.json')
    parser.add_argument('--save_path', type=str, default='./logs/')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    with open(args.arch_path, 'r') as f:
        exported_arch = json.load(f)

    plot_single_cell(exported_arch, 'normal', os.path.join(args.save_path, 'normal_cell'), args.debug)
    plot_single_cell(exported_arch, 'reduce', os.path.join(args.save_path, 'reduce_cell'), args.debug)
