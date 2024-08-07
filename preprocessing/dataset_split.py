import argparse
import os
import numpy as np
import dgl
import torch
import time
from tqdm import tqdm

import data_utils


def save_sequence(args, data):
    pass


def graph2seq(args, graph_data):
    group_loader = data_utils.GroupFeatureSequenceLoader(graph_data)
    g = graph_data.graph
    labels = graph_data.labels
    train_nid = graph_data.train_nid
    val_nid = graph_data.val_nid
    test_nid = graph_data.test_nid
    n_classes = graph_data.n_classes

    feat_dim = graph_data.feat_dim
    n_relations = graph_data.n_relations
    n_groups = n_classes + 1
    n_hops = len(args['fanouts'])
    n_nodes = g.num_nodes()

    seq_len = n_relations * (n_hops * n_groups + 1)

    all_nid = g.nodes()
    file_dir = os.path.join(args['save_dir'], args['dataset'])
    os.makedirs(file_dir, exist_ok=True)

    infos = np.array([feat_dim, n_classes, n_relations], dtype=np.int64)
    
    info_name = f"{args['dataset']}_infos_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}.npz"
    info_file = os.path.join(file_dir, info_name)
    print(f"Saving infos to {info_file}")
    np.savez(info_file, label=labels.numpy(), train_nid=train_nid, val_nid=val_nid, 
             test_nid=test_nid, infos=infos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graph2seq')
    parser.add_argument('--dataset', type=str, default='amazon',
                        help='Dataset name, [amazon, yelp]')

    parser.add_argument('--train_size', type=float, default=0.4,
                        help='Train size of nodes.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Val size of nodes.')
    parser.add_argument('--seed', type=int, default=717,
                        help='Collecting neighbots in n hops.')

    parser.add_argument('--norm_feat', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--grp_norm', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--force_reload', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--add_self_loop', action='store_true', default=False,
                    help='add self-loop to all the nodes')

#     parser.add_argument('--n_hops', type=int, default=1,
#                         help='Collecting neighbots in n hops.')
    parser.add_argument('--fanouts', type=int, default=[-1], nargs='+',
                        help='Sampling neighbors, default [-1] means full neighbors')

    parser.add_argument('--base_dir', type=str, default='/your_path',
                        help='Directory for loading graph data.')
    parser.add_argument('--save_dir', type=str, default='seq_data',
                        help='Directory for saving the processed sequence data.')
    args = vars(parser.parse_args())

    print(args)
    tic = time.time()
    data = data_utils.prepare_data(args)
    graph2seq(args, data)
    toc = time.time()
    print(f"Elapsed time={toc -tic:.2f}(s)")
