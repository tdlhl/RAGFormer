import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import dgl
import time
from tqdm import tqdm

from data import sequence, fraud_dataset
from modules import models_ragformer
from utils import utility, metrics, earlystopping, log_tools, plot_tools
from model import GCN


def evaluation(args, model, gnn_model, eval_loader,val_graph1,val_graph2,val_graph3, threshold_moving=True, thres=0.5, device='cpu'):
    r"""Evaluate model.
    Parameters
    ----------
    args : dict
        Train configurations. (useless)
    model : nn.Module
        Trained model. 
    eval_loader : DataLoader
        Dataloader for evaluation data.
    threshold_moving : bool
        Whether to use threshold-moving strategy. Default=True.
    thres : float
        The selected threshold to convert probabilities to crisp class labels.
    device : str
        Evalutaion on cpu or gpu. Default='cpu'.

    Return : tuple
        Returns (y_true, y_prob, y_pred), namely the ground truth labels, 
        the probabilities and predicted labels.

    """
    model.to(device)
    model.eval()
    gnn_model.to(device)
    gnn_model.eval()
    prob_list = []
    label_list = []
    batch_size = args['batch_size']
    with torch.no_grad():
        val_indices = val_graph1.nodes()
        for step, (batch_seq, batch_labels) in enumerate(tqdm(eval_loader)):
            batch_seq = batch_seq.to(device)
            batch_seq = torch.transpose(batch_seq, 1, 0)

            feat1 = batch_seq[0]
            feat2 = batch_seq[7]
            feat3 = batch_seq[14]
            current_indices = val_indices[(step*batch_size):((step+1)*batch_size)]
            subgraph_1 = dgl.node_subgraph(val_graph1, current_indices)
            subgraph_2 = dgl.node_subgraph(val_graph2, current_indices)
            subgraph_3 = dgl.node_subgraph(val_graph3, current_indices)
            # subgraph_1.to(device)
            # subgraph_2.to(device)
            # subgraph_3.to(device)
            gnn_emb_1 = gnn_model[0](subgraph_1, feat1)
            gnn_emb_2 = gnn_model[1](subgraph_2, feat2)
            gnn_emb_3 = gnn_model[2](subgraph_3, feat3)

            batch_logits = model(batch_seq, gnn_emb_1, gnn_emb_2, gnn_emb_3)
            prob_list.append(batch_logits.cpu())
            label_list.append(batch_labels.cpu())

    # shape=(len(eval_loader), 2)
    probs = torch.cat(prob_list, dim=0)

    # for label alighment when using train_loader
    eval_labels = torch.cat(label_list, dim=0)

    return metrics.convert_probs(eval_labels, probs, threshold_moving=threshold_moving, thres=thres)


def train(args, data, dataset, run_id):
    """Training process.
    Parameters
    ----------
    args : dict
        Train configurations.
    data : tuple
        Reference <data.sequence.load_sequence_data>.
    dataset : tuple
        It contains train/val/test dataset. Reference <data.sequence.SequenceDataset>
    run_id : int
        The execution id in multiple runs.

    Return : list
        Model performance. Listed as follows:
        F1-macro, 
        AUC, 
        GMean, 
        Precision of positive samples, 
        Recall of positive samples,
        Average Precision (aka Area Under Precision-Recall Curve), 
        F1-score of positive samples, 
        F1-score of negative samples, 
        Recall-macro, 
        best validation epoch.
    """
    # setup devices for training and evaluation
    if args['gpu'] < 0:
        device = 'cpu'
    else:
        device = f"cuda:{args['gpu']}"

    # evaluation device
    eval_device = 'cpu' if args['cpu_eval'] else device

    # dataset
    seq_data, labels, train_nid, val_nid, test_nid, feat_dim, n_classes, n_relations = data
    train_set, val_set, test_set = dataset

    # dataloaders (without under-sampling)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=False, #shuffle=True
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False,
                            drop_last=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False,
                             drop_last=False, num_workers=0)
                        


    gnn_dataset = fraud_dataset.FraudDataset(args['dataset'], raw_dir='/your_path',train_size=0.4, val_size=0.1)
    gnn_graph = gnn_dataset[0]
    num_classes = gnn_dataset.num_classes
    if args['dataset']=='yelp':
        relation1 = dgl.edge_type_subgraph(gnn_graph, [('review', 'net_rsr', 'review')])
        relation2 = dgl.edge_type_subgraph(gnn_graph, [('review', 'net_rtr', 'review')])
        relation3 = dgl.edge_type_subgraph(gnn_graph, [('review', 'net_rur', 'review')])
    elif args['dataset']=='amazon':
        relation1 = dgl.edge_type_subgraph(gnn_graph, [('user', 'net_upu', 'user')])
        relation2 = dgl.edge_type_subgraph(gnn_graph, [('user', 'net_usu', 'user')])
        relation3 = dgl.edge_type_subgraph(gnn_graph, [('user', 'net_uvu', 'user')])


    relation1 = dgl.add_self_loop(relation1)
    relation2 = dgl.add_self_loop(relation2)
    relation3 = dgl.add_self_loop(relation3)  #GCN need add_self_loop, SAGE don't
    gnn_graph = gnn_graph.to(device)
    relation1 = relation1.to(device)
    relation2 = relation2.to(device)
    relation3 = relation3.to(device)

    train_mask_1 = relation1.ndata["train_mask"]
    train_mask_2 = relation2.ndata["train_mask"]
    train_mask_3 = relation3.ndata["train_mask"]
    train_idx_1 = torch.nonzero(train_mask_1, as_tuple=False).squeeze(1).to(device)
    train_idx_2 = torch.nonzero(train_mask_2, as_tuple=False).squeeze(1).to(device)
    train_idx_3 = torch.nonzero(train_mask_3, as_tuple=False).squeeze(1).to(device)
    train_graph1 = dgl.node_subgraph(relation1, train_idx_1)
    train_graph2 = dgl.node_subgraph(relation2, train_idx_2)
    train_graph3 = dgl.node_subgraph(relation3, train_idx_3)
    val_mask_1 = relation1.ndata["val_mask"]
    val_mask_2 = relation2.ndata["val_mask"]
    val_mask_3 = relation3.ndata["val_mask"]
    val_idx_1 = torch.nonzero(val_mask_1, as_tuple=False).squeeze(1).to(device)
    val_idx_2 = torch.nonzero(val_mask_2, as_tuple=False).squeeze(1).to(device)
    val_idx_3 = torch.nonzero(val_mask_3, as_tuple=False).squeeze(1).to(device)
    val_graph1 = dgl.node_subgraph(relation1, val_idx_1)
    val_graph2 = dgl.node_subgraph(relation2, val_idx_2)
    val_graph3 = dgl.node_subgraph(relation3, val_idx_3)
    test_mask_1 = relation1.ndata["test_mask"]
    test_mask_2 = relation2.ndata["test_mask"]
    test_mask_3 = relation3.ndata["test_mask"]
    test_idx_1 = torch.nonzero(test_mask_1, as_tuple=False).squeeze(1).to(device)
    test_idx_2 = torch.nonzero(test_mask_2, as_tuple=False).squeeze(1).to(device)
    test_idx_3 = torch.nonzero(test_mask_3, as_tuple=False).squeeze(1).to(device)
    test_graph1 = dgl.node_subgraph(relation1, test_idx_1)
    test_graph2 = dgl.node_subgraph(relation2, test_idx_2)
    test_graph3 = dgl.node_subgraph(relation3, test_idx_3)


    # model
    model = models_ragformer.TransformerEncoderNet(feat_dim=feat_dim, emb_dim=args['emb_dim'],
                                         n_classes=n_classes, n_hops=args['n_hops'],
                                         n_relations=n_relations, dim_feedforward=args['ff_dim'],
                                         n_layers=args['n_layers'], n_heads=args['n_heads'],
                                         dropout=args['dropout'])

    model.to(device)
    gnn_model = nn.ModuleList(GCN(feat_dim, args['emb_dim'], args['emb_dim']) for i in range(n_relations))
    gnn_model.to(device)

    # loss fuction and optimizer
    trainable_list = nn.ModuleList([])
    trainable_list.append(model)
    trainable_list.append(gnn_model)

    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        trainable_list.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # log tools
    summary = log_tools.SummaryBox(task_name=f"{args['dataset']}", flush_secs=args['flush_seconds'],
                                   log_dir=args['log_dir'])

    summary.save_config(args)

    # setup earlystopper to save best validation model
    if args['early_stop'] > 0:
        stopper = earlystopping.EarlyStopper(patience=args['early_stop'],
                                             dataset_name=args['dataset'],
                                             start_wall_time=summary.start_wall_time,
                                             log_dir=args['log_dir'])

    # record elapsed time per epoch
    timer = log_tools.Timer(task_name=f"Train on {args['dataset']}")
    batch_size = args['batch_size']

    for epoch in range(args['max_epochs']):
        print(f"Train on epoch {epoch:>4d}:")
        model.train()
        gnn_model.train()
        timer.start()

        # average train loss
        total_loss = 0.0

        # print(train_graph1.nodes())
        indices = train_graph1.nodes()

        for step, (batch_seq, batch_labels) in enumerate(tqdm(train_loader)):
            torch.cuda.empty_cache()
            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.to(device)

            batch_seq = torch.transpose(batch_seq, 1, 0)
            feat1 = batch_seq[0]
            feat2 = batch_seq[7]
            feat3 = batch_seq[14]
            current_indices = indices[(step*batch_size):((step+1)*batch_size)]
            subgraph_1 = dgl.node_subgraph(train_graph1, current_indices)
            subgraph_2 = dgl.node_subgraph(train_graph2, current_indices)
            subgraph_3 = dgl.node_subgraph(train_graph3, current_indices)

            gnn_emb_1 = gnn_model[0](subgraph_1, feat1)
            gnn_emb_2 = gnn_model[1](subgraph_2, feat2)
            gnn_emb_3 = gnn_model[2](subgraph_3, feat3)

            batch_logits = model(batch_seq, gnn_emb_1, gnn_emb_2, gnn_emb_3)

            loss = loss_func(batch_logits, batch_labels)
            total_loss += loss

            # log train loss per step
            summary.update_loss(loss, global_step=epoch *
                                len(train_loader) + step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timer.end()

        if epoch % args['eval_interval'] == 0:
            print(f"AVG. loss={total_loss / len(train_loader): 3.4f}, "
                  f"Elapsed time={timer.avg_time:.2f}(s)")
            val_true, val_prob, val_pred = evaluation(
                args, model, gnn_model, val_loader,val_graph1,val_graph2,val_graph3, device=eval_device)
            results = metrics.eval_model(val_true, val_prob, val_pred)

            summary.update_metrics(results, global_step=epoch)

            if args['early_stop'] > 0:
                if stopper.step(results.auc_gnn, epoch, trainable_list):
                    break

    summary.close()

    print("\nBest Epoch {}, Val {:.4f}".format(
        stopper.best_ep, stopper.best_score))

    if args['early_stop']:
        stopper.load_checkpoint(trainable_list)
        val_true, val_prob, val_pred = evaluation(
            args, trainable_list[0], trainable_list[1], val_loader,val_graph1,val_graph2,val_graph3, device=eval_device)
        val_results = metrics.eval_model(val_true, val_prob, val_pred)

        # log ROC and PRC
        summary.add_figure(figure=plot_tools.plot_roc_curve(val_true, val_prob),
                           fig_name=f"ROC-AUC Curve ({args['dataset']}_{run_id})")
        summary.add_figure(figure=plot_tools.plot_pr_curve(val_true, val_prob),
                           fig_name=f"PR Curve ({args['dataset']}_{run_id})")

        # report results on best threshold on Precision Recall Curve
        print(f"best_roc_thres: {val_results.best_roc_thres} \n"
              f"best_pr_thres: {val_results.best_pr_thres}")
        te_true, te_prob, te_pred = evaluation(args, trainable_list[0], trainable_list[1], test_loader,test_graph1,test_graph2,test_graph3,
                                               thres=val_results.best_pr_thres,
                                               device=eval_device)
        results = metrics.eval_model(te_true, te_prob, te_pred)

    return [results.f1_macro, results.auc_gnn, results.gmean_gnn,
            results.precision_1, results.recall_1, results.ap_gnn,
            results.f1_binary_1, results.f1_binary_0, results.recall_macro,
            stopper.best_ep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGFormer")

    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='<dir> to store train logs.')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='The patience when using early stop.\n'
                             'Default: 30, 0 disables earlystopper.')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Which gpu to use 0/1/..., -1 using only cpu')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of extra processes for dataloader.')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Repeat the training n times.')

    args = vars(parser.parse_args())
    train_config = utility.load_config(args['config'])
    args = utility.setup_args(args, train_config)

    # load input sequence preprocessed by graph2seq_mp.py
    data = sequence.load_sequence_data(args)
    dataset = sequence.split_dataset(data)

    # multiple runs
    result_list = []
    for i in range(args['n_runs']):
        res = train(args, data, dataset, i)
        result_list.append(res)

    # calculate average value and standard deviation
    print(result_list)
    metrics.calc_mean_sd(result_list)

#python main_ragformer.py --config configs/yelpchi.json --gpu 0  --log_dir logs --early_stop 150
