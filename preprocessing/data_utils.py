import dgl
import copy
import torch
import numpy as np
import math
from collections import namedtuple
from sklearn.cluster import KMeans
from tqdm import tqdm

from data import fraud_dataset, data_helper


class GroupFeatureSequenceLoader:
    def __init__(self, graph_data: namedtuple, default_feat=None, fanouts=None, grp_norm=False):
        if not default_feat:
            default_feat = torch.zeros(graph_data.feat_dim)

        self.default_feat = default_feat
        
        self.relations = list(graph_data.graph.etypes)
        self.features = graph_data.features
        self.labels = graph_data.labels
        self.n_groups = graph_data.n_classes + 1

        self.grp_norm = grp_norm

        self.graph = graph_data.graph
        
        self.fanouts = [-1] if fanouts is None else fanouts

        self.train_nid = graph_data.train_nid
        self.val_nid = graph_data.val_nid
        self.test_nid = graph_data.test_nid
        self.train_nid_set = set(self.train_nid.tolist())

        self.val_nid_set = set(self.val_nid.tolist())
        self.test_nid_set = set(self.test_nid.tolist())
        # print(self.val_nid_set)

    def load_batch(self, batch_nid: torch.Tensor, device='cpu', pid=0):
        grp_feat_list = []

        if batch_nid.shape == torch.Size([]):
            batch_nid = batch_nid.unsqueeze(0)
        cnt = 0
        for nid in tqdm(batch_nid):
#             cnt +=1
#             if cnt % 10000 == 0:
#                 print(f"[Process {pid:>2d}] Total processed nodes={cnt}")
            feat_list = []
            # dict{etype1: list[hop1_tensor1, hop2_tensor2,...],...}
            neighbor_dict = self._sample_multi_hop_neighbors(nid)
        
            for etype in self.graph.etypes:
                multi_hop_neighbor_list = neighbor_dict[etype]
                # ((n_classes + 1) * n_hops + 1), E)
                feat_list.append(self._group_aggregation(nid, batch_nid, multi_hop_neighbor_list))

            #  (n_relations*(n_classes + 1 + 1),E)
            grp_feat = torch.cat(feat_list, dim=0)
            grp_feat_list.append(grp_feat)
            
            del neighbor_dict

        # N,(S,E)->(N,S,E)
        batch_feats = torch.stack(grp_feat_list, dim=0)
        print(batch_feats.shape)
        # (S,N,E)
#         batch_feats = torch.transpose(batch_feats, 0, 1)

        return batch_feats

    def _sample_multi_hop_neighbors(self, nid, replace=True, probs=None):
        sampling_results = {}
        for etype in self.graph.etypes:
            rel_g = self.graph.edge_type_subgraph([etype])
            multi_hop_neighbor_list = []
            nbs = nid.item()
            for max_degree in self.fanouts:

                if max_degree == -1:
                    nbs = rel_g.in_edges(nbs)[0]
                else:
                    nbs = rel_g.in_edges(nbs)[0]
                    sample_num = min(nbs.shape[0], max_degree)
                    # print(f"[{self.__class__.__name__}] sample_num = {sample_num}")
                    nbs = np.random.choice(nbs.numpy(), size=(sample_num,), replace=replace, p=probs)
                    nbs = torch.LongTensor(nbs)


                nbs = nbs.unique()
                # print("nbs shape", nbs.shape)
                multi_hop_neighbor_list.append(nbs)


            sampling_results[etype] = multi_hop_neighbor_list

        return sampling_results

    def _kmeans_aggregation(self, multi_hop_neighbors):

        pass

    def _group_aggregation(self, nid, batch_nid, multi_hop_neighbor_list):
        nid = nid.item()

        center_feat = self.features[nid]
        feat_list = [center_feat.unsqueeze(0)]

        for neighbors in multi_hop_neighbor_list:
            if neighbors.shape == torch.Size([0]):

                agg_feat = torch.stack([self.default_feat, self.default_feat, self.default_feat], dim=0)
            else:

                nb_set = set(neighbors.tolist())


                batch_nid_set = set(batch_nid.tolist())


                # train_nb_set = nb_set.intersection(self.train_nid_set)
                # unmasked_set = train_nb_set.difference(batch_nid_set)
                unmasked_set = nb_set.intersection(self.train_nid_set)

                unmasked_set.discard(nid)
                unmasked_nid = torch.LongTensor(list(unmasked_set))
                

#                 assert nid not in unmasked_set
#                 assert unmasked_set.intersection(self.val_nid_set) == set(), \
#                     "label leakage in val_nids"
#                 assert unmasked_set.intersection(self.test_nid_set) == set(), \
#                     "label leakage in test_nids"


                masked_set = nb_set.difference(unmasked_set)
                masked_nid = torch.LongTensor(list(masked_set))

 
                pos_nid, neg_nid = data_helper.pos_neg_split(unmasked_nid, self.labels[unmasked_nid])
                h_0 = self._feat_aggregation(neg_nid)
                h_1 = self._feat_aggregation(pos_nid)
                h_2 = self._feat_aggregation(masked_nid)

                # (E,)
                agg_feat = torch.stack([h_0, h_1, h_2], dim=0)
                assert nid not in unmasked_set, f"error, node {nid} label leakage"
                # print(f"nid={nid} in unmasked_set={unmasked_set}, masked_set={masked_set}")

            feat_list.append(agg_feat)

        feat_sequence = torch.cat(feat_list, dim=0)

        return feat_sequence

    def _feat_aggregation(self, nids: torch.LongTensor):
        if nids.shape == torch.Size([0]):
            return self.default_feat
        # (*, E)
        feats = torch.index_select(self.features, dim=0, index=nids)
        # (E,)
        feats = torch.mean(feats, dim=0)

        if self.grp_norm is True:
            feats = feats * (1 / math.sqrt(nids.shape[0]))

        return feats


def prepare_data(args, add_self_loop=False):
    g = load_graph(dataset_name=args['dataset'], raw_dir=args['base_dir'],
                   train_size=args['train_size'], val_size=args['val_size'],
                   seed=args['seed'], norm=args['norm_feat'],
                   force_reload=args['force_reload'])
    
    relations = list(g.etypes)
    if add_self_loop is True:
        for etype in relations:
            g = dgl.remove_self_loop(g, etype=etype)
            g = dgl.add_self_loop(g, etype=etype)
        
        print('add self-loop for ', g)
    
    
    # Processing mask
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    val_nid = torch.nonzero(val_mask, as_tuple=True)[0]
    test_nid = torch.nonzero(test_mask, as_tuple=True)[0]

    # Processing features and labels
    n_classes = 2
    n_relations = len(g.etypes)
    features = g.ndata['feature']
    feat_dim = features.shape[1]
    labels = g.ndata['label'].squeeze().long()

    print(f"[Global] Dataset <{args['dataset']}> Overview\n"
          f"\tEntire (postive/total) {torch.sum(labels):>6} / {labels.shape[0]:<6}\n"
          f"\tTrain  (postive/total) {torch.sum(labels[train_nid]):>6} / {labels[train_nid].shape[0]:<6}\n"
          f"\tValid  (postive/total) {torch.sum(labels[val_nid]):>6} / {labels[val_nid].shape[0]:<6}\n"
          f"\tTest   (postive/total) {torch.sum(labels[test_nid]):>6} / {labels[test_nid].shape[0]:<6}\n")

    Datatype = namedtuple('GraphData', ['graph', 'features', 'labels', 'train_nid', 'val_nid',
                                        'test_nid', 'n_classes', 'feat_dim', 'n_relations'])
    graph_data = Datatype(graph=g, features=features, labels=labels, train_nid=train_nid,
                          val_nid=val_nid, test_nid=test_nid, n_classes=n_classes,
                          feat_dim=feat_dim, n_relations=n_relations)

    return graph_data


def load_graph(dataset_name='amazon', raw_dir='/your_path', train_size=0.4, val_size=0.1,
               seed=717, norm=True, force_reload=False, verbose=True) -> dict:

    if dataset_name in ['amazon', 'yelp']:
        fraud_data = fraud_dataset.FraudDataset(dataset_name,raw_dir='/your_path', train_size=train_size, val_size=val_size,
                                                random_seed=seed, force_reload=force_reload)

    g = fraud_data[0]

    if norm and (dataset_name not in ['BF10M']):
        h = data_helper.row_normalize(g.ndata['feature'], dtype=np.float32)
        g.ndata['feature'] = torch.from_numpy(h)
    else:
        g.ndata['feature'] = g.ndata['feature'].float()

    return g
