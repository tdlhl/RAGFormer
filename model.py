import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler


class CAREConv(nn.Module):
    """One layer of CARE-GNN."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_classes,
        edges,
        activation=None,
        step_size=0.02,
    ):
        super(CAREConv, self).__init__()

        self.activation = activation
        self.step_size = step_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.edges = edges
        self.dist = {}

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.MLP = nn.Linear(self.in_dim, self.num_classes)

        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}
        for etype in edges:
            self.p[etype] = 0.5
            self.last_avg_dist[etype] = 0
            self.f[etype] = []
            self.cvg[etype] = False

    def _calc_distance(self, edges):
        # formula 2
        d = th.norm(
            th.tanh(self.MLP(edges.src["h"]))
            - th.tanh(self.MLP(edges.dst["h"])),
            1,
            1,
        )
        return {"d": d}

    def _top_p_sampling(self, g, p):
        # this implementation is low efficient
        # optimization requires dgl.sampling.select_top_p requested in issue #3100
        dist = g.edata["d"]
        neigh_list = []
        for node in g.nodes():
            edges = g.in_edges(node, form="eid")
            num_neigh = th.ceil(g.in_degrees(node) * p).int().item()
            neigh_dist = dist[edges]
            if neigh_dist.shape[0] > num_neigh:
                neigh_index = np.argpartition(
                    neigh_dist.cpu().detach(), num_neigh
                )[:num_neigh]
            else:
                neigh_index = np.arange(num_neigh)
            neigh_list.append(edges[neigh_index])
        return th.cat(neigh_list)

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata["h"] = feat

            hr = {}
            for i, etype in enumerate(g.canonical_etypes):
                g.apply_edges(self._calc_distance, etype=etype)
                self.dist[etype] = g.edges[etype].data["d"]
                sampled_edges = self._top_p_sampling(g[etype], self.p[etype])

                # formula 8
                g.send_and_recv(
                    sampled_edges,
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_%s" % etype[1]),
                    etype=etype,
                )
                hr[etype] = g.ndata["h_%s" % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = (
                th.Tensor(list(self.p.values())).view(-1, 1, 1).to(g.device)
            )
            h_homo = th.sum(th.stack(list(hr.values())) * p_tensor, dim=0)
            h_homo += feat
            if self.activation is not None:
                h_homo = self.activation(h_homo)

            return self.linear(h_homo)


class CAREGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        hid_dim=64,
        edges=None,
        num_layers=2,
        activation=None,
        step_size=0.02,
    ):
        super(CAREGNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.edges = edges
        self.activation = activation
        self.step_size = step_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        if self.num_layers == 1:
            # Single layer
            self.layers.append(
                CAREConv(
                    self.in_dim,
                    self.num_classes,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

        else:
            # Input layer
            self.layers.append(
                CAREConv(
                    self.in_dim,
                    self.hid_dim,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

            # Hidden layers with n - 2 layers
            for i in range(self.num_layers - 2):
                self.layers.append(
                    CAREConv(
                        self.hid_dim,
                        self.hid_dim,
                        self.num_classes,
                        self.edges,
                        activation=self.activation,
                        step_size=self.step_size,
                    )
                )

            # Output layer
            self.layers.append(
                CAREConv(
                    self.hid_dim,
                    self.num_classes,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

    def forward(self, graph, feat):
        # For full graph training, directly use the graph
        # formula 4
        sim = th.tanh(self.layers[0].MLP(feat))

        # Forward of n layers of CARE-GNN
        for layer in self.layers:
            feat = layer(graph, feat)

        return feat, sim

    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in self.edges:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(idx, form="eid", etype=etype)
                    avg_dist = th.mean(layer.dist[etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -= self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.20)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h

class GATv2(nn.Module):
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATv2Conv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    bias=False,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                bias=False,
                share_weights=True,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gatv2_layers[-1](g, h).mean(1)
        return logits

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
