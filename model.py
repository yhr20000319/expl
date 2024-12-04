from copy import copy
from loss_func import CudaCKA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree

from nets import *

def gcn_conv(x, edge_index):
    N = x.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    value = torch.ones_like(row) * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    return matmul(adj, x) # [N, D]

class GraphConvolutionBase(nn.Module):

    def __init__(self, in_features, out_features, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.residual = residual
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, x0):
        hi = gcn_conv(x, adj)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(x, self.weight_r)
        return output

class Base(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Base, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           use_bn=not args.no_bn)
            #self.gnn = GraphConvolutionBase(in_features = d, out_features= args.hidden_channels)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=args.hidden_channels,
                            num_layers=args.num_layers,
                            dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                              hidden_channels=args.hidden_channels,
                              out_channels=args.hidden_channels,
                              dropout=args.dropout,
                              alpha=args.gpr_alpha,
                              )
        elif gnn == 'gcnii':
            self.gnn = GCNII(in_channels=d,
                             hidden_channels=args.hidden_channels,
                             out_channels=args.hidden_channels,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                             alpha=args.gcnii_alpha,
                             lamda=args.gcnii_lamda)
        self.n = n
        self.device = device
        self.gnn_name = gnn
        self.args = args
        self.cls = Node_Cls(args.hidden_channels, args.hidden_channels, c, device)

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data, criterion):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)

        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        out = self.cls(out)
        if self.args.dataset == 'elliptic':
            loss = self.sup_loss(y[data.mask], out[data.mask], criterion)
        else:
            loss = self.sup_loss(y, out, criterion)
        return loss

    def importance(self, x, y, edge_index, data, criterion):

        out = self.gnn(x, edge_index)
        out = self.cls(out)
        if self.args.dataset == 'elliptic':
            loss = self.sup_loss(y[data.mask], out[data.mask], criterion)
        else:
            loss = self.sup_loss(y, out, criterion)
        return loss

    def inference(self, data, partial=False):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        if partial:
            x[:, -10:] = torch.zeros(x.size(0), 10, dtype=x.dtype).to(x.device)
        out = self.gnn(x, edge_index)
        out = self.cls(out)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss


class irrelavant_Learner(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, device):
        super(irrelavant_Learner, self).__init__()
        self.gnn = GCN(in_channels=in_dim,
                       hidden_channels=h_dim,
                       out_channels=out_dim,
                       num_layers=2,
                       dropout=0.0)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.Tanh()
        self.device = device

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, adj):
        out = self.gnn(x, adj)
        out = self.act(out)
        return out  #


class relavant_Learner(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, device):
        super(relavant_Learner, self).__init__()
        self.gnn = GCN(in_channels=in_dim,
                       hidden_channels=h_dim,
                       out_channels=out_dim,
                       num_layers=2,
                       dropout=0.0)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.Tanh()
        self.device = device

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, adj):
        out = self.gnn(x, adj)
        out = self.act(out)
        return out  #


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, device):
        super(Decoder, self).__init__()
        self.gnn = GCN(in_channels=in_dim,
                       hidden_channels=h_dim,
                       out_channels=out_dim,
                       num_layers=2,
                       dropout=0.0)
        self.norm = nn.LayerNorm(out_dim)
        self.device = device

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, adj):
        out = self.gnn(x, adj)
        out = self.norm(out)
        return out  #


class Ir_Cls(nn.Module):
    def __init__(self, in_dim, h_dim, c_dim, device):
        super(Ir_Cls, self).__init__()
        self.cls_lin1 = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.cls_lin2 = nn.Linear(h_dim, c_dim)
        self.norm = nn.LayerNorm(c_dim)
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.cls_lin1.weight)
        nn.init.uniform_(self.cls_lin2.weight)

    def forward(self, ir_feature):
        out = self.cls_lin1(ir_feature)
        out = self.act(out)
        out = self.cls_lin2(out)
        out = self.norm(out)
        return out  #


class Environment_Cls(nn.Module):
    def __init__(self, in_dim, h_dim, c_dim, device):
        super(Environment_Cls, self).__init__()
        self.cls_lin1 = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.cls_lin2 = nn.Linear(h_dim, c_dim)
        self.norm = nn.LayerNorm(c_dim)
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.cls_lin1.weight)
        nn.init.uniform_(self.cls_lin2.weight)

    def forward(self, ir_feature):
        out = self.cls_lin1(ir_feature)
        out = self.act(out)
        out = self.cls_lin2(out)
        out = self.act(out)
        return out  #


class Node_Cls(nn.Module):
    def __init__(self, in_dim, h_dim, c_dim, device):
        super(Node_Cls, self).__init__()
        self.cls_lin1 = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.cls_lin2 = nn.Linear(h_dim, c_dim)
        self.norm = nn.LayerNorm(c_dim)
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.cls_lin1.weight)
        nn.init.uniform_(self.cls_lin2.weight)

    def forward(self, output_g):
        out = self.cls_lin1(output_g)
        out = self.act(out)
        out = self.cls_lin2(out)
        out = self.norm(out)
        return out  #


class Ir_Learner(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Ir_Learner, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=args.hidden_channels,
                            num_layers=args.num_layers,
                            dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                              hidden_channels=args.hidden_channels,
                              out_channels=args.hidden_channels,
                              dropout=args.dropout,
                              alpha=args.gpr_alpha,
                              )
        elif gnn == 'gcnii':
            self.gnn = GCNII(in_channels=d,
                             hidden_channels=args.hidden_channels,
                             out_channels=args.hidden_channels,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                             alpha=args.gcnii_alpha,
                             lamda=args.gcnii_lamda)
        self.p = 0.2
        self.n = n
        self.d = d
        self.device = device
        self.gnn_name = gnn
        self.args = args
        self.ir_Learner = irrelavant_Learner(d, args.hidden_channels, args.hidden_channels, device)
        self.re_Learner = relavant_Learner(d, args.hidden_channels, args.hidden_channels, device)
        self.decoder = Decoder(args.hidden_channels + args.hidden_channels + 1, args.hidden_channels, d, device)
        self.ir_cls = Ir_Cls(args.hidden_channels, args.hidden_channels, c, device)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.ir_Learner.reset_parameters()
            self.re_Learner.reset_parameters()
            self.decoder.reset_parameters()
            self.g_cls.reset_parameters()

    def forward(self, data, criterion):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        Loss = []
        ir_feature = self.ir_Learner(x, edge_index)
        re_feature = self.re_Learner(x, edge_index)
        rebuiled_x = self.decoder(torch.cat([ir_feature, re_feature, y], dim=1), edge_index)
        g_class = self.ir_cls(ir_feature)

        if self.args.dataset == 'elliptic':
            loss = self.sup_loss(y[data.mask], g_class[data.mask], criterion)
        else:
            loss = self.sup_loss(y, g_class, criterion)
        Loss.append(loss.view(-1))
        Loss = torch.cat(Loss, dim=0)
        ir_loss = torch.mean(Loss)
        return ir_loss, rebuiled_x

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss


class Model(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Model, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=args.hidden_channels,
                            num_layers=args.num_layers,
                            dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=args.hidden_channels,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                              hidden_channels=args.hidden_channels,
                              out_channels=args.hidden_channels,
                              dropout=args.dropout,
                              alpha=args.gpr_alpha,
                              )
        elif gnn == 'gcnii':
            self.gnn = GCNII(in_channels=d,
                             hidden_channels=args.hidden_channels,
                             out_channels=args.hidden_channels,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                             alpha=args.gcnii_alpha,
                             lamda=args.gcnii_lamda)
        self.p = 0.2
        self.n = n
        self.d = d
        self.e = args.e
        self.device = device
        self.gnn_name = gnn
        self.args = args
        self.adj_continuous = torch.nn.parameter.Parameter(torch.FloatTensor(n, n)).to(self.device)
        self.adj_continuous.data.fill_(0)
        self.ir_Learner = irrelavant_Learner(d, args.hidden_channels, args.hidden_channels, device)
        self.re_Learner = relavant_Learner(d, args.hidden_channels, args.hidden_channels, device)
        self.e_cls = Environment_Cls(args.hidden_channels, args.hidden_channels, args.e, device)
        self.decoder = Decoder(args.hidden_channels + args.hidden_channels, args.hidden_channels, d, device)
        self.dif_cls = []
        self.cls = Node_Cls(args.hidden_channels, args.hidden_channels, c, device)
        self.env_adj = []
        for i in range(args.e):
            cls = Node_Cls(args.hidden_channels, args.hidden_channels, c, device).to(self.device)
            self.dif_cls.append(cls)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.cls.reset_parameters()
        self.e_cls.reset_parameters()
        self.ir_Learner.reset_parameters()
        self.decoder.reset_parameters()
        for i in range(self.args.e):
            self.dif_cls[i].reset_parameters()

    def init_env_adj(self, data):
        for i in range(self.e):
            edge_index = data.graph['edge_index'].to(self.device)
            self.env_adj.append(edge_index)

    def forward(self, data, criterion, step):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        ir_feature = self.ir_Learner(x, edge_index)
        e_new = self.e_cls(ir_feature)
        Loss = []
        scale = torch.tensor(1.).cuda().requires_grad_()
        if step == 1:

            out = self.gnn(x, edge_index).to(self.device)
            for i in range(self.e):
                dif_out = self.dif_cls[i](out)
                if self.args.dataset == 'elliptic':
                    loss = self.sup_loss(y[data.mask], dif_out[data.mask], criterion)
                else:
                    loss = self.sup_loss(y, dif_out, criterion)
                # Loss.append(loss)
                Loss.append(loss.view(-1))
            Loss = torch.cat(Loss, dim=0)
            Loss = torch.mul(Loss, torch.mean(e_new, dim=0))
            Mean = torch.mean(Loss)
            return Mean
        if step == 2:
            Loss = []
            fine_out = self.gnn(x, edge_index)
            fine_out = self.cls(fine_out)
            out = self.gnn(x, edge_index).to(self.device)
            for i in range(self.e):
                dif_out = self.dif_cls[i](out)
                if self.args.dataset == 'elliptic':
                    y = y[data.mask]
                    dif_out = dif_out[data.mask]
                    fine_out = fine_out[data.mask]
                loss = self.sup_loss(y, dif_out, criterion)
                loss2 = self.sup_loss(y, fine_out, criterion)
                Mean = loss2
                loss = loss2 - loss
                Loss.append(loss.view(-1))
            Loss = torch.cat(Loss, dim=0)
            Loss = torch.mul(Loss, torch.mean(e_new, dim=0))
            penalty = torch.mean(Loss)
            target = Mean + penalty * self.args.penalty_weight
            return target
        if step == 3:
            env_feature = self.ir_Learner(x, edge_index)
            inv_feature = self.gnn(x, edge_index)
            rebuiled_x = self.decoder(torch.cat([env_feature, inv_feature], dim=1), edge_index)
            ind_loss = HSIC(self.args, env_feature, inv_feature, 0, 0)
            return ind_loss, rebuiled_x
        if step == 4:  # calculate penalty to update ro
            x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
            edge_index = data.graph['edge_index'].to(self.device)
            ir_feature = self.ir_Learner(x, edge_index)
            e_partition = self.e_cls(ir_feature)
            Loss = []
            out = self.gnn(x, edge_index)
            fine_out = self.cls(out)
            for i in range(self.e):
                dif_out = self.dif_cls[i](out)
                if self.args.dataset == 'elliptic':
                    loss = self.CELoss_no_sum(dif_out[data.mask], y[data.mask])
                    loss2 = self.CELoss_no_sum(fine_out[data.mask], y[data.mask])
                    loss = loss2 - loss
                else:
                    loss = self.CELoss_no_sum(dif_out, y)
                    loss2 = self.CELoss_no_sum(fine_out, y)
                    loss = loss2 - loss
                Loss.append(loss)
                # Loss.append(loss.view(-1))
            Loss = torch.cat(Loss, dim=1)
            Loss = torch.mul(Loss, e_partition)
            penalty = torch.mean(torch.sum(Loss, dim=1))
            return penalty

        if step == 5:
            x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
            edge_index = data.graph['edge_index'].to(self.device)
            ir_feature = self.ir_Learner(x, edge_index)
            e_new = self.e_cls(ir_feature)
            Loss_env = []
            out_env = []
            for i in range(self.e):
                out = self.gnn(x, self.env_adj[i])
                out = self.cls(out)
                out_env.append(out)
            # penalty_var
            if self.args.var_type == 'ene':
                if self.args.dataset == 'elliptic':
                    y = y[data.mask]
                    for i in range(self.e):
                        out_env[i] = out_env[i][data.mask]
                for i in range(self.e):
                    loss = self.CELoss_no_sum(out_env[i], y)
                    Loss_env.append(loss)
                Loss_env = torch.cat(Loss_env, dim=1)
                Var = torch.var(Loss_env, dim=1)
                Var = torch.mean(Var)
            else:
                if self.args.dataset == 'elliptic':
                    for i in range(self.e):
                        loss = self.sup_loss(y[data.mask], out_env[i][data.mask], criterion)
                        Loss_env.append(loss.view(-1))
                else:
                    for i in range(self.e):
                        loss = self.sup_loss(y, out_env[i], criterion)
                        Loss_env.append(loss.view(-1))
                Loss_env = torch.cat(Loss_env, dim=0)
                Var = torch.var(Loss_env)
            # penalty_cls_env
            Loss = []
            fine_out = self.gnn(x, edge_index)
            fine_out = self.cls(fine_out)
            out = self.gnn(x, edge_index).to(self.device)
            for i in range(self.e):
                dif_out = self.dif_cls[i](out)
                if self.args.dataset == 'elliptic':
                    y = y[data.mask]
                    dif_out = dif_out[data.mask]
                    fine_out = fine_out[data.mask]
                loss = self.sup_loss(y, dif_out, criterion)
                loss2 = self.sup_loss(y, fine_out, criterion)
                Mean = loss2
                loss = loss2 - loss
                Loss.append(loss.view(-1))
            Loss = torch.cat(Loss, dim=0)
            Loss = torch.mul(Loss, torch.mean(e_new, dim=0))
            penalty = torch.mean(Loss)
            target = Mean + penalty * self.args.penalty_weight + Var * self.args.beta
            return target
        if step == 6:
            if self.args.mode == 'x':
                x.requires_grad_(True)
                x.retain_grad()
            Loss = []
            for i in range(self.e):
                self.adj_continuous.data = torch.eye(self.n).to(self.device)
                env_feature = self.ir_Learner(self.adj_continuous @ x, self.env_adj[i])
                env_partition = self.e_cls(env_feature)
                inv_feature_before = self.gnn(x, edge_index)
                inv_feature_now = self.gnn(self.adj_continuous @ x, self.env_adj[i])
                CEloss = nn.CrossEntropyLoss()
                target = torch.full((self.n,), i).to(self.device)
                # target = F.one_hot(target)
                ce_loss = CEloss(env_partition, target)
                l2_loss = F.mse_loss(inv_feature_now, inv_feature_before)
                loss = ce_loss + l2_loss*self.args.niu
                if self.args.mode == 'adj':
                    num_sample = self.args.num_sample
                    n = self.n
                    grad = torch.autograd.grad(loss, self.adj_continuous, retain_graph=True)[0]
                    # grad = adj_old.grad
                    Bk = torch.clamp(grad, 0, 1)
                    # Bk = torch.mm(ir_feature, torch.transpose(ir_feature, 0, 1))
                    A = to_dense_adj(edge_index)[0].to(torch.int)
                    A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
                    P = torch.softmax(Bk, dim=0)
                    S = torch.multinomial(P, num_samples=num_sample)
                    M = torch.zeros(n, n, dtype=torch.float).to(self.device)
                    col_idx = torch.arange(0, n).unsqueeze(1).repeat(1,
                                                                     num_sample)
                    M[S, col_idx] = 1.
                    C = A + M * (A_c - A)
                    adj_new = dense_to_sparse(C)[0]  # Reduce complexity  Return row and column indexes
                    self.env_adj[i] = adj_new
                elif self.args.mode == 'x':
                    num_sample = self.args.num_sample
                    n = self.n
                    adj_new = edge_index
                    x_new = torch.ones(n, self.d, dtype=torch.float).to(self.device)
                    # x_c = self.decoder(torch.cat([ir_feature, re_feature, y], dim=1))
                    x_c = torch.clamp(x.grad, 0, 1)
                    P = torch.softmax(x_c, dim=0)
                    S = torch.multinomial(P, num_samples=num_sample)
                    row_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
                    x_new[row_idx, S] = 0
                    x_new = torch.mul(x_new, x).detach()

    def importance(self, x, y, edge_index, data, criterion):

        out = self.gnn(x, edge_index)
        out = self.cls(out)
        if self.args.dataset == 'elliptic':
            loss = self.sup_loss(y[data.mask], out[data.mask], criterion)
        else:
            loss = self.sup_loss(y, out, criterion)
        return loss

    def inference(self, data, partial=False):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        if partial:
            x[:, -10:] = torch.zeros(x.size(0), 10, dtype=x.dtype).to(x.device)
        out = self.gnn(x, edge_index)
        out = self.cls(out)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss

    def CELoss_no_sum(self, logits, target):
        # logits: [N, C], target: [N, 1]
        # loss = sum(-y_i * log(c_i))
        logits = F.log_softmax(logits, 1)
        logits = logits.gather(1, target)
        loss_no_sum = -1 * logits
        return loss_no_sum


def HSIC(args, xo, xc, o_logs, c_logs):
    cka = CudaCKA(device=args.device)
    if args.kernel == 'rbf':
        if args.idp_type == 'xo':
            idp_loss = cka.rbf_CKA(xo, xc, sigma=None)
        elif args.idp_type == 'o_logs':
            idp_loss = cka.rbf_CKA(o_logs, c_logs, sigma=None)
    elif args.kernel == 'linear':
        if args.idp_type == 'xo':
            idp_loss = cka.linear_CKA(xo, xc)
        elif args.idp_type == 'o_logs':
            idp_loss = cka.linear_CKA(o_logs, c_logs)
        # idp_loss = cka.linear_CKA(o_logs, c_logs)
    elif args.kernel == 'poly':
        if args.idp_type == 'xo':
            idp_loss = cka.poly_CKA(xo, xc)
        elif args.idp_type == 'o_logs':
            idp_loss = cka.poly_CKA(o_logs, c_logs)
    elif args.kernel == 'rq':
        if args.idp_type == 'xo':
            idp_loss = cka.rq_CKA(xo, xc)
        elif args.idp_type == 'o_logs':
            idp_loss = cka.rq_CKA(o_logs, c_logs)
    else:
        assert False
    # if torch.isnan(idp_loss):
    #    assert False
    return idp_loss
