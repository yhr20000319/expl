import argparse
import sys
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
from sklearn.cluster import KMeans
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, \
    to_sparse_tensor, load_fixed_splits
from parse import parse_method_base, parse_method_ours, parse_method_pre, parser_add_main_args
from sklearn.metrics import mutual_info_score
import pandas as pd

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True


fix_seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


def get_dataset(dataset, sub_dataset=None, gen_model=None):
    ### Load and preprocess data ###
    if dataset == 'cora':
        dataset = load_nc_dataset(args.data_dir, 'cora', sub_dataset, gen_model)
    elif dataset == 'amazon-photo':
        dataset = load_nc_dataset(args.data_dir, 'amazon-photo', sub_dataset, gen_model)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]  # the number of features

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']
    return dataset


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


if args.dataset == 'cora':
    tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
    gen_model = args.gnn_gen
    dataset_tr = get_dataset(dataset='cora', sub_dataset=tr_sub[0], gen_model=gen_model)
    dataset_val = get_dataset(dataset='cora', sub_dataset=val_sub[0], gen_model=gen_model)
    datasets_te = [get_dataset(dataset='cora', sub_dataset=te_subs[i], gen_model=gen_model) for i in
                   range(len(te_subs))]
elif args.dataset == 'amazon-photo':
    tr_sub, val_sub, te_subs = [0], [1], list(range(2, 10))
    gen_model = args.gnn_gen
    dataset_tr = get_dataset(dataset='amazon-photo', sub_dataset=tr_sub[0], gen_model=gen_model)
    dataset_val = get_dataset(dataset='amazon-photo', sub_dataset=val_sub[0], gen_model=gen_model)
    datasets_te = [get_dataset(dataset='amazon-photo', sub_dataset=te_subs[i], gen_model=gen_model) for i in
                   range(len(te_subs))]
else:
    raise ValueError('Invalid dataname')

print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
for i in range(len(te_subs)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

### Load method ###
if args.method == 'erm':
    model = parse_method_base(args, dataset_tr, dataset_tr.n, dataset_tr.c, dataset_tr.d, device)
else:
    model = parse_method_ours(args, dataset_tr, dataset_tr.n, dataset_tr.c, dataset_tr.d, device)

# using rocauc as the eval function
criterion = nn.NLLLoss()
CELoss = torch.nn.CrossEntropyLoss()
eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)
print('DATASET:', args.dataset)

### Training loop ###
for run in range(args.runs):
    ### Load method ###
    if args.method == 'erm':
        model = parse_method_base(args, dataset_tr, dataset_tr.n, dataset_tr.c, dataset_tr.d, device)
    else:
        model = parse_method_ours(args, dataset_tr, dataset_tr.n, dataset_tr.c, dataset_tr.d, device)

    if args.method == 'erm':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.method == 'iene':
        model.init_env_adj(dataset_tr)
        optimizer_ir_learner = torch.optim.SGD(list(model.ir_Learner.parameters())
                                               + list(model.decoder.parameters()), lr=args.lr_a)
        optimizer_gnn_cls = torch.optim.AdamW(list(model.gnn.parameters())
                                              + list(model.cls.parameters())
                                              + list(model.e_cls.parameters()), lr=args.lr,
                                              weight_decay=args.weight_decay)
        dif_cls_list = list(model.dif_cls[0].parameters())
        for i in range(1, args.e):
            dif_cls_list = dif_cls_list + list(model.dif_cls[i].parameters())
        optimizer_cls = torch.optim.AdamW(dif_cls_list, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_env_cls = torch.optim.AdamW(model.e_cls.parameters(), lr=args.lr_a)

    best_val = float('-inf')
    x = dataset_tr.graph['node_feat'].to(args.device)
    edge_index = dataset_tr.graph['edge_index'].to(args.device)
    y = dataset_tr.label.squeeze(-1).to(args.device)
    if args.method == 'iene':
        model.train()
        for epoch in range(args.pre_epochs):
            # minimize dif_cls
            Mean = model(dataset_tr, criterion, step=1)
            dif_cls_loss = Mean
            optimizer_cls.zero_grad()
            dif_cls_loss.backward()
            optimizer_cls.step()

            # minimize cls and gnn_inv with dif_cls
            Mean = model(dataset_tr, criterion, step=2)
            cls_loss = Mean
            optimizer_gnn_cls.zero_grad()
            cls_loss.backward()
            optimizer_gnn_cls.step()

            # learn h_s by h_v
            ind_loss, rebuiled_x = model(dataset_tr, criterion, step=3)
            #rebuild_loss = F.kl_div(torch.log(rebuiled_x), x, reduction='batchmean')
            rebuild_loss = F.mse_loss(rebuiled_x, x)
            env_feature_loss = rebuild_loss + ind_loss * args.idp
            optimizer_ir_learner.zero_grad()
            env_feature_loss.backward()
            optimizer_ir_learner.step()

            #  update partition to maximize penalty
            if epoch % args.pud_ro_step == 0:
                Mean_penalty = model(dataset_tr, criterion, step=4)
                Mean_penalty = -Mean_penalty
                optimizer_env_cls.zero_grad()
                Mean_penalty.backward()
                optimizer_env_cls.step()

            accs, test_outs = evaluate_whole_graph(args, model, dataset_tr, dataset_val, datasets_te, eval_func)
            logger.add_result(run, accs)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Mean Loss: {Mean:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)

    print("****************preparing end***************")
    for epoch in range(args.epochs):
        model.train()
        if args.method == 'erm':
            optimizer.zero_grad()
            loss = model(dataset_tr, criterion)
            loss.backward()
            optimizer.step()
        elif args.method == 'iene':
            # minimize dif_cls w
            Mean = model(dataset_tr, criterion, step=1)
            dif_cls_loss = Mean
            optimizer_cls.zero_grad()
            dif_cls_loss.backward()
            optimizer_cls.step()
            # minimize cls w with dif_cls
            Mean = model(dataset_tr, criterion, step=5)
            cls_loss = Mean
            optimizer_gnn_cls.zero_grad()
            cls_loss.backward()
            optimizer_gnn_cls.step()
            # x/a = maximize penalty
            if epoch % args.pud_a_step == 0:
                model(dataset_tr, criterion, step=6)
        accs, test_outs = evaluate_whole_graph(args, model, dataset_tr, dataset_val, datasets_te, eval_func)
        logger.add_result(run, accs)

        if epoch % args.display_step == 0:
            if args.method == 'erm':
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)
            elif args.method == 'iene':
                print(f'Epoch: {epoch:02d}, '
                      f'Mean Loss: {Mean:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)

    logger.print_statistics(run)

### Save results ###

results = logger.print_statistics()
filename = f'./results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    log = f"{args.method}," + f"{args.gnn},"
    for i in range(results.shape[1]):
        r = results[:, i]
        log += f"{r.mean():.3f} ± {r.std():.3f},"
    write_obj.write(log + f"\n")
    for i in range(3, results.shape[1]):
        log = ''
        for k in range(results.shape[0]):
            log += f"{results[k, i]:.4f} "
        write_obj.write(log + f"\n")

# 训练时特征重要性

x, y = dataset_tr.graph['node_feat'].to(device), dataset_tr.label.to(device)
x.retain_grad()
x.requires_grad_(True)  # 设置输入样本需要计算梯度
edge_index = dataset_tr.graph['edge_index'].to(device)

# 前向传播
output = model.importance(x, y, edge_index, dataset_tr, criterion)

print(output)
# 计算输出对输入特征的梯度
output.backward()

# 获取输入特征的梯度
feature_gradients = x.grad.abs()

# 计算每个特征的重要性
feature_importance = feature_gradients.mean(dim=0)

# 将结果保存到CSV文件
results = {'Feature': [f'Feature {i+1}' for i in range(len(feature_importance))],
           'Importance': feature_importance.cpu().detach().numpy()}
df = pd.DataFrame(results)
df.to_csv('train_feature_importance_iene.csv', index=False)

# 打印每个特征的重要性
#for i, importance in enumerate(feature_importance):
#    print(f"Feature {i+1} importance: {importance.item()}")


# 测试时时特征重要性

x, y = dataset_te.graph['node_feat'].to(device), dataset_te.label.to(device)
x.retain_grad()
x.requires_grad_(True)  # 设置输入样本需要计算梯度
edge_index = dataset_te.graph['edge_index'].to(device)

# 前向传播
output = model.importance(x, y, edge_index, dataset_te, criterion)
print(output)
# 计算输出对输入特征的梯度
output.backward()

# 获取输入特征的梯度
feature_gradients = x.grad.abs()

# 计算每个特征的重要性
feature_importance = feature_gradients.mean(dim=0)

# 将结果保存到CSV文件
results = {'Feature': [f'Feature {i+1}' for i in range(len(feature_importance))],
           'Importance': feature_importance.cpu().detach().numpy()}
df = pd.DataFrame(results)
df.to_csv('test_feature_importance_iene.csv', index=False)