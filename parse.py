from model import Base, Model, Ir_Learner

def parse_method_base(args, dataset, n, c, d, device):
    if args.gnn == 'gcn':
        model = Base(args, n, c, d, 'gcn', device).to(device)
    elif args.gnn == 'sage':
        model = Base(args, n, c, d, 'sage', device).to(device)
    elif args.gnn == 'gat':
        model = Base(args, n, c, d, 'gat', device).to(device)
    elif args.gnn == 'gpr':
        model = Base(args, n, c, d, 'gpr', device).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parse_method_ours(args, dataset, n, c, d, device):
    if args.gnn == 'gcn':
        model = Model(args, n, c, d, 'gcn', device).to(device)
    elif args.gnn == 'sage':
        model = Model(args, n, c, d, 'sage', device).to(device)
    elif args.gnn == 'gat':
        model = Model(args, n, c, d, 'gat', device).to(device)
    elif args.gnn == 'gpr':
        model = Model(args, n, c, d, 'gpr', device).to(device)
    else:
        raise ValueError('Invalid method')
    return model

def parse_method_pre(args, dataset, n, c, d, device):
    pre_model = Ir_Learner(args, n, c, d, 'gcn', device).to(device)
    return pre_model

def parser_add_main_args(parser):
    # dataset and protocol
    parser.add_argument('--data_dir', type=str, default='../../data')  # need to be specified
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn_gen', type=str, default='gcn', choices=['gcn', 'gat', 'sgc'],
                        help='random initialized gnn for data generation')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')

    # model
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gnn', type=str, default='gcn', help='gnn backbone')
    parser.add_argument('--method', type=str, default='erm',
                        choices=['erm', 'iene'])
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pre_epochs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--gcnii_alpha', type=float, default=.1,
                        help='alpha for gcnii')
    parser.add_argument('--gcnii_lamda', type=float, default=1.0,
                        help='lambda for gcnii')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')

    # for graph edit model
    parser.add_argument('--e', type=int, default=3,
                        help='num of views for environment')
    parser.add_argument('--K', type=int, default=3,
                        help='num of views for data augmentation')
    parser.add_argument('--T', type=int, default=1,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--num_sample', type=int, default=5,
                        help='num of samples for each node with graph edit')
    parser.add_argument('--beta', type=float, default=3.0,
                        help='weight for mean of risks from multiple domains')
    parser.add_argument('--lr_a', type=float, default=0.005,
                        help='learning rate for graph edit model')
    parser.add_argument('--ir_step', type=float, default=0.005,
                        help='learning step for ir learner')
    parser.add_argument('--mode', type=str,  default='adj',
                        help='rebuild what kind of object of graph')
    parser.add_argument('--var_type', type=str,
                        help='the inviriant penalty type')
    parser.add_argument('--penalty_weight', type=float, default=4,
                        help='the weight of penalty')

    # for HSIC
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='rbf/linear/poly/rq')
    parser.add_argument('--idp_type', type=str, default='xo',
                        help='xo/o_logs')
    parser.add_argument('--idp', type=float, default=0.1,
                        help='[0.1, 0.2, 0.3, 0.4, 0.5], independent penalty factor')

    # for loss
    parser.add_argument('--niu', type=float, default=1,
                        help='trade off for reconstruction and extrapolation in step 6')
    parser.add_argument('--pud_ro_step', type=int,
                        default=5, help='penalty ro update step')
    parser.add_argument('--pud_a_step', type=int,
                        default=5, help='penalty A update step')

