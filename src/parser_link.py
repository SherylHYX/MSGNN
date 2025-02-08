import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--year', type=int, default=2000)
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes.')
    parser.add_argument('--direction_only_task', action='store_true', help='Whether to degrade the task to consider direction only.')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--q', type=float, default=0, help='55 means 0.5/max_{i,j}(A_{i,j} - A_{j,i}), 11, 22, 33 and 44 takes 1/5, 1/5, 3/5, 4/5 of this amount.')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--trainable_q', action='store_true')
    parser.add_argument('--emb_loss_coeff', type=float, default=0, help='Coefficient for the embedding loss term.')
    parser.add_argument('--method', type=str, default='MSGNN')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--in_dim', type=int, default=20)
    parser.add_argument('--out_dim', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--input_unweighted', action='store_true', help='Whether to use unweighted edge weights for the input graph.')
    parser.add_argument('--weighted_input_feat', action='store_true', help='Whether to use edge weights to calculate degree features as input.')
    parser.add_argument('--weighted_nonnegative_input_feat', action='store_true', help='Whether to use absolute values of edge weights to calculate degree features as input.')
    parser.add_argument('--absolute_degree', action='store_true', help='Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix.')
    parser.add_argument('--sd_input_feat', action='store_true', help='Whether to use both signed and directed features as input.')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cpu', action='store_true',
                            help='use cpu')
    parser.add_argument('--debug','-D', action='store_true',
                            help='debug mode')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.') 
    parser.add_argument('--tau', type=float, default=0.5,
                        help='the regularization parameter when adding self-loops to the positive part of adjacency matrix, i.e. A -> A + tau * I, where I is the identity matrix.')
    return parser.parse_args()
