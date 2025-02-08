import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SSBM')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--K_model', type=int, default=1)
    parser.add_argument('--q', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--trainable_q', action='store_true')
    parser.add_argument('--method', type=str, default='MSGNN')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--year', type=int, default=2000)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed_ratio', type=float, default=0.1)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cpu', action='store_true',
                            help='use cpu')
    parser.add_argument('--debug','-D', action='store_true',
                            help='debug mode')
    parser.add_argument('--p', type=float, default=0.02,
                        help='probability of the existence of a link within communities, with probability (1-p), we have 0.')
    parser.add_argument('--N', type=int, default=1000,
                        help='number of nodes in the signed stochastic block model.')
    parser.add_argument('--total_n', type=int, default=1050,
                        help='total number of nodes in the polarized network.')
    parser.add_argument('--num_com', type=int, default=2,
                        help='number of polarized communities (SSBMs).')
    parser.add_argument('--K', type=int, default=2,
                        help=' number of blocks in each SSBM.')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.') 
    parser.add_argument('--tau', type=float, default=0.5,
                        help='the regularization parameter when adding self-loops to the positive part of adjacency matrix, i.e. A -> A + tau * I, where I is the identity matrix.')
    parser.add_argument('--imbalance_loss_ratio', type=float, default=1,
                        help='Ratio of imbalance loss to signed loss. Default 1.')
    parser.add_argument("--imb_normalization", type=str, default='vol_sum',
                        help="Normalization method to choose from: vol_min, vol_sum, vol_max and plain.")
    parser.add_argument("--imb_threshold", type=str, default='sort',
                        help="Thresholding method to choose from: sort, std and naive.")
    parser.add_argument('--triplet_loss_ratio', type=float, default=0.1,
                        help='Ratio of triplet loss to cross entropy loss in supervised loss part. Default 0.1.')
    parser.add_argument('--pbnc_loss_ratio', type=float, default=1,
                        help='Ratio of probablistic balanced normlized cut loss in the self-supervised loss part. Default 1.')
    parser.add_argument('--supervised_loss_ratio', type=float, default=50,
                        help='Ratio of factor of supervised loss part to self-supervised loss part.')
    parser.add_argument('--absolute_degree', action='store_true', help='Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix.')
    parser.add_argument('--sd_input_feat', action='store_true', help='Whether to use both signed and directed features as input.')
    parser.add_argument('--weighted_input_feat', action='store_true', help='Whether to use edge weights to calculate degree features as input.')
    parser.add_argument('--weighted_nonnegative_input_feat', action='store_true', help='Whether to use absolute values of edge weights to calculate degree features as input.')
    parser.add_argument('--size_ratio', type=float, default=1.5,
                        help='The size ratio of the largest to the smallest block. 1 means uniform sizes. should be at least 1.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='sign flip probability in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='direction noise level in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Number of iterations to consider for early stopping.')
    return parser.parse_args()
