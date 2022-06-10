import os
import pickle as pk
import sys

from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
import numpy as np
import torch
from torch_geometric_signed_directed.nn import SSSNET_node_clustering
from torch_geometric_signed_directed.data import \
    SignedData
from torch_geometric_signed_directed.utils import \
    (Prob_Balanced_Normalized_Loss, Prob_Imbalance_Loss, scipy_sparse_to_torch_sparse, 
triplet_loss_node_classification, in_out_degree)
from tensorboardX import SummaryWriter
import numpy.random as rnd

from MSGNN import MSGNN_node_classification
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), 'SigMaNet'))
from Signum import SigMaNet_node_prediction_one_laplacian
import laplacian
from parser_node import parameter_parser

args = parameter_parser()

device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

sub_dir_name = 'runs' + str(args.runs) + 'epochs' + str(args.epochs) + 'early' + str(args.early_stopping) + \
    '100train_ratio' + str(int(100*args.train_ratio)) + '100val_ratio' + str(int(100*args.val_ratio)) + '100seed_ratio' + str(int(100*args.seed_ratio)) + \
        '1000lr' + str(int(1000*args.lr)) + '1000weight_decay' + str(int(1000*args.weight_decay)) + '100dropout' + str(int(100*args.dropout)) 
if args.seed != 0:
    sub_dir_name += 'seed' + str(args.seed)
suffix = 'hidden' + str(args.hidden) + 'trip_ratio' + str(int(100*args.triplet_loss_ratio)) 
suffix += 'sup_ratio' + str(int(100*args.supervised_loss_ratio)) + 'imb_ratio' + str(int(100*args.imbalance_loss_ratio))
if args.imbalance_loss_ratio > 0:
    suffix += 'imb_norm' + str(args.imb_normalization) + 'imb_thre' + str(args.imb_threshold) 
if args.supervised_loss_ratio > 0:
    suffix += '10sup_loss_rat' + str(int(10*args.supervised_loss_ratio))
    if args.triplet_loss_ratio > 0:
        suffix += 'trip_ratio' + str(int(100*args.triplet_loss_ratio))
if args.pbnc_loss_ratio != 1:
    suffix += '10pbnc_loss_rat' + str(int(10*args.pbnc_loss_ratio))

if args.method == 'MSGNN':
    suffix += 'hidden' + str(args.hidden) + 'K' + str(args.K_model) + '100q' + str(int(100*args.q)) + 'trainable_q' + str(args.trainable_q)
elif args.method == 'SSSNET':
    suffix += 'hidden' + str(args.hidden) + 'hop' + str(args.hop) + '100tau' + str(int(100*args.tau))
num_input_feat = 2
if args.sd_input_feat:
    suffix += 'SdInput'
    num_input_feat = 4
if args.weighted_input_feat:
    suffix += 'WeightedInput'
logs_folder_name = 'Finance_runs'
if args.debug: 
    args.runs = 2
    args.epochs = 2
    logs_folder_name = 'debug_Finance_runs'
log_dir = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../' + logs_folder_name, args.dataset, args.method, sub_dir_name)
writer = SummaryWriter(log_dir=log_dir+'/'+suffix)

rnd.seed(args.seed)
args.K = 10
F_plain = 10 # suppose there are 10 edges in the meta-graph
assert args.dataset in ['OPCL', 'pvCLCL']
labels = torch.LongTensor(np.load(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/Lead_Lag/'+args.dataset+'_node_labels.npy')))
args.dataset += str(args.year)
save_path = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/Lead_Lag', args.dataset+'.npy')
A = sp.csr_matrix(np.load(save_path))
data = SignedData(A=A, y=labels)


A_abs = A.copy()
A_abs.data = np.abs(A_abs.data)
num_classes = int(data.y.max() - data.y.min() + 1)
if args.weighted_input_feat:
    data.x = in_out_degree(data.edge_index, size=int(data.edge_index.max()-data.edge_index.min())+1, signed=args.sd_input_feat, \
        edge_weight=data.edge_weight).to(device)
else:
    data.x = in_out_degree(data.edge_index, size=int(data.edge_index.max()-data.edge_index.min())+1, signed=args.sd_input_feat).to(device)

data.node_split(data_split=args.runs, train_size_per_class=args.train_ratio, val_size_per_class=args.val_ratio, 
    test_size_per_class=1-args.train_ratio-args.val_ratio, seed_size_per_class=args.seed_ratio)
data.separate_positive_negative()
data = data.to(device)
loss_func_ce = torch.nn.NLLLoss()
loss_func_imbalance = Prob_Imbalance_Loss(F_plain)

def train(A_abs_torch_train, features, edge_index, edge_weight, mask, seed_mask, loss_func_pbnc, y):
    model.train()
    Z, log_prob, _, prob = model(features, features, edge_index=edge_index, 
                    edge_weight=edge_weight)
    loss_pbnc = loss_func_pbnc(prob[mask])
    loss_triplet = 0
    if args.triplet_loss_ratio > 0:
        loss_triplet = triplet_loss_node_classification(y=y[seed_mask], Z=Z[seed_mask], n_sample=500, thre=0.1)
    loss_imbalance = 0
    if args.imbalance_loss_ratio:
        loss_imbalance = loss_func_imbalance(prob[mask], A_abs_torch_train,
                num_classes, args.imb_normalization, args.imb_threshold)
    loss_ce = loss_func_ce(log_prob[seed_mask], y[seed_mask])
    loss = args.supervised_loss_ratio*(loss_ce + args.triplet_loss_ratio*loss_triplet) + \
        args.pbnc_loss_ratio * loss_pbnc + args.imbalance_loss_ratio * loss_imbalance
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return loss.detach().item(), train_ari

def test(features, edge_index, edge_weight, mask, y):
    model.eval()
    with torch.no_grad():
        _, _, _, prob = model(features, features, edge_index=edge_index, 
                    edge_weight=edge_weight)
    test_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return test_ari

def train_SSSNET(A_abs_torch_train, features, edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, mask, seed_mask, loss_func_pbnc, y):
    model.train()
    Z, log_prob, _, prob = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, features)
    loss_pbnc = loss_func_pbnc(prob[mask])
    loss_triplet = 0
    if args.triplet_loss_ratio > 0:
        loss_triplet = triplet_loss_node_classification(y=y[seed_mask], Z=Z[seed_mask], n_sample=500, thre=0.1)
    loss_imbalance = 0
    if args.imbalance_loss_ratio:
        loss_imbalance = loss_func_imbalance(prob[mask], A_abs_torch_train,
                num_classes, args.imb_normalization, args.imb_threshold)
    loss_ce = loss_func_ce(log_prob[seed_mask], y[seed_mask])
    loss = args.supervised_loss_ratio*(loss_ce + args.triplet_loss_ratio*loss_triplet) + \
        args.pbnc_loss_ratio * loss_pbnc + args.imbalance_loss_ratio * loss_imbalance
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return loss.detach().item(), train_ari

def test_SSSNET(features, edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, mask, y):
    model.eval()
    with torch.no_grad():
        _, _, _, prob = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, features)
    test_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return test_ari

def train_SigMaNet(features, mask, seed_mask, loss_func_pbnc, y):
    model.train()
    Z, log_prob, _, prob = model(features, features)
    loss_pbnc = loss_func_pbnc(prob[mask])
    loss_triplet = 0
    if args.triplet_loss_ratio > 0:
        loss_triplet = triplet_loss_node_classification(y=y[seed_mask], Z=Z[seed_mask], n_sample=500, thre=0.1)
    loss_imbalance = 0
    if args.imbalance_loss_ratio:
        loss_imbalance = loss_func_imbalance(prob[mask], A_abs_torch_train,
                num_classes, args.imb_normalization, args.imb_threshold)
    loss_ce = loss_func_ce(log_prob[seed_mask], y[seed_mask])
    loss = args.supervised_loss_ratio*(loss_ce + args.triplet_loss_ratio*loss_triplet) + \
        args.pbnc_loss_ratio * loss_pbnc + args.imbalance_loss_ratio * loss_imbalance
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return loss.detach().item(), train_ari

def test_SigMaNet(features, mask, y):
    model.eval()
    with torch.no_grad():
        _, _, _, prob = model(features, features)
    test_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return test_ari

# log dir
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../Finance_logs/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../Finance_logs/'+args.dataset)

if os.path.isdir(os.path.join(dir_name, sub_dir_name, args.method, suffix)) == False:
    try:
        os.makedirs(os.path.join(dir_name, sub_dir_name, args.method, suffix))
    except FileExistsError:
        print('Folder exists for {}!'.format(sub_dir_name, args.method))

res_array = np.zeros((args.runs, 3))
for split in range(data.train_mask.shape[1]):
    best_model_path = os.path.join(dir_name, sub_dir_name, args.method, suffix, str(split) + '.pkl')
    if args.method == 'MSGNN':
        model = MSGNN_node_classification(q=args.q, K=args.K_model, num_features=data.x.shape[1], hidden=args.hidden, label_dim=num_classes, 
        dropout=args.dropout, cached=(not args.trainable_q)).to(device)
    elif args.method == 'SSSNET':
        model = SSSNET_node_clustering(nfeat=data.x.shape[1], hidden=args.hidden, nclass=num_classes, dropout=args.dropout, 
        hop=args.hop, fill_value=args.tau, directed=data.is_directed).to(device)
    elif args.method == 'SigMaNet':
        edge_index, norm_real, norm_imag = laplacian.process_magnetic_laplacian(edge_index=data.edge_index, gcn=False, net_flow=False, x_real=data.x, edge_weight=data.edge_weight, \
         normalization = 'sym', return_lambda_max = False)
        model = SigMaNet_node_prediction_one_laplacian(K=args.K, num_features=data.x.size(-1), hidden=1, label_dim=num_classes,
                            i_complex = False,  layer=2, follow_math=False, gcn=False, net_flow=False, unwind = True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_index = data.train_mask[:, split].cpu().numpy()
    val_index = data.val_mask[:, split]
    test_index = data.test_mask[:, split]
    seed_index = data.seed_mask[:, split]
    A_abs_torch_train = scipy_sparse_to_torch_sparse(A_abs[train_index][:, train_index]).to(device)
    loss_func_pbnc = Prob_Balanced_Normalized_Loss(A_p=sp.csr_matrix(data.A_p)[train_index][:, train_index], 
    A_n=sp.csr_matrix(data.A_n)[train_index][:, train_index])
    best_val_ari = -100
    no_improve_epoch = 0
    if args.method == 'SSSNET':
        for epoch in range(args.epochs):
            train_loss, train_ari = train_SSSNET(A_abs_torch_train, data.x, data.edge_index_p, data.edge_weight_p,
                                        data.edge_index_n, data.edge_weight_n, train_index, seed_index, loss_func_pbnc, data.y)
            Val_ari = test_SSSNET(data.x, data.edge_index_p, data.edge_weight_p,
                        data.edge_index_n, data.edge_weight_n, val_index, data.y)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_ARI: {train_ari:.4f}, Val_ARI: {Val_ari:.4f}')
            writer.add_scalar('train_loss_'+str(split), train_loss, epoch)
            writer.add_scalar('train_ARI_'+str(split), train_ari, epoch)
            writer.add_scalar('val_ARI_'+str(split), Val_ari, epoch)
            if Val_ari > best_val_ari:
                best_val_ari = Val_ari
                torch.save(model.state_dict(), best_model_path)
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= args.early_stopping:
                    break
        
        print('With {} out of {} epochs:'.format(epoch + 1, args.epochs))
        model.load_state_dict(torch.load(best_model_path))
        train_ari = test_SSSNET(data.x, data.edge_index_p, data.edge_weight_p,
                        data.edge_index_n, data.edge_weight_n, train_index, data.y)
        val_ari = test_SSSNET(data.x, data.edge_index_p, data.edge_weight_p,
                        data.edge_index_n, data.edge_weight_n, val_index, data.y)
        test_ari = test_SSSNET(data.x, data.edge_index_p, data.edge_weight_p,
                        data.edge_index_n, data.edge_weight_n, test_index, data.y)
        print(f'Split: {split:02d}, Train_ARI: {train_ari:.4f}, Val_ARI: {val_ari:.4f}, Test_ARI: {test_ari:.4f}')
        res_array[split] = [train_ari, val_ari, test_ari]
    elif args.method == 'MSGNN':
        for epoch in range(args.epochs):
            train_loss, train_ari = train(A_abs_torch_train, data.x, data.edge_index, data.edge_weight, train_index, seed_index, loss_func_pbnc, data.y)
            Val_ari = test(data.x, data.edge_index, data.edge_weight, val_index, data.y)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_ARI: {train_ari:.4f}, Val_ARI: {Val_ari:.4f}')
            writer.add_scalar('train_loss_'+str(split), train_loss, epoch)
            writer.add_scalar('train_ARI_'+str(split), train_ari, epoch)
            writer.add_scalar('val_ARI_'+str(split), Val_ari, epoch)
            if Val_ari > best_val_ari:
                best_val_ari = Val_ari
                torch.save(model.state_dict(), best_model_path)
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= args.early_stopping:
                    break
        print('With {} out of {} epochs:'.format(epoch + 1, args.epochs))
        model.load_state_dict(torch.load(best_model_path))
        train_ari = test(data.x, data.edge_index, data.edge_weight, train_index, data.y)
        val_ari = test(data.x, data.edge_index, data.edge_weight, val_index, data.y)
        test_ari = test(data.x, data.edge_index, data.edge_weight, test_index, data.y)
        print(f'Split: {split:02d}, Train_ARI: {train_ari:.4f}, Val_ARI: {val_ari:.4f}, Test_ARI: {test_ari:.4f}')
        res_array[split] = [train_ari, val_ari, test_ari]
    elif args.method == 'SigMaNet':
        for epoch in range(args.epochs):
            train_loss, train_ari = train_SigMaNet(data.x, train_index, seed_index, loss_func_pbnc, data.y)
            Val_ari = test_SigMaNet(data.x, val_index, data.y)
            print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_ARI: {train_ari:.4f}, Val_ARI: {Val_ari:.4f}')
            writer.add_scalar('train_loss_'+str(split), train_loss, epoch)
            writer.add_scalar('train_ARI_'+str(split), train_ari, epoch)
            writer.add_scalar('val_ARI_'+str(split), Val_ari, epoch)
            if Val_ari > best_val_ari:
                best_val_ari = Val_ari
                torch.save(model.state_dict(), best_model_path)
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= args.early_stopping:
                    break
        
        print('With {} out of {} epochs:'.format(epoch + 1, args.epochs))
        model.load_state_dict(torch.load(best_model_path))
        train_ari = test_SigMaNet(data.x, train_index, data.y)
        val_ari = test_SigMaNet(data.x, val_index, data.y)
        test_ari = test_SigMaNet(data.x, test_index, data.y)
        print(f'Split: {split:02d}, Train_ARI: {train_ari:.4f}, Val_ARI: {val_ari:.4f}, Test_ARI: {test_ari:.4f}')
        res_array[split] = [train_ari, val_ari, test_ari]

print("For dataset{}, {} obtains average train, val and test ARIs: {}".format(args.dataset, args.method, res_array.mean(0)))
# save results
if args.debug:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../Finance_results/debug/'+args.dataset)
else:
    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../Finance_results/'+args.dataset)

if os.path.isdir(os.path.join(dir_name, sub_dir_name, args.method)) == False:
    try:
        os.makedirs(os.path.join(dir_name, sub_dir_name, args.method))
    except FileExistsError:
        print('Folder exists for {}!'.format(sub_dir_name, args.method))

np.save(os.path.join(dir_name, sub_dir_name, args.method, suffix), res_array)