'''
SigMaNet architecture
'''

import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv import MessagePassing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import laplacian

class complex_relu_layer(nn.Module):
    """The complex ReLU layer from the `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    """
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real:torch.FloatTensor, img:torch.FloatTensor):
        """
        Complex ReLU function.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real:torch.FloatTensor, img:torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, img = self.complex_relu(real, img)
        return real, img



class SigMaNetConv(MessagePassing):
    

    def __init__(self, in_channels:int, out_channels:int, K:int, i_complex:bool=False, follow_math:bool=True, gcn:bool=False, net_flow:bool=True,
                 normalization:str='sym', bias:bool=True, edge_index=None, norm_real=None, norm_imag=None,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SigMaNetConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        kwargs.setdefault('flow', 'target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if gcn: # devo eliminare i pesi creati per moltiplicarli con il self-loop e creo solo un peso nel caso Theta moltiplica tutto [(I + A)\Theta]
            K = 1
            self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        elif not gcn and follow_math:
            self.weight = Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        else:
            self.weight = Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        self.gcn = gcn
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.i_complex = i_complex
        self.follow_math = follow_math
        self.net_flow = net_flow

        #Inserisco qui i valori di edge index, norm_real e norm_imagla creazione i valori come self
        self.edge_index=edge_index
        self.norm_real = norm_real
        self.norm_imag = norm_imag

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    
  # Possiamo utilizzare  questa funzione per elaborare la parte Tx0=I
    def process(self, mul_L_real, mul_L_imag, weight, X_real, X_imag):
        #data = torch.spmm(mul_L_real, X_real) sparse matrix
        Tx_0_real_real = torch.spmm(mul_L_real, X_real)
        real_real = torch.matmul(Tx_0_real_real, weight) 
        Tx_0_imag_imag = torch.matmul(mul_L_imag, X_imag)
        imag_imag = torch.matmul(Tx_0_imag_imag, weight) 

        Tx_0_real_imag = torch.matmul(mul_L_imag, X_real) # L_imag e x_reale --> real_imag
        real_imag = torch.matmul(Tx_0_real_imag, weight)
        Tx_0_imag_real = torch.matmul(mul_L_real, X_imag) # L_real e x_imag --> imag_real
        imag_real = torch.matmul(Tx_0_imag_real, weight)
        return real_real,Tx_0_real_real, imag_imag, Tx_0_imag_imag, imag_real, Tx_0_imag_real, real_imag, Tx_0_real_imag #torch.stack([real, imag])

    def forward(
        self,
        x_real: torch.FloatTensor, 
        x_imag: torch.FloatTensor, 
        #edge_index: torch.LongTensor,
        #edge_weight: OptTensor = None,
        #lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the SigMaNet Convolution layer.
        
        Arg types:
            * x_real, x_imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long TensSor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * out_real, out_imag (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (N_nodes, F_out).
        """
        
        self.n_dim = x_real.shape[0]

        norm_imag = self.norm_imag
        norm_real = self.norm_real
        edge_index = self.edge_index


        if self.follow_math:
            norm_imag = - norm_imag
            norm_real = - norm_real


        if not self.gcn:
            if self.follow_math:

                if self.i_complex:
                   i_real = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
                   i_imag = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
                else:
                   i_real = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
                   i_imag = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.zeros(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)

                out_real_real, Tx_0_real_real, out_imag_imag, Tx_0_imag_imag, \
                out_imag_real, Tx_0_imag_real, out_real_imag, Tx_0_real_imag = self.process(i_real, i_imag, self.weight[0], x_real, x_imag)
           
      

                if self.weight.size(0) > 1:
                    Tx_1_real_real = self.propagate(edge_index, x=x_real, norm=norm_real, size=None).to(torch.float) # x_real - norm_real
                    #print("Tx_1_real_real", Tx_1_real_real)
                    #out_real_real = out_real_real + torch.matmul(Tx_1_real_real, self.weight[0])
                    out_real_real = out_real_real + torch.matmul(Tx_1_real_real, self.weight[1])
                    #print("output_real_real", out_real_real)
                    Tx_1_imag_imag = self.propagate(edge_index, x=x_imag, norm=norm_imag, size=None).to(torch.float) # x_imag - norm_imag
                    #print("Tx_1_imag_imag", Tx_1_imag_imag)
                    out_imag_imag = out_imag_imag + torch.matmul(Tx_1_imag_imag, self.weight[1])
                    #print("output_imag_imag", out_imag_imag)
                    Tx_1_imag_real = self.propagate(edge_index, x=x_imag, norm=norm_real, size=None).to(torch.float) # x_imag - norm_real
                    out_imag_real = out_imag_real + torch.matmul(Tx_1_imag_real, self.weight[1])
                    Tx_1_real_imag = self.propagate(edge_index, x=x_real, norm=norm_imag, size=None).to(torch.float) # x_real - norm_imag
                    out_real_imag = out_real_imag + torch.matmul(Tx_1_real_imag, self.weight[1])
            

                #for k in range(1, self.weight.size(0)): # Polinomio di Cheb (Corretto!)
                for k in range(2, self.weight.size(0)): # Polinomio di Cheb (Corretto!)
                    Tx_2_real_real = self.propagate(edge_index, x=Tx_1_real_real, norm=norm_real, size=None) # x_real - norm_real
                    Tx_2_real_real = 2. * Tx_2_real_real - Tx_0_real_real
                    out_real_real = out_real_real + torch.matmul(Tx_2_real_real, self.weight[k])
                    Tx_0_real_real, Tx_1_real_real = Tx_1_real_real, Tx_2_real_real

                    Tx_2_imag_imag = self.propagate(edge_index, x=Tx_1_imag_imag, norm=norm_imag, size=None) # x_imag - norm_imag
                    Tx_2_imag_imag = 2. * Tx_2_imag_imag - Tx_0_imag_imag
                    out_imag_imag = out_imag_imag + torch.matmul(Tx_2_imag_imag, self.weight[k])
                    Tx_0_imag_imag, Tx_1_imag_imag = Tx_1_imag_imag, Tx_2_imag_imag

                    Tx_2_imag_real = self.propagate(edge_index, x=Tx_1_imag_real, norm=norm_real, size=None) # x_imag - norm_real
                    Tx_2_imag_real = 2. * Tx_2_imag_real - Tx_0_imag_real
                    out_imag_real = out_imag_real + torch.matmul(Tx_2_imag_real, self.weight[k])
                    Tx_0_imag_real, Tx_1_imag_real = Tx_1_imag_real, Tx_2_imag_real

                    Tx_2_real_imag = self.propagate(edge_index, x=Tx_1_real_imag, norm=norm_imag, size=None) # x_real - norm_imag
                    Tx_2_real_imag = 2. * Tx_2_real_imag - Tx_0_real_imag
                    out_real_imag = out_real_imag + torch.matmul(Tx_2_real_imag, self.weight[k])
                    Tx_0_real_imag, Tx_1_real_imag = Tx_1_real_imag, Tx_2_real_imag

            else:

                if self.i_complex:
                   i_real = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
                   i_imag = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
                else:
                   i_real = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
                   i_imag = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.zeros(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)

                out_real_real, Tx_0_real_real, out_imag_imag, Tx_0_imag_imag, \
                out_imag_real, Tx_0_imag_real, out_real_imag, Tx_0_real_imag = self.process(i_real, i_imag, self.weight[0], x_real, x_imag)

            
      

                # Nuovo codice con i cambi opportuni
                if self.weight.size(0) > 1:
                    Tx_1_real_real = self.propagate(edge_index, x=x_real, norm=norm_real, size=None).to(torch.float) # x_real - norm_real
                    #print("Tx_1_real_real", Tx_1_real_real)
                    out_real_real = out_real_real + torch.matmul(Tx_1_real_real, self.weight[1])
                    #print("output_real_real", out_real_real)
                    Tx_1_imag_imag = self.propagate(edge_index, x=x_imag, norm=norm_imag, size=None).to(torch.float) # x_imag - norm_imag
                    #print("Tx_1_imag_imag", Tx_1_imag_imag)
                    out_imag_imag = out_imag_imag + torch.matmul(Tx_1_imag_imag, self.weight[1])
                    #print("output_imag_imag", out_imag_imag)
                    Tx_1_imag_real = self.propagate(edge_index, x=x_imag, norm=norm_real, size=None).to(torch.float) # x_imag - norm_real
                    out_imag_real = out_imag_real + torch.matmul(Tx_1_imag_real, self.weight[1])
                    Tx_1_real_imag = self.propagate(edge_index, x=x_real, norm=norm_imag, size=None).to(torch.float) # x_real - norm_imag
                    out_real_imag = out_real_imag + torch.matmul(Tx_1_real_imag, self.weight[1])
            


                for k in range(2, self.weight.size(0)): # Polinomio di Cheb (Corretto!)
                    Tx_2_real_real = self.propagate(edge_index, x=Tx_1_real_real, norm=norm_real, size=None) # x_real - norm_real
                    Tx_2_real_real = 2. * Tx_2_real_real - Tx_0_real_real
                    out_real_real = out_real_real + torch.matmul(Tx_2_real_real, self.weight[k])
                    Tx_0_real_real, Tx_1_real_real = Tx_1_real_real, Tx_2_real_real

                    Tx_2_imag_imag = self.propagate(edge_index, x=Tx_1_imag_imag, norm=norm_imag, size=None) # x_imag - norm_imag
                    Tx_2_imag_imag = 2. * Tx_2_imag_imag - Tx_0_imag_imag
                    out_imag_imag = out_imag_imag + torch.matmul(Tx_2_imag_imag, self.weight[k])
                    Tx_0_imag_imag, Tx_1_imag_imag = Tx_1_imag_imag, Tx_2_imag_imag

                    Tx_2_imag_real = self.propagate(edge_index, x=Tx_1_imag_real, norm=norm_real, size=None) # x_imag - norm_real
                    Tx_2_imag_real = 2. * Tx_2_imag_real - Tx_0_imag_real
                    out_imag_real = out_imag_real + torch.matmul(Tx_2_imag_real, self.weight[k])
                    Tx_0_imag_real, Tx_1_imag_real = Tx_1_imag_real, Tx_2_imag_real

                    Tx_2_real_imag = self.propagate(edge_index, x=Tx_1_real_imag, norm=norm_imag, size=None) # x_real - norm_imag
                    Tx_2_real_imag = 2. * Tx_2_real_imag - Tx_0_real_imag
                    out_real_imag = out_real_imag + torch.matmul(Tx_2_real_imag, self.weight[k])
                    Tx_0_real_imag, Tx_1_real_imag = Tx_1_real_imag, Tx_2_real_imag

            out_real = out_real_real - out_imag_imag
            out_imag = out_imag_real + out_real_imag
        
        else:
            # Nuovo codice con i cambi opportuni
            Tx_1_real_real = self.propagate(edge_index, x=x_real, norm=norm_real, size=None).to(torch.float) # x_real - norm_real
            #print("Tx_1_real_real", Tx_1_real_real)
            out_real_real = torch.matmul(Tx_1_real_real, self.weight[0])
            #print("output_real_real", out_real_real)
            Tx_1_imag_imag = self.propagate(edge_index, x=x_imag, norm=norm_imag, size=None).to(torch.float) # x_imag - norm_imag
            #print("Tx_1_imag_imag", Tx_1_imag_imag)
            out_imag_imag = torch.matmul(Tx_1_imag_imag, self.weight[0])
            #print("output_imag_imag", out_imag_imag)
            Tx_1_imag_real = self.propagate(edge_index, x=x_imag, norm=norm_real, size=None).to(torch.float) # x_imag - norm_real
            out_imag_real = torch.matmul(Tx_1_imag_real, self.weight[0])
            Tx_1_real_imag = self.propagate(edge_index, x=x_real, norm=norm_imag, size=None).to(torch.float) # x_real - norm_imag
            out_real_imag = torch.matmul(Tx_1_real_imag, self.weight[0])

            out_real = out_real_real - out_imag_imag
            out_imag = out_imag_real + out_real_imag        


        if self.bias is not None:
            out_real += self.bias
            out_imag += self.bias

        return out_real, out_imag


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

class SigMaNet_link_prediction_one_laplacian(nn.Module):
    r"""The SigManet model for link prediction from the    
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MagNetConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
    """
    def __init__(self, num_features:int, hidden:int=2, K:int=2, label_dim:int=2, \
        activation:bool=True, layer:int=2, dropout:float=0.5, normalization:str='sym',\
        i_complex:bool=True, follow_math:bool=False,gcn:bool=False, net_flow:bool=True, unwind:bool=False, 
        edge_index=None, norm_real=None, norm_imag=None):
        super(SigMaNet_link_prediction_one_laplacian, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(SigMaNetConv(in_channels=num_features, out_channels=hidden, K=K,\
                                  i_complex=i_complex, follow_math=follow_math,\
            gcn=gcn, net_flow=net_flow, normalization=normalization, edge_index=edge_index,\
            norm_real=norm_real, norm_imag=norm_imag))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(SigMaNetConv(in_channels=hidden, out_channels=hidden, K=K, \
            i_complex=i_complex, follow_math=follow_math,\
            gcn=gcn, net_flow=net_flow, normalization=normalization, \
            edge_index=edge_index, norm_real=norm_real, norm_imag=norm_imag))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden*4, label_dim)      
        self.dropout = dropout
        #self.complex_dropout = drop.Dropout(self.dropout)
        #self.complex_softmax = act.modLogSoftmax(dim=1)
        self.unwind = unwind

        # nel caso di no unwind --> l'ultima operazione deve darmi come output il numero di classi da prevedere
        if not self.unwind:
            print('no unwind!!!')
            chebs[-1] = SigMaNetConv(in_channels=hidden, out_channels=label_dim, K=K, \
            i_complex=i_complex, follow_math=follow_math,\
            gcn=gcn, net_flow=net_flow, normalization=normalization)
            #chebs.append(SignumConv(in_channels=hidden, out_channels=label_dim, K=K, \
            #i_complex=i_complex, follow_math=follow_math,\
            #gcn=gcn, net_flow=net_flow, normalization=normalization))

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, \
        query_edges: torch.LongTensor) -> torch.FloatTensor:
        """
      
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
        if not self.unwind:
            real = real[query_edges[:,0]] + real[query_edges[:,1]]
            imag = imag[query_edges[:,0]] + imag[query_edges[:,1]]
            x = real + 1j* imag
        # Unwind operation
        else:
            x = torch.cat((real[query_edges[:,0]], real[query_edges[:,1]], imag[query_edges[:,0]], imag[query_edges[:,1]]), dim = -1)
            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)
            x = self.linear(x)
            x = F.log_softmax(x, dim=1)
        return x


class SigMaNet_node_prediction_one_laplacian(nn.Module):
    r"""The SigMaNet model for node classification 
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MagNetConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
    """
    def __init__(self, num_features:int, hidden:int=2, K:int=1, label_dim:int=2, \
        activation:bool=True, layer:int=2, dropout:float=0.5, normalization:str='sym',\
        i_complex:bool=True, follow_math:bool=False,gcn:bool=False, net_flow:bool=True, unwind:bool=False,
        edge_index=None, norm_real=None, norm_imag=None):
        super(SigMaNet_node_prediction_one_laplacian, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(SigMaNetConv(in_channels=num_features, out_channels=hidden, K=K,\
                                  i_complex=i_complex, follow_math=follow_math,\
            gcn=gcn, net_flow=net_flow, normalization=normalization, edge_index=edge_index,\
            norm_real=norm_real, norm_imag=norm_imag))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(SigMaNetConv(in_channels=hidden, out_channels=hidden, K=K, \
            i_complex=i_complex, follow_math=follow_math,\
            gcn=gcn, net_flow=net_flow, normalization=normalization, \
            edge_index=edge_index, norm_real=norm_real, norm_imag=norm_imag))

        self.Chebs = chebs
        last_dim = 2
        self.Conv = nn.Conv1d(hidden*last_dim, label_dim, kernel_size=1)
        self.dropout = dropout
        self.unwind = unwind

        # nel caso di no unwind --> l'ultima operazione deve darmi come output il numero di classi da prevedere
        if not self.unwind:
            print('no unwind!!!')
            chebs[-1] = SigMaNetConv(in_channels=hidden, out_channels=label_dim, K=K, \
            i_complex=i_complex, follow_math=follow_math,\
            gcn=gcn, net_flow=net_flow, normalization=normalization)
            #chebs.append(SignumConv(in_channels=hidden, out_channels=label_dim, K=K, \
            #i_complex=i_complex, follow_math=follow_math,\
            #gcn=gcn, net_flow=net_flow, normalization=normalization))

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.Conv.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor) -> torch.FloatTensor:
        """
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
        if not self.unwind:
            x = real + 1j* imag
            #if self.dropout > 0:
            #    x = self.complex_dropout(x)
            #x = self.complex_softmax(x)
        # Unwind operation
        else:
            x = torch.cat((real, imag), dim = -1)
            if self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)
            z = x.clone() # embedding
            x = x.unsqueeze(0)
            x = x.permute((0,2,1))
            x = self.Conv(x)
            x = x.permute((0,2,1)).squeeze()
            output = F.log_softmax(x, dim=1)
            predictions_cluster = torch.argmax(output, dim=1)

            prob = F.softmax(output, dim=1)

        return F.normalize(z), output, predictions_cluster, prob