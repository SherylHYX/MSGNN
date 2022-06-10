'''
Code for net flow application
'''



import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, triu
import scipy

# Creation of subdivision edges
def add_edges(G, n_rows, biconnections, dictionary, i):
    G.add_weighted_edges_from([(biconnections[i][0],  biconnections[i][1], dictionary[biconnections[i]]),
                               ( biconnections[i][1], biconnections[i][0], dictionary[biconnections[i+1]])])
    return G

# identification of antiparallel edges
def have_bidirectional_relationship(G, node1, node2):
    return G.has_edge(node1, node2) and G.has_edge(node2, node1)

# list of antiparalell edges
def biconnection(graph):
    biconnections= []
    for u, v in graph.edges():
        if u != v: # Avoid self_loop
            if u > v:  # Avoid duplicates, such as (1, 2) and (2, 1)
                #v, u = u, v
                continue
            if have_bidirectional_relationship(graph, u, v):
                biconnections.append((u,v))
                biconnections.append((v,u))
    return biconnections

# Creation of a dictionary with initial node and weight indication
def dictionary_connection(graph):
    dictionary = {(node1,node2) : data['weight'] for node1, node2, data in graph.edges(data=True)}
    return dictionary

# Creation of the subdivision graph
def flipping(graph):
    graph_1 = nx.from_scipy_sparse_matrix(graph, create_using=nx.DiGraph)
    graph_bidirectional = nx.from_scipy_sparse_matrix(coo_matrix((graph_1.number_of_nodes(), graph_1.number_of_nodes()), 
                                               dtype=np.int8), create_using=nx.DiGraph)
    dictionary = dictionary_connection(graph_1)
    biconnections = biconnection(graph_1)
    A = nx.to_scipy_sparse_matrix(graph_1)
    n_rows = A.shape[0]
    lista = [add_edges(graph_bidirectional, n_rows, biconnections,dictionary, i) for i in range(0, len(biconnections),2)]
    return nx.to_scipy_sparse_matrix(graph_bidirectional)
    
def new_adj(A):
    AB = flipping(A)
    AU = triu(AB - AB.T)
    return A - AB + AU