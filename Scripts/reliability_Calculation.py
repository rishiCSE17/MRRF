import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import random

def get_reliability(adj):
    pass

def create_topo(adj_matrix):
    graph = nx.MultiDiGraph()

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix.item(i,j) != 0:
                graph.add_edge('node_' + str(i + 1), 'node_' + str(j + 1))
            print(f'node_{i + 1} , node_{j + 1} : Added')
    return graph


def create_topo_with_weight(weighted_matrix):
    g=nx.Graph()
    for i in range(len(weighted_matrix)):
        for j in range(len(weighted_matrix)):
            if weighted_matrix.item(i, j) != 0:
                val = round(weighted_matrix.item(i, j), 3)
                g.add_edge('node_' + str(i + 1),
                           'node_' + str(j + 1),
                           weight=val,
                           lenghth=val)

                #print(f"g.add_edge('node_{i + 1}' , 'node_{j + 1}', weight={round(val, 3)})")
    return g

def show_graph(graph):
    nx.draw_networkx(graph, pos=nx.spring_layout(graph), with_labels=True)
    plt.axis('off')
    plt.show()

def put_adj():
    m = [[0, 1, 1, 0, 0, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 0, 0, 1, 1, 0],
           [0, 0, 1, 0, 1, 1],
           [0, 0, 1, 1, 0, 0],
           [1, 1, 0, 1, 0, 0]]
    return np.matrix(m)

'''
----------- section generates random matrix-----------------------
'''

def get_random_mat(adj_b):
    #create a random matrix of identical order to adj_b
    rand = np.random.rand(adj_b.shape[0], adj_b.shape[1])

    #element wise multiplicartio
    adj_e = np.multiply(rand, adj_b)

    #calculating adj_n
    adj_n = np.multiply(np.random.rand(adj_b.shape[0], adj_b.shape[1]),
                    np.identity(adj_b.shape[0])) + adj_e
    return adj_n

'''
------------- ransom matrix gen ends here --------------------------
'''

'''
------------------ STEN Starts here --------------------------------
'''
def get_non_zero_index(adj):
    ret=[]
    for i in range(adj.shape[0]):
        row=[]
        for j in range(adj.shape[1]):
            if adj.item(i,j)!=0 and i !=j :
                row.append(j)
        ret.append(row)
    return np.matrix(ret)

def distribute_rand_over_nz_row(mat):
    ret = []
    for row in range(mat.shape[1]):
        temp=[]
        for i in range(len(mat.item(row))):
            temp.append(random.randint(0,100))
        r_sum=sum(temp)
        for i in range(len(temp)):
            temp[i] = round((temp[i] / r_sum), 3)
        ret.append(temp)
    return ret

def generate_affinity(nz_indx_mat , nz_rand_mat, adj_n):
    ret=[]
    for i in range(adj_n.shape[0]):
        n_util = adj_n.item(i,i)
        row=[]
        k=0
        for j in range(adj_n.shape[1]):
            if j in nz_indx_mat.item(i):
                offset = nz_rand_mat[i][k] * n_util
                row.append(offset)
                k+=1
            else:
                row.append(0)
        row[i]=-n_util
        ret.append(row)
    return ret

def sten(adj_n):
    nz_indx_mat = get_non_zero_index(adj_n)
    nz_rand_mat = distribute_rand_over_nz_row(get_non_zero_index(adj_n))
    affifity_mat = generate_affinity(nz_indx_mat, nz_rand_mat, adj_n)
    adj_s = affifity_mat + adj_n
    return adj_s

'''
---------------------- STEN ends here -------------------------------------
'''

def main():
    adj=put_adj() # get adjacencey matrix
    #graph=create_topo(adj) #get the graph structure
    #show_graph(create_topo(adj)) #plot the graph
    adj_n=get_random_mat(adj) #get a matrix with random node & link cost
    adj_s=sten(adj_n)   # normalising with sten
    print(adj_s)
    show_graph(create_topo_with_weight(adj_n))



main()