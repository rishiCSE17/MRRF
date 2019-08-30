'''
This code is written as a part of PhD project
Purpose : from a given binary adjacency matrix this code
    1. generates a random matrix sequence of node and link cost.
    2. Normalises node cost to links using STEN ('Energy aware IP Routing in SDN' S.Ghosh Globecom 2018)
    3. Calulates Reliability from past samples
    4. plots the eadgewise reliability

Project is running under : H2020 SONNET
@author Saptarshi Ghosh
All right Reserved : SUITE Lab, London Southbank University, UK
'''
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import random
import time

topo = nx.MultiDiGraph()
regression_vector=[]

def create_topo(graph, adj_matrix):
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] != 0:
                graph.add_edge('node_' + str(i + 1), 'node_' + str(j + 1))
                print(f'node_{i+1} , node_{j+1} : Added')
    return graph


def create_topo_with_weight(g, weighted_matrix):
    for i in range(len(weighted_matrix)):
        for j in range(len(weighted_matrix)):
            if weighted_matrix.item(i, j) != 0:
                val = round(weighted_matrix.item(i, j), 3)
                g.add_edge('node_' + str(i + 1),
                           'node_' + str(j + 1),
                           weight=val,
                           lenghth=val)
                #print(f'node_{i+1} , node_{j+1} : Added')
                '''print(f"g.add_edge('node_{i+1}' , "
                      f"'node_{j+1}', "
                      f"weight={round(val,3)})")'''
    return g


'''
topo=create_topo(graph=topo, adj_matrix=adj)
nx.draw(topo, pos=nx.spring_layout(topo), with_labels=True)
plt.show()
'''


def Radom_matrix_transform(adj):
    '''creating a numpy matrix from list structure'''
    adj_b = np.matrix(adj)
    # print(adj_b)

    '''creating random matrix'''
    rand = np.random.rand(adj_b.shape[0], adj_b.shape[1])
    # pd.DataFrame(rand)

    '''element wise multiplication'''
    adj_e = np.multiply(rand, adj_b)
    # pd.DataFrame(adj_e)

    '''calculating Adj_n'''
    x = np.multiply(np.random.rand(
        adj_b.shape[0], adj_b.shape[1]),
        np.identity(adj_b.shape[0])
    )
    # pd.DataFrame(x)

    adj_n = x + adj_e
    # pd.DataFrame(adj_n)
    return adj_b, adj_n


def get_non_zero_index(adj):
    ret = []
    for i in range(adj.shape[0]):
        row = []
        for j in range(adj.shape[1]):
            if adj.item(i, j) != 0:
                row.append(j)
        ret.append(row)
    return np.matrix(ret)


def distribute_rand_over_nz_row(mat):
    ret = []
    for row in range(mat.shape[1]):
        temp = []
        for i in range(len(mat.item(row))):
            temp.append(random.randint(0, 100))
        r_sum = sum(temp)
        for i in range(len(temp)):
            temp[i] = round((temp[i] / r_sum), 3)
        ret.append(temp)
    return ret


def generate_affinity(nz_indx_mat, nz_rand_mat, adj_n):
    ret = []
    for i in range(adj_n.shape[0]):
        n_util = adj_n.item(i, i)
        row = []
        k = 0
        for j in range(adj_n.shape[1]):
            if j in nz_indx_mat.item(i):
                offset = nz_rand_mat[i][k] * n_util
                row.append(offset)
                k += 1
            else:
                row.append(0)
        row[i] = -n_util
        ret.append(row)
    return ret


def get_adj_s(adj):
    adj_b, adj_n = Radom_matrix_transform(adj)
    nz_indx_mat = get_non_zero_index(adj_b)
    nz_rand_mat = distribute_rand_over_nz_row(get_non_zero_index(adj_b))
    affifity_mat = generate_affinity(nz_indx_mat, nz_rand_mat, adj_n)
    adj_s = affifity_mat + adj_n

    # prints the STEn converted matrix
    print('STEN Converted....')
    print(adj_s)
    return adj_s


'''returns the weights of edges'''


def get_edge_weight(adj_s):
    mat = np.matrix(adj_s)
    arr = mat.reshape(mat.shape[0] * mat.shape[1])
    indx = np.where(arr != 0)[1]
    vect = np.take(arr, indx)
    return np.matrix.tolist(vect)


'''calculates the lement wise median ans SD of a given matrix'''


def get_median_sd(sampled_matrix):
    import statistics as stat
    median = []
    sd = []

    for node_history in sampled_matrix:
        median.append(stat.median(node_history))
        sd.append(stat.stdev(node_history))
        print('mu :', median)
        print('sd :', sd)
    return (median, sd)


def get_Quality(s_median, s_sd, w1=0.5, w2=0.5):
    q = []
    for i in range(len(s_median)):
        q.append(w1 * s_median[i] + w2 * s_sd[i])
    return q


def get_Reliability(quality_vct):
    R = []
    for q in quality_vct:
        R.append((q - min(quality_vct)) / (max(quality_vct) - min(quality_vct)))

    return R


def calculate_reliabiliy(sample_space, adj_s):
    edge_sample = get_edge_weight(adj_s)
    # print(edge_sample[0])
    r_vector = [0] * len(edge_sample[0])
    t_series_edge = [[0]]* len(edge_sample[0])

    if len(sample_space) == 10:
        sample_space.pop(0)
    sample_space.append(edge_sample[0])
    # print(pd.DataFrame(sample_space).shape)

    temp_mat = np.matrix(sample_space)
    temp_mat1 = np.transpose(temp_mat)
    sample_space = np.matrix.tolist(temp_mat1)
    # print(pd.DataFrame(sample_space))
    # print(len(sample_space[0]))

    #time.sleep(1)

    if (len(sample_space[0]) >= 2):
        sample_median, sample_sd = get_median_sd(sample_space)
        q_vector = get_Quality(sample_median, sample_sd, w1=0.4, w2=0.6)
        r_vector = get_Reliability(q_vector)
        r_vector_rounded = [round(x,3) for x in r_vector] # rounding-off values of r_vec to 3 decimal pl.
        #print(f'r_vector : {r_vector_rounded}')
        t_series_edge = populate_time_series(r_vector_rounded, window=100)

    return r_vector, t_series_edge

    # return None

# polulate a time series from r_vector_rounded
def populate_time_series(r_vector, window):
    if len(regression_vector) > window:
        regression_vector.pop(0)
    regression_vector.append(r_vector)
    return np.transpose(np.matrix(regression_vector)).tolist() #convers list time series


def loop(adj, topo_s):
    weighted_sample_space = []

    ts=16
    fig_t_series = plt.figure('Time Series')
    ax_ts_list = []
    for i in range(ts):
        ax_ts_list.append(fig_t_series.add_subplot(ts, 1, i + 1))

    while True:
        adj_s = get_adj_s(adj)
        # print(pd.DataFrame(adj_s))
        topo_s.clear()
        topo_s = create_topo_with_weight(topo_s, adj_s)
        plt.ion()

        fig_topo = plt.figure('Topology')
        fig_topo = plt.title('Topology')
        fig_topo = plt.ion()
        fig_topo = plt.pause(2)
        fig_topo = plt.clf()
        nx.draw(topo_s, pos=nx.spectral_layout(topo_s), with_labels=True)

        fig_reliab = plt.figure('Reliability')
        fig_reliab = plt.title('Relaibility')
        fig_reliab = plt.ion()
        fig_reliab = plt.pause(2)
        fig_reliab = plt.clf()

        r_vector, t_series_edge = calculate_reliabiliy(weighted_sample_space, adj_s)


        plt.bar(np.arange(len(topo_s.edges)),
                r_vector,
                label='reliability',
                alpha=1)

        # fig_topo.show()
        # fig_reliab.show()
        plt.grid('True')

        # plot for  time series

        fig_t_series = plt.title('Tesries')


        fig_t_series = plt.ion()
        fig_t_series = plt.pause(2)
        fig_t_series = plt.clf()

        for i in range(len(t_series_edge)):
            ax_ts_list[i].plot(np.arange(len(t_series_edge[i])),t_series_edge[i],'b:o',label=f'edge_{i}')
            ax_ts_list[i].set_axis=False
            ax_ts_list[i].grid(True)
            #ax_ts_list[i].legend()

        plt.show()


def main():
    print('Adjacency Matrix Initiaged... ')
    adj = [[0, 1, 1, 0, 0, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 0, 0, 1, 1, 0],
           [0, 0, 1, 0, 1, 1],
           [0, 0, 1, 1, 0, 0],
           [1, 1, 0, 1, 0, 0]]
    print('[1/3] Binary Adjacency Matrix Initiaged... ')
    time.sleep(1)
    print(adj)
    topo_s = nx.MultiDiGraph()
    print('[2/3] Topology Graph Initiated...')
    time.sleep(1)
    print('[3/3] Initialising... Entering indefinite loop, Close to Terminate...')
    time.sleep(1)
    loop(adj, topo_s)


main()
