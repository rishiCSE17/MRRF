'''
    opendaylight - python integrator master file
    author : Saptarshi Ghosh
    features :
        1. fetaches topology from ODL (API : network-topology | GET)
        2. fetches ovs flow information (API : inverntory | GET)
        3. writes mac forwaridding information to MySQL DB

'''
import pprint
import requests
import json
import matplotlib.pyplot as plt
import networkx as nx
import sqlite3
import pymysql

''' -------------------------------------------------------------------------------------------------
    class definitions
    -------------------------------------------------------------------------------------------------'''


class host:
    def __init__(self, node_id, h_id, mac, ip):
        self.node_id = node_id
        self.h_id = h_id
        self.mac = mac
        self.ip = ip
        self.host_data = [self.node_id, self.h_id, self.mac, self.ip]

    def get_host(self):
        return self.host_data


class flow_entry:
    def __init__(self, source="", destination="", out_port=0, prio=0, h_to=0, p_count=0, b_count=0, duration=0):
        self.source = source
        self.destination = destination
        self.out_port = out_port
        self.prio = prio
        self.h_to = h_to
        self.p_count = p_count
        self.b_count = b_count
        self.duration = duration

    def get_flow(self):
        return [self.source, self.destination, self.out_port, self.prio, self.h_to, self.p_count, self.b_count,
                self.duration]


''' -------------------------------------------------------------------------------------------------
    list definitions
    -------------------------------------------------------------------------------------------------'''

host_list = []  # list of host object
h_list = []  # list of host names
node_list = []  # list of nodes (host + switches)
switch_list = []  # list of switches
edge_list = []  # list of edges format : (node1,node2)
adj_mat = [[]]  # adjacency matrix of the graph
switch_list = []
flow_tab_list = []

'''------------------------------------------------------------------------------------------------
   Shortest path algo 
   -----------------------------------------------------------------------------------------------'''
import networkx as nx
import matplotlib.pyplot as plt_spa
import math
import drawnow
import random
from matplotlib import style

# style.use('fivethirtyeight')

fig_spa = plt_spa.figure()
route_list = []
g_list = []


def gen_random_weight(edge_count):
    ret_weigt_list = []
    for i in range(0, edge_count):
        ret_weigt_list.append(random.randrange(5, edge_count))
    return ret_weigt_list


def create_graph(adj):
    g = nx.Graph()
    g.add_weighted_edges_from(adj)
    return g


# nx.draw(g, nodcolor='r',edgecolor='b')
# plt.show()

def get_node_seq(lst_rt):
    ret = []
    for i in range(0, len(lst_rt) - 1):
        ret.append((lst_rt[i], lst_rt[i + 1]))
    return ret


def gen_route_graphs(lst_edge):
    g_temp = nx.Graph()
    g_temp.add_edges_from(lst_edge)
    return g_temp


def plot_all_sp(g):
    pos = nx.spring_layout(g)  # positions for all nodes

    for node_s in g.nodes:
        for node_d in g.nodes:
            if node_s < node_d and node_s != node_d:
                print('R : ' + str(node_s) + ' <--> ' + str(node_d))
                n_lst = nx.dijkstra_path(g, node_s, node_d)
                e_lst = get_node_seq(n_lst)
                route_list.append(gen_route_graphs(e_lst))
                print('\t : ', n_lst)
                print('\t : ', e_lst)

    axes = math.sqrt(len(route_list))
    sp_list = []
    k = 1
    for g in route_list:
        sp_spa = fig_spa.add_subplot(axes + 1, axes + 1, k)
        nx.draw_networkx(g,
                         pos=nx.spring_layout(g),
                         edge_labes=nx.get_edge_attributes(g, 'weight'),
                         with_labels=True,
                         font_size=12,
                         font_color='black',
                         style='dashed',
                         axes='off',
                         edge_color='r',
                         node_color='g'
                         )
        plt.axis('off')
        plt_spa.plot()

        sp_list.append(sp_spa)
        k = k + 1

    plt_spa.suptitle("All pair Dijkstrs's algo", )
    plt_spa.show()


def gen_weighted_list(edge_list, node_list):
    w_list = gen_random_weight(len(edge_list))
    index = 0
    adj = []
    for i in edge_list:
        print(i[0], i[1], w_list[index])
        # dj.append(i,j,w_list[index])
        index += 1
    '''            
    adj = [(1, 2, w_list[0]), (1, 3, w_list[1]), (1, 4, w_list[2]),
           (2, 3, w_list[3]),
           (3, 5, w_list[4]),
           (4, 5, w_list[5])]
    '''
    return adj


def spa_main(g):
    global w_list
    global fig_spa

    w_list = []
    fig_spa.clear()
    g_list.append(nx.Graph)
    adj = gen_weighted_list(g.edges, g.nodes)
    print('creating graph...')
    # g=create_graph(adj)
    print('Running Shortest path algo...')
    plot_all_sp(g)


'''-------------------------------------------------------------------------------------------------'''

''' -------------------------------------------------------------------------------------------------
    mysql database integration
    -------------------------------------------------------------------------------------------------'''

host_ip = input(r'Enter MySQL Server IP : ')
un = input('user name : ')
pw = input(r'Enter Password')
db = input(r'Enter Schema name')

db = pymysql.connect(host=host_ip, user=un, passwd=pw, db=db)


def create_tab():
    db.query(
        "CREATE TABLE IF NOT EXISTS flow_table (switch varchar(45), \
                                                source varchar(45), \
                                                destination varchar(45), \
                                                output_port REAL, \
                                                priority REAL, \
                                                hard_timeout REAL, \
                                                packet_count REAL, \
                                                byte_count REAL, \
                                                duration REAL)"
    )

    db.query("commit")


def clear_db():
    db.query("delete from flow_table")
    db.query("commit")


def insert_data(sw, flow_entry):
    src = flow_entry[0]
    dst = flow_entry[1]
    o_port = flow_entry[2]
    prio = flow_entry[3]
    h_to = flow_entry[4]
    p_cnt = flow_entry[5]
    b_cnt = flow_entry[6]
    dur = flow_entry[7]

    '''
    print("INSERT INTO flow_table VALUES('" + sw + "','" + src + "','" + dst + "'," \
                                                  + str(o_port) + "," + str(prio) + "," \
                                                  + str(h_to) + "," + str(p_cnt) +"," \
                                                  + str(b_cnt) + "," + str(dur) + ")" )
    '''
    db.query("INSERT INTO flow_table VALUES('" + sw + "','" + src + "','" + dst + "'," \
             + str(o_port) + "," + str(prio) + "," \
             + str(h_to) + "," + str(p_cnt) + "," \
             + str(b_cnt) + "," + str(dur) + ")")
    db.query("commit")


# init_db('shellmon')
# create_tab()

''' -------------------------------------------------------------------------------------------------
    ODL integration : topology fetch 
    -------------------------------------------------------------------------------------------------'''


# populates the lists from the RESTfull api content processing
def generate_lists(cont_ip):
    url = 'http://' + cont_ip + ':8181/restconf/operational/network-topology:network-topology/topology/flow:1'
    get_topo = requests.get(url, auth=('admin', 'admin'))
    get_topo_json = get_topo.json()
    elem = get_topo_json['topology'][0]

    for x in elem['node']:
        node_id = str(x['node-id'])
        if (node_id.__contains__('host')):
            # print(node_id)
            temp = x['host-tracker-service:addresses'][0]
            # print('\t id:',temp['id'], ' mac:',temp['mac'], ' ip: ',temp['ip'])
            host_list.append(host(node_id, temp['id'], temp['mac'], temp['ip']))
            node_list.append(node_id)
            h_list.append(node_id)

        if (node_id.__contains__('openflow')):
            # print(node_id)
            switch_list.append(node_id)
            node_list.append(node_id)

    for y in elem['link']:
        # print(y['link-id'])
        # print('\t',y['source']['source-node'],' ---> ',y['destination']['dest-node'])
        edge_list.append((y['source']['source-node'], y['destination']['dest-node']))


'''----------------------------------------------------------------------------------------------------------'''


def show_discovery_message():
    print('hosts discovered...')
    for host in host_list:
        print(host.get_host())

    print('switches discovered...')
    for switch in switch_list:
        print(switch)

    print('links discovered...')
    for link in edge_list:
        print(link)

    print('nodes discovered...')
    for node in node_list:
        print(node)


'''-------------------------------------------------------------------------------------------------------------'''


def build_adj_mat():
    print('from build_adj_mat()')


'''-------------------------------------------------------------------------------------------------------------'''


def generate_graph():
    g = nx.Graph()
    g.add_nodes_from(h_list, color='green')
    g.add_nodes_from(switch_list, color='blue')
    g.add_edges_from(edge_list, color='green')

    nx.draw_networkx(g, with_labels=True, font_size=10, font_color='blue', style='dashed')
    # plt.axes('off')
    'plot _control'
    # plt.show()
    return g


'''--------------------------------------------------------------------------------------------------------------'''

''' -------------------------------------------------------------------------------------------------
    ODL integration : flow tables fetch 
    -------------------------------------------------------------------------------------------------'''


def generate_flow_tab(switch, cont_ip):
    url = 'http://' + cont_ip + ':8181/restconf/operational/opendaylight-inventory:nodes/node/' + switch + '/flow-node-inventory:table/0'
    get_flow_tab = requests.get(url, auth=('admin', 'admin'))
    get_flow_tab_json = get_flow_tab.json()

    temp_flow_tab = []

    for flow in get_flow_tab_json['flow-node-inventory:table'][0]['flow']:
        if 'ethernet-match' in flow['match'] and 'ethernet-source' in flow['match']['ethernet-match']:
            s_addr = flow['match']['ethernet-match']['ethernet-source']['address']
            d_addr = flow['match']['ethernet-match']['ethernet-destination']['address']
            out_port = flow['instructions']['instruction'][0]['apply-actions']['action'][0]['output-action'][
                'output-node-connector']
            prio = flow['priority']
            hto = flow['hard-timeout']
            p_count = flow['opendaylight-flow-statistics:flow-statistics']['packet-count']
            b_count = flow['opendaylight-flow-statistics:flow-statistics']['byte-count']
            duration = flow['opendaylight-flow-statistics:flow-statistics']['duration']['second']

            temp_flow_tab.append(flow_entry(s_addr, d_addr, out_port, prio, hto, p_count, b_count, duration))

            '''
            print(' source=',s_addr,
                  ' destination=',d_addr,
                  ' output=',out_port,
                  'prio=',prio,
                  'hto=',hto,
                  'p_count',p_count,
                  'b_count',b_count,
                  'duration',duration
                  )

            '''
        # endif
    # endloop

    if len(temp_flow_tab) == 0:
        temp_flow_tab.append(flow_entry())
    if switch not in flow_tab_list:
        flow_tab_list.append([switch, temp_flow_tab])


def show_flow_tab():
    create_tab()
    clear_db()
    for flow_tab in flow_tab_list:
        print(flow_tab[0])
        for flow_entry in flow_tab[1]:
            fe = flow_entry.get_flow()
            print(fe)
            insert_data(flow_tab[0], fe)


def main_flow_fetch(switch_l, cont_ip):
    '''cont_ip=input('Enter Controller IP : ')'''
    switch_list = switch_l
    for switch in switch_list:
        generate_flow_tab(switch, cont_ip)

    show_flow_tab()


''' -------------------------------------------------------------------------------------------------
    main function  
    -------------------------------------------------------------------------------------------------'''


def main():
    cont_ip = input('enter controller ip : ')
    generate_lists(cont_ip)
    show_discovery_message()
    # build_adj_mat()
    main_flow_fetch(switch_list, cont_ip)
    g = generate_graph()
    spa_main(g)


''' -------------------------------------------------------------------------------------------------
    call starts here !! 
    -------------------------------------------------------------------------------------------------'''

main()
