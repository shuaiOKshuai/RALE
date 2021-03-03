#encoding=utf-8
import numpy as np
import networkx as nx
import scipy as spy
import scipy.sparse as sp

def readNodesFromFile(node_file, datasetName):
    remove_first_two_values_in_features = False 
    nodes_num = 0
    features_num = 0
    node_label=None 
    label_nodes={} 
    features = None
    with open(node_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)<=0:
                continue
            arr=tmp.split()
            if len(arr)>2:
                n = int(arr[0])
                l = int(arr[1])
                node_label[n]=l 
                if l in label_nodes: 
                    label_nodes[l].append(n)
                else:
                    label_nodes[l]=[n]
                if not remove_first_two_values_in_features: 
                    features[n] = arr[2:] 
                else: 
                    features[n] = arr[4:] 
                continue
            if len(arr)==2: 
                nodes_num=int(arr[0])
                if not remove_first_two_values_in_features: 
                    features_num=int(arr[1])
                else: 
                    features_num=int(arr[1])-2
                node_label = np.zeros((nodes_num,), dtype=np.int32)
                features=np.zeros((nodes_num, features_num), dtype=np.float32)
    f.close()
    label_nodes_list = []
    for i in range(len(label_nodes)):
        label_nodes_list.append(label_nodes[i])
    
    if 'reddit' in datasetName: 
        features[:, 0] = np.log(features[:, 0] + 1.0)
        features[:, 1] = np.log(features[:, 1] - min(np.min(features[:, 1]), -1))
        features = features / features.sum(axis=1)[:,None]
    else: 
        features = features / features.sum(axis=1)[:,None]
    
    return node_label, label_nodes_list, features


def readEdgesFromFile(edge_file, node_num):
    adj=np.zeros((node_num,node_num))
    with open(edge_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                adj[int(arr[0]), int(arr[1])]=1.0
    return adj


def readEdgesFromFile_sparse(edge_file, nodes_num):
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes_num))
    with open(edge_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                graph.add_edge(int(arr[0]), int(arr[1]))
    adj = nx.adjacency_matrix(graph)
    neis_num = adj.sum(axis=1)
    neis_num = np.squeeze(neis_num.A)
    return adj, neis_num


def preprocess_adj_normalization(adj):
    num_nodes = adj.shape[0] 
    adj = adj + np.eye(num_nodes)  
    adj[adj > 0.0] = 1.0
    D_ = np.diag(np.power(np.sum(adj, axis=1), -0.5)) 
    adjNor = np.dot(np.dot(D_, adj), D_)
    return adj, adjNor


def preprocess_adj_normalization_sparse(adj):
    num_nodes = adj.shape[0] 
    ones = np.ones((num_nodes))
    ones_diag = sp.spdiags(ones, 0, ones.size, ones.size)
    adj_self = adj + ones_diag  
    tmp = np.sum(adj_self, axis=1)
    tmp = tmp.A
    tmp = np.squeeze(tmp)
    tmp = np.power(tmp, -0.5)
    D_ = sp.spdiags(tmp, 0, tmp.size, tmp.size) 
    adj_self_nor = np.dot(np.dot(D_, adj_self), D_)
    return adj_self, adj_self_nor


def processNodeInfo(adj, mask_nor, node_num):
    max_nei_num = np.max(np.sum(adj, axis=1)).astype(np.int32)
    neis = np.zeros((node_num,max_nei_num), dtype=np.int32)
    neis_mask = np.zeros((node_num,max_nei_num), dtype=np.float32)
    neis_mask_nor = np.zeros((node_num,max_nei_num), dtype=np.float32)
    neighboursDict = []
    inner_index = 0 
    for i in range(node_num):
        inner_index = 0
        nd = [] 
        for j in range(node_num):
            if adj[i][j]==1.0: 
                neis[i][inner_index] = j 
                neis_mask[i][inner_index] = 1.0
                neis_mask_nor[i][inner_index] = mask_nor[i][j]
                if i!=j: 
                    nd.append(j)
                inner_index += 1
        neighboursDict.append(nd)
    
    return neis, neis_mask, neis_mask_nor, neighboursDict


def processNodeNeighborsDict(edge_file, nodes_num):
    neighboursDict = [[] for i in range(nodes_num)]
    with open(edge_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                n0 = int(arr[0])
                n1 = int(arr[1])
                neighboursDict[n0].append(n1)
    return neighboursDict


def processNodeInfo_sparse_sparse(adj, mask_nor, node_num):
    max_nei_num = np.max(np.sum(adj, axis=1)).astype(np.int32)
    neis_row = []
    neis_col = []
    neis_value = []
    neis_mask_row = []
    neis_mask_col = []
    neis_mask_value = []
    neis_mask_nor_row = []
    neis_mask_nor_col = []
    neis_mask_nor_value = []
    AB = adj.nonzero()
    A = AB[0] 
    B = AB[1] 
    former = -1 
    col_index = 0 
    for i in range(len(A)):
        if A[i] != former: 
            col_index = 0
            former = A[i]
        neis_row.append(A[i])
        neis_mask_row.append(A[i])
        neis_mask_nor_row.append(A[i])
        neis_col.append(col_index)
        neis_mask_col.append(col_index)
        neis_mask_nor_col.append(col_index)
        neis_value.append(B[i])
        neis_mask_value.append(1.0)
        neis_mask_nor_value.append(mask_nor[A[i], B[i]])
        col_index += 1
    neis = sp.csc_matrix((spy.array(neis_value),(spy.array(neis_row), spy.array(neis_col))),shape=(node_num, max_nei_num))
    neis_mask_nor = sp.csc_matrix((spy.array(neis_mask_nor_value),(spy.array(neis_mask_nor_row), spy.array(neis_mask_nor_col))),shape=(node_num, max_nei_num))
    neis = neis.toarray()
    neis_mask_nor = neis_mask_nor.toarray().astype(np.float32)
    return neis, neis_mask_nor


def processNodeInfo_sparse_dense_maxDegree(adj, mask_nor, nodes_num, max_degree):
    max_nei_num = np.max(np.sum(adj, axis=1)).astype(np.int32)
    neis_row = []
    neis_col = []
    neis_value = []
    neis_mask_row = []
    neis_mask_col = []
    neis_mask_value = []
    neis_mask_nor_row = []
    neis_mask_nor_col = []
    neis_mask_nor_value = []
    AB = adj.nonzero()
    A = AB[0] 
    B = AB[1] 
    former = -1 
    col_index = 0 
    for i in range(len(A)):
        if A[i] != former: 
            col_index = 0
            former = A[i]
        neis_row.append(A[i])
        neis_mask_row.append(A[i])
        neis_mask_nor_row.append(A[i])
        neis_col.append(col_index)
        neis_mask_col.append(col_index)
        neis_mask_nor_col.append(col_index)
        neis_value.append(B[i])
        neis_mask_value.append(1.0)
        neis_mask_nor_value.append(mask_nor[A[i], B[i]])
        col_index += 1
    neis = sp.csc_matrix((spy.array(neis_value),(spy.array(neis_row), spy.array(neis_col))),shape=(nodes_num, max_nei_num))
    neis = neis.toarray() 
    
    neis_new = np.zeros((nodes_num, max_degree), dtype=np.int32) 
    degrees = np.squeeze(np.sum(adj, axis=1).A.astype(np.int32))
    for i in range(nodes_num): 
        neighbour = None
        if degrees[i] >= max_degree: 
            neighbour = np.random.choice(neis[i, :degrees[i]], max_degree, replace=False) 
        else: 
            neighbour1 = neis[i, :degrees[i]] 
            neighbour2 = np.random.choice(neis[i, :degrees[i]], max_degree-degrees[i], replace=True) 
            neighbour = np.concatenate((neighbour1, neighbour2))
            np.random.shuffle(neighbour)
            
        neis_new[i] = neighbour
    return neis_new


def readEdgesFromFile_onlyNeighbours(edge_file, node_num):
    neighbours = {} 
    with open(edge_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                start = int(arr[0])
                end = int(arr[1])
                if start in neighbours:
                    neighbours[start].append(end)
                else:
                    neighbours[start] = [end]
    assert len(neighbours) == node_num 
    neighborsList = [neighbours[i] for i in range(node_num)]
    maxLen = max(len(l) for l in neighborsList)
    neighboursArray = np.zeros((node_num, maxLen), dtype=np.int32)
    neighboursMask = np.zeros((node_num, maxLen), dtype=np.float32)
    for i in range(node_num):
        length = len(neighborsList[i])
        neighboursArray[i][:length]=neighborsList[i]
        neighboursMask[i][:length] = np.ones((length,), dtype=np.int32)
    return neighboursArray, neighboursMask


def readAllClassIdsFromFile(file_path):
    li = []
    with open(file_path) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                li.append(int(tmp))
    return li

