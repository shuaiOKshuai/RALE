#encoding=utf-8
'''
some tool functions
'''
import numpy as np
import tensorflow as tf
import os
import pickle
import processTools
import tqdm
import random
import math
import datetime, time
import copy
import networkx as nx


def tasks_genetation_and_load(options, all_labels_nodes_map, metatrain_classes_file, metaval_classes_file, metatest_classes_file):
    
    n_nodes = options['kshot']+options['kquery'] 
    train_tasks_file = options['root_dir'] + 'datasets-splits/trainTasksSplit_'+str(options['split_index'])+'.pkl'
    val_tasks_file = options['root_dir'] + 'datasets-splits/valTasksSplit_'+str(options['split_index'])+'.pkl'
    test_tasks_file = options['root_dir'] + 'datasets-splits/testTasksSplit_'+str(options['split_index'])+'.pkl'
    if os.path.exists(train_tasks_file): 
        with open(train_tasks_file, 'rb') as f: 
            all_select_nodes_train = pickle.load(f) 
            print('load training tasks splits for split--' + str(options['split_index']) + ' from file, len == ', len(all_select_nodes_train))
    else: 
        all_select_nodes_train = generate_tasks_and_save(metatrain_classes_file, train_tasks_file, options['train_batch_num']*options['batch_size'], all_labels_nodes_map, options['nway'], n_nodes)
    
    if os.path.exists(val_tasks_file): 
        with open(val_tasks_file, 'rb') as f: 
            all_select_nodes_val = pickle.load(f) 
            print('load val tasks splits for split--' + str(options['split_index']) + ' from file, len == ', len(all_select_nodes_val))
    else: 
        all_select_nodes_val = generate_tasks_and_save(metaval_classes_file, val_tasks_file, options['val_batch_num']*options['batch_size'], all_labels_nodes_map, options['nway'], n_nodes)
    
    if os.path.exists(test_tasks_file): 
        with open(test_tasks_file, 'rb') as f: 
            all_select_nodes_test = pickle.load(f) 
            print('load test tasks splits for split--' + str(options['split_index']) + ' from file, len == ', len(all_select_nodes_test))
    else: 
        all_select_nodes_test = generate_tasks_and_save(metatest_classes_file, test_tasks_file, options['test_batch_num']*options['batch_size'], all_labels_nodes_map, options['nway'], n_nodes)
    
    return all_select_nodes_train, all_select_nodes_val, all_select_nodes_test


def tasks_genetation_just_load(options):
    
    n_nodes = options['kshot']+options['kquery'] 
    train_tasks_file = options['root_dir'] + 'datasets-splits/trainTasksSplit_'+str(options['split_index'])+'.pkl'
    val_tasks_file = options['root_dir'] + 'datasets-splits/valTasksSplit_'+str(options['split_index'])+'.pkl'
    test_tasks_file = options['root_dir'] + 'datasets-splits/testTasksSplit_'+str(options['split_index'])+'.pkl'
    if os.path.exists(train_tasks_file): 
        with open(train_tasks_file, 'rb') as f: 
            all_select_nodes_train = pickle.load(f) 
            print('load training tasks splits for split--' + str(options['split_index']) + ' from file, len == ', len(all_select_nodes_train))
    
    if os.path.exists(val_tasks_file): 
        with open(val_tasks_file, 'rb') as f: 
            all_select_nodes_val = pickle.load(f) 
            print('load val tasks splits for split--' + str(options['split_index']) + ' from file, len == ', len(all_select_nodes_val))
    
    if os.path.exists(test_tasks_file): 
        with open(test_tasks_file, 'rb') as f: 
            all_select_nodes_test = pickle.load(f) 
            print('load test tasks splits for split--' + str(options['split_index']) + ' from file, len == ', len(all_select_nodes_test))
    
    return all_select_nodes_train, all_select_nodes_val, all_select_nodes_test
    
    

def get_nodes(sampled_classes, all_labels_nodes_map, labels_list, nb_samples=None, shuffle=False):
    
    sampled_classes = np.array(sampled_classes)
    all_labels_nodes_map = np.array(all_labels_nodes_map) 
    if nb_samples is not None: 
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    label_node_tuples = [(label, node) for label, node_list in zip(labels_list, all_labels_nodes_map[sampled_classes]) for node in sampler(node_list)]
    if shuffle:
        random.shuffle(label_node_tuples)
    return label_node_tuples



def generate_tasks_and_save(classes_file, tasks_save_file, total_batches_num, all_labels_nodes_map, nway, nimg):
    
    all_select_nodes = [] 
    classes = processTools.readAllClassIdsFromFile(classes_file)
    for _ in tqdm.tqdm(range(total_batches_num), 'generating batches and tasks'): 
        sampled_classes = random.sample(classes, nway)
        random.shuffle(sampled_classes) 
        labels_and_nodes_task = get_nodes(sampled_classes, all_labels_nodes_map, range(nway), nb_samples=nimg, shuffle=False) 
        select_labels_task = [li[0] for li in labels_and_nodes_task] 
        select_nodes_task = [li[1] for li in labels_and_nodes_task] 
        all_select_nodes.extend(select_nodes_task) 
    with open(tasks_save_file, 'wb') as f:
        pickle.dump(all_select_nodes,f)
        print('save training tasks splits ' + ' to ' + tasks_save_file)
    
    return all_select_nodes


def mapSortByValueDESC(map,top):
    if top>len(map): 
        top=len(map)
    items=map.items() 
    backitems=[[v[1],v[0]] for v in items]  
    backitems.sort(reverse=True) 
    e=[ backitems[i][1] for i in range(top)]  
    return e


def getHubNodes_by_pagerank(nodes_num, edge_file, topk):
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes_num))
    with open(edge_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr=tmp.split()
                graph.add_edge(int(arr[0]), int(arr[1]))
    pr = nx.pagerank(graph)
    topNodes = mapSortByValueDESC(pr, topk)
    return topNodes
        

def getAllNodePairs(options, batchs_num, all_select_nodes):
    node_pairs = {} 
    n_nodes = options['kshot'] + options['kquery'] 
    examples_per_task = n_nodes * options['nway'] 
    examples_per_batch = examples_per_task * options['batch_size'] 
    for i in tqdm.tqdm(range(batchs_num), 'loop for batches num'):
        batch_nodes = all_select_nodes[i * examples_per_batch : (i + 1) * examples_per_batch] 
        for j in range(options['batch_size']): 
            task_nodes = batch_nodes[j * examples_per_task : (j+1) * examples_per_task]
            support_nodes, query_nodes = support_query_from_taskNodes(task_nodes, options['kshot'], options['kquery'], options['nway']) 
            for node in support_nodes:
                if node in node_pairs: 
                    node_pairs[node].update(support_nodes) 
                    node_pairs[node].update(query_nodes) 
                else: 
                    node_pairs[node] = set(support_nodes) 
                    node_pairs[node].update(query_nodes) 
            for node in query_nodes:
                if node in node_pairs: 
                    node_pairs[node].update(support_nodes)
                else: 
                    node_pairs[node] = set(support_nodes)
    return node_pairs


def support_query_from_taskNodes(task_nodes, kshot, kquery, nway):
    nodes_num_per_class = kshot + kquery 
    task_nodes_array = np.array(task_nodes)
    task_nodes_array = np.reshape(task_nodes_array, (nway, nodes_num_per_class)) 
    support_nodes = task_nodes_array[:, :kshot] 
    query_nodes = task_nodes_array[:, kshot:] 
    support_nodes = list(support_nodes.reshape((-1,))) 
    query_nodes = list(query_nodes.reshape((-1,)))
    
    return support_nodes, query_nodes
    
    

def randomWalkSampling(neighboursDict, nodes_num, batch_size, timesPerNode, maxLengthPerPath, randomWalkPathsFile):
    batchs_num = math.ceil(nodes_num / batch_size) 
    all_nodes = np.arange(nodes_num) 
    with open(randomWalkPathsFile, 'w') as output:
        for b in tqdm.tqdm(range(batchs_num), 'random walk sampling batches'): 
            ids_arr = all_nodes[b*batch_size : (b+1)*batch_size] 
            walks = np.repeat(ids_arr, timesPerNode)[None,:] 
            for i in range(1, maxLengthPerPath): 
                newRow = np.array([random.choice(neighboursDict[j]) for j in walks[i-1]])[None,:]
                walks = np.concatenate((walks, newRow), axis=0)
            all_str = ""
            for i in range(walks.shape[1]): 
                tmp = " ".join([str(j) for j in walks[:,i]]) 
                all_str += tmp + "\n"
            output.write(all_str) 
    output.close()

    
def subpathsExtractionMatch(randomWalkPathsFile, process_batch_size, minLen_subpath, maxLen_subpath, maxLen_subpath_hub, subpathsFile, node_pairs, hub_nodes):
    pathsBatch = []
    hub_nodes_set = set(hub_nodes)
    subpaths_output = open(subpathsFile, 'w')
    with open(randomWalkPathsFile) as f:
        for line in f:
            tmp=line.strip()
            if len(tmp)>0:
                arr = tmp.split() 
                arr = [int(x) for x in arr] 
                pathsBatch.append(arr)
                if len(pathsBatch) == process_batch_size: 
                    pathsArray = np.array(pathsBatch) 
                    walks = np.transpose(pathsArray) 
                    pathsBatch = []
                    for i in range(walks.shape[0]-1): 
                        for j in range(i+minLen_subpath-1, i+maxLen_subpath_hub): 
                            if j<walks.shape[0]: 
                                rows = np.arange(i, j+1)
                                subpaths = walks[rows] 
                                all_str = ""
                                for l in range(subpaths.shape[1]): 
                                    if subpaths[0,l] not in node_pairs or (subpaths[-1,l] not in node_pairs[subpaths[0,l]] and subpaths[-1,l] not in hub_nodes_set): 
                                        continue
                                    if subpaths[-1,l] not in node_pairs[subpaths[0,l]] and j-i+1>maxLen_subpath_hub: 
                                        continue
                                    tmp = " ".join([str(n) for n in subpaths[:,l]]) 
                                    tmp = str(subpaths[0,l]) + "\t" +  str(subpaths[-1,l]) + "\t" + tmp
                                    all_str += tmp + "\n"
                                subpaths_output.write(all_str)
                            else: 
                                break
    subpaths_output.close()
    

def getNodePairsFromBatch(nodes_batch, kshot, kquery, nway, task_node_size, batch_size):
    node_pairs_s_list = [] 
    node_pairs_q_list = []
    node_pairs_s_labels = [] 
    node_pairs_q_labels = []
    for index in range(batch_size):
        node_pairs_s_list_t = [] 
        node_pairs_q_list_t = []
        node_pairs_s_labels_t = [] 
        node_pairs_q_labels_t = []
        task_nodes = nodes_batch[index * task_node_size : (index+1) * task_node_size]
        support_nodes, query_nodes = support_query_from_taskNodes(task_nodes, kshot, kquery, nway) 
        support_len = len(support_nodes)
        query_len = len(query_nodes)
        for i in range(support_len):
            n_i = support_nodes[i]
            for j in range(support_len):
                n_j = support_nodes[j]
                node_pairs_s_list_t.append([n_i, n_j])
                if int(i/kshot) == int(j/kshot):
                    node_pairs_s_labels_t.append(1.)
                else:
                    node_pairs_s_labels_t.append(0.)
                    
        for i in range(query_len):
            n_i = query_nodes[i]
            for j in range(support_len):
                n_j = support_nodes[j]
                node_pairs_q_list_t.append([n_i, n_j])
                if int(i/kquery) == int(j/kshot): 
                    node_pairs_q_labels_t.append(1.)
                else:
                    node_pairs_q_labels_t.append(0.)
        node_pairs_s_list.append(node_pairs_s_list_t)
        node_pairs_q_list.append(node_pairs_q_list_t)
        node_pairs_s_labels.append(node_pairs_s_labels_t)
        node_pairs_q_labels.append(node_pairs_q_labels_t)
    assert len(node_pairs_s_list[0]) == (kshot*nway)*(kshot*nway)
    assert len(node_pairs_q_list[0]) == (kshot*nway)*(kquery*nway)
    return node_pairs_s_list, node_pairs_q_list, node_pairs_s_labels, node_pairs_q_labels


def prepareSubpatshForEachNodePair(subpaths_map, node_pair, direct_subpaths_num, hub_subpaths_num, subpaths_num, subpaths_len, non_subpaths, non_subpaths_lens):
    node_0 = node_pair[0]
    node_1 = node_pair[1]
    
    subpaths_list_t_pair = [] 
    subpaths_lens_list_t_pair = [] 
    valid_node_pair_list_t = [] 
    if node_0 not in subpaths_map or node_1 not in subpaths_map: 
        non_subpaths_tmp = copy.deepcopy(non_subpaths)
        for item in non_subpaths_tmp:
            item[0] = node_0
            item[-1] = node_1
        subpaths_list_t_pair.extend(non_subpaths_tmp)
        subpaths_lens_list_t_pair.extend(copy.deepcopy(non_subpaths_lens))
        valid_node_pair_list_t.append(0.) 
        
        non_subpaths_tmp = copy.deepcopy(non_subpaths)
        for item in non_subpaths_tmp: 
            item[0] = node_0
            item[-1] = node_1
        subpaths_list_t_pair.extend(non_subpaths_tmp)
        subpaths_lens_list_t_pair.extend(copy.deepcopy(non_subpaths_lens))
        valid_node_pair_list_t.append(0.) 
        return subpaths_list_t_pair, subpaths_lens_list_t_pair, valid_node_pair_list_t
    
    subpaths_map_0 = subpaths_map[node_0]
    subpaths_map_1 = subpaths_map[node_1]
    
    end_nodes_0 = set(subpaths_map_0.keys())
    end_nodes_1 = set(subpaths_map_1.keys())
    intersection = end_nodes_0 & end_nodes_1
    
    flag_subpaths_f = node_1 in subpaths_map_0 
    flag_hub_f = len(intersection)>0
    
    if flag_subpaths_f and flag_hub_f: 
        indeces = np.arange(len(subpaths_map_0[node_1])) 
        if len(subpaths_map_0[node_1]) >= direct_subpaths_num: 
            ids = np.random.choice(indeces, direct_subpaths_num, replace=False) 
        else:
            ids = np.random.choice(indeces, direct_subpaths_num, replace=True) 
        direct_subpaths = [[int(nid) for nid in subpaths_map_0[node_1][id].split(" ")] for id in ids] 
        
        intersection_list = list(intersection) 
        hub_subpaths = []
        for _ in range(hub_subpaths_num): 
            hub_n = random.choice(intersection_list)
            subpath_0 = random.choice(subpaths_map_0[hub_n])
            subpath_0 = [int(nid) for nid in subpath_0.split(" ")]
            subpath_1 = random.choice(subpaths_map_1[hub_n])
            subpath_1 = [int(nid) for nid in subpath_1.split(" ")]
            subpath_1.reverse() 
            subpath_com = subpath_0 + subpath_1[1:] 
            hub_subpaths.append(subpath_com)
        subpaths_all = direct_subpaths + hub_subpaths
        subpaths_all_mask = [[1. if xx<len(each_subpath) else 0. for xx in range(subpaths_len)] for each_subpath in subpaths_all] 
        zeros = [[node_1 for iii in range(len(each_subpath), subpaths_len)] for each_subpath in subpaths_all] 
        subpaths_all = [subpaths_all[subpaths_index] + zeros[subpaths_index] for subpaths_index in range(subpaths_num)] 
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(1.) 
    elif flag_subpaths_f and not flag_hub_f: 
        indeces = np.arange(len(subpaths_map_0[node_1])) 
        if len(subpaths_map_0[node_1]) >= subpaths_num: 
            ids = np.random.choice(indeces, subpaths_num, replace=False) 
        else:
            ids = np.random.choice(indeces, subpaths_num, replace=True) 
        direct_subpaths = [[int(nid) for nid in subpaths_map_0[node_1][id].split(" ")] for id in ids] 
        
        subpaths_all = direct_subpaths
        subpaths_all_mask = [[1. if xx<len(each_subpath) else 0. for xx in range(subpaths_len)] for each_subpath in subpaths_all] 
        zeros = [[node_1 for iii in range(len(each_subpath), subpaths_len)] for each_subpath in subpaths_all] 
        subpaths_all = [subpaths_all[subpaths_index] + zeros[subpaths_index] for subpaths_index in range(subpaths_num)] 
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(1.) 
    elif not flag_subpaths_f and flag_hub_f: 
        intersection_list = list(intersection) 
        hub_subpaths = []
        for _ in range(subpaths_num): 
            hub_n = random.choice(intersection_list)
            subpath_0 = random.choice(subpaths_map_0[hub_n])
            subpath_0 = [int(nid) for nid in subpath_0.split(" ")]
            subpath_1 = random.choice(subpaths_map_1[hub_n])
            subpath_1 = [int(nid) for nid in subpath_1.split(" ")]
            subpath_1.reverse() 
            subpath_com = subpath_0 + subpath_1[1:] 
            hub_subpaths.append(subpath_com)
        subpaths_all = hub_subpaths
        subpaths_all_mask = [[1. if xx<len(each_subpath) else 0. for xx in range(subpaths_len)] for each_subpath in subpaths_all] 
        zeros = [[node_1 for iii in range(len(each_subpath), subpaths_len)] for each_subpath in subpaths_all] 
        subpaths_all = [subpaths_all[subpaths_index] + zeros[subpaths_index] for subpaths_index in range(subpaths_num)] 
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(1.) 
    else: 
        subpaths_all = copy.deepcopy(non_subpaths)
        for item in subpaths_all:
            item[0] = node_0
            item[-1] = node_1
        
        subpaths_all_mask = copy.deepcopy(non_subpaths_lens)
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(0.) 
    
    node_1 = node_pair[0] 
    node_0 = node_pair[1]
    subpaths_map_0 = subpaths_map[node_0]
    subpaths_map_1 = subpaths_map[node_1]
    flag_subpaths_f = node_1 in subpaths_map_0 
    flag_hub_f = len(intersection)>0 
    if flag_subpaths_f and flag_hub_f: 
        indeces = np.arange(len(subpaths_map_0[node_1])) 
        if len(subpaths_map_0[node_1]) >= direct_subpaths_num: 
            ids = np.random.choice(indeces, direct_subpaths_num, replace=False) 
        else:
            ids = np.random.choice(indeces, direct_subpaths_num, replace=True) 
        direct_subpaths = [[int(nid) for nid in subpaths_map_0[node_1][id].split(" ")] for id in ids] 
        
        intersection_list = list(intersection) 
        hub_subpaths = []
        for _ in range(hub_subpaths_num): 
            hub_n = random.choice(intersection_list)
            subpath_0 = random.choice(subpaths_map_0[hub_n])
            subpath_0 = [int(nid) for nid in subpath_0.split(" ")]
            subpath_1 = random.choice(subpaths_map_1[hub_n])
            subpath_1 = [int(nid) for nid in subpath_1.split(" ")]
            subpath_1.reverse() 
            subpath_com = subpath_0 + subpath_1[1:] 
            hub_subpaths.append(subpath_com)
        subpaths_all = direct_subpaths + hub_subpaths
        subpaths_all_mask = [[1. if xx<len(each_subpath) else 0. for xx in range(subpaths_len)] for each_subpath in subpaths_all] 
        zeros = [[node_1 for iii in range(len(each_subpath), subpaths_len)] for each_subpath in subpaths_all] 
        subpaths_all = [subpaths_all[subpaths_index] + zeros[subpaths_index] for subpaths_index in range(subpaths_num)] 
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(1.) 
    elif flag_subpaths_f and not flag_hub_f: 
        indeces = np.arange(len(subpaths_map_0[node_1])) 
        if len(subpaths_map_0[node_1]) >= subpaths_num: 
            ids = np.random.choice(indeces, subpaths_num, replace=False)
        else:
            ids = np.random.choice(indeces, subpaths_num, replace=True) 
        direct_subpaths = [[int(nid) for nid in subpaths_map_0[node_1][id].split(" ")] for id in ids] 
        
        subpaths_all = direct_subpaths
        subpaths_all_mask = [[1. if xx<len(each_subpath) else 0. for xx in range(subpaths_len)] for each_subpath in subpaths_all] 
        zeros = [[node_1 for iii in range(len(each_subpath), subpaths_len)] for each_subpath in subpaths_all] 
        subpaths_all = [subpaths_all[subpaths_index] + zeros[subpaths_index] for subpaths_index in range(subpaths_num)] 
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(1.) 
    elif not flag_subpaths_f and flag_hub_f: 
        intersection_list = list(intersection) 
        hub_subpaths = []
        for _ in range(subpaths_num): 
            hub_n = random.choice(intersection_list)
            subpath_0 = random.choice(subpaths_map_0[hub_n])
            subpath_0 = [int(nid) for nid in subpath_0.split(" ")]
            subpath_1 = random.choice(subpaths_map_1[hub_n])
            subpath_1 = [int(nid) for nid in subpath_1.split(" ")]
            subpath_1.reverse() 
            subpath_com = subpath_0 + subpath_1[1:] 
            hub_subpaths.append(subpath_com)
        subpaths_all = hub_subpaths
        subpaths_all_mask = [[1. if xx<len(each_subpath) else 0. for xx in range(subpaths_len)] for each_subpath in subpaths_all] 
        zeros = [[node_1 for iii in range(len(each_subpath), subpaths_len)] for each_subpath in subpaths_all] 
        subpaths_all = [subpaths_all[subpaths_index] + zeros[subpaths_index] for subpaths_index in range(subpaths_num)] 
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(1.) 
    else: 
        subpaths_all = copy.deepcopy(non_subpaths)
        for item in subpaths_all:
            item[0] = node_0
            item[-1] = node_1
            
        subpaths_all_mask = copy.deepcopy(non_subpaths_lens)
        subpaths_list_t_pair.extend(subpaths_all)
        subpaths_lens_list_t_pair.extend(subpaths_all_mask)
        valid_node_pair_list_t.append(0.) 
    
    return subpaths_list_t_pair, subpaths_lens_list_t_pair, valid_node_pair_list_t
    

def prepareSubpathsToNodePairsAndSaveMatch(options, subpaths_map_batch, writer, batchs_num, all_select_nodes, subpaths_len, subpaths_num, data_save_file, hub_nodes, hub_nodes_set, subpaths_ratio, select_hub_nodes_num_per_node, subpaths_num_per_hubnode, hub_paths_len):
    n_nodes = options['kshot'] + options['kquery'] 
    examples_per_task = n_nodes * options['nway']
    examples_per_batch = examples_per_task * options['batch_size']
    examples_num_support = options['kshot'] * options['nway']
    
    non_subpaths = [[0 for j in range(subpaths_len)] for i in range(subpaths_num)] 
    non_subpaths_lens = [[0. for j in range(subpaths_len)] for i in range(subpaths_num)] 
    
    direct_subpaths_num = int(subpaths_num * subpaths_ratio) 
    hub_subpaths_num = subpaths_num - direct_subpaths_num 
    
    s_subpaths_valid_count = 0
    s_subpaths_valid_count_ideal = 0
    q_subpaths_valid_count = 0
    q_subpaths_valid_count_ideal = 0
    
    for i in tqdm.tqdm(range(batchs_num), 'prepare subpaths and save'):
        nodes_batch = all_select_nodes[i * examples_per_batch : (i + 1) * examples_per_batch] 
        node_pairs_s_list, node_pairs_q_list, node_pairs_s_labels, node_pairs_q_labels = getNodePairsFromBatch(nodes_batch, options['kshot'], options['kquery'], options['nway'], examples_per_task, options['batch_size'])
        node_pairs_s_valid = [] 
        node_pairs_q_valid = [] 
        subpaths_s_list = []
        subpaths_lens_s_list = []
        subpaths_q_list = []
        subpaths_lens_q_list = []
        hub_subpaths_s_list = []
        hub_subpaths_s_mask = []
        hub_subpaths_s_valid = []
        hub_subpaths_q_list = []
        hub_subpaths_q_mask = []
        hub_subpaths_q_valid = []
        for x in range(options['batch_size']): 
            node_pairs_s_valid_t = [] 
            node_pairs_q_valid_t = [] 
            subpaths_s_list_t = []
            subpaths_lens_s_list_t = []
            subpaths_q_list_t = []
            subpaths_lens_q_list_t = []
            
            hub_subpaths_s_list_t = []
            hub_subpaths_s_mask_t = []
            hub_subpaths_s_valid_t = []
            hub_subpaths_q_list_t = []
            hub_subpaths_q_mask_t = []
            hub_subpaths_q_valid_t = []
            
            node_pairs_s_list_t = node_pairs_s_list[x] 
            for j in range(len(node_pairs_s_list_t)):
                node_pair = node_pairs_s_list_t[j]
                subpaths_s_list_t_pair, subpaths_lens_s_list_t_pair, valid_node_pair_s_list_t = prepareSubpatshForEachNodePair(subpaths_map_batch, node_pair, direct_subpaths_num, hub_subpaths_num, subpaths_num, subpaths_len, non_subpaths, non_subpaths_lens)
                subpaths_s_list_t.append(subpaths_s_list_t_pair)
                subpaths_lens_s_list_t.append(subpaths_lens_s_list_t_pair)
                node_pairs_s_valid_t.append(valid_node_pair_s_list_t)
            
            node_pairs_q_list_t = node_pairs_q_list[x] 
            for j in range(len(node_pairs_q_list_t)):
                node_pair = node_pairs_q_list_t[j]
                subpaths_q_list_t_pair, subpaths_lens_q_list_t_pair, valid_node_pair_q_list_t = prepareSubpatshForEachNodePair(subpaths_map_batch, node_pair, direct_subpaths_num, hub_subpaths_num, subpaths_num, subpaths_len, non_subpaths, non_subpaths_lens)
                subpaths_q_list_t.append(subpaths_q_list_t_pair)
                subpaths_lens_q_list_t.append(subpaths_lens_q_list_t_pair)
                node_pairs_q_valid_t.append(valid_node_pair_q_list_t)
            
            task_nodes = nodes_batch[x * examples_per_task : (x+1) * examples_per_task] 
            support_nodes, query_nodes = support_query_from_taskNodes(task_nodes, options['kshot'], options['kquery'], options['nway']) 
            for s_node in support_nodes:
                if s_node not in subpaths_map_batch: 
                    hub_subpaths_s_list_i = [[[s_node for jj in range(hub_paths_len)] for ii in range(subpaths_num_per_hubnode)] for xx in range(select_hub_nodes_num_per_node)] 
                    hub_subpaths_s_mask_i = [[[0. for jj in range(hub_paths_len)] for ii in range(subpaths_num_per_hubnode)] for xx in range(select_hub_nodes_num_per_node)] 
                    hub_subpaths_s_valid_i = 0.
                    print(str(s_node) + ' has no subpaths ... ')
                else:
                    hub_subpaths_s_list_i, hub_subpaths_s_mask_i, hub_subpaths_s_valid_i = generateHubPahtsForEachNode(s_node, subpaths_map_batch[s_node], hub_nodes_set, select_hub_nodes_num_per_node, subpaths_num_per_hubnode, hub_paths_len)
                hub_subpaths_s_list_t.append(hub_subpaths_s_list_i)
                hub_subpaths_s_mask_t.append(hub_subpaths_s_mask_i)
                hub_subpaths_s_valid_t.append(hub_subpaths_s_valid_i)
            for q_node in query_nodes:
                if q_node not in subpaths_map_batch: 
                    hub_subpaths_q_list_i = [[[q_node for jj in range(hub_paths_len)] for ii in range(subpaths_num_per_hubnode)] for xx in range(select_hub_nodes_num_per_node)] 
                    hub_subpaths_q_mask_i = [[[0. for jj in range(hub_paths_len)] for ii in range(subpaths_num_per_hubnode)] for xx in range(select_hub_nodes_num_per_node)] 
                    hub_subpaths_q_valid_i = 0.
                    print(str(q_node) + ' has no subpaths ... ')
                else:
                    hub_subpaths_q_list_i, hub_subpaths_q_mask_i, hub_subpaths_q_valid_i = generateHubPahtsForEachNode(q_node, subpaths_map_batch[q_node], hub_nodes_set, select_hub_nodes_num_per_node, subpaths_num_per_hubnode, hub_paths_len)
                hub_subpaths_q_list_t.append(hub_subpaths_q_list_i)
                hub_subpaths_q_mask_t.append(hub_subpaths_q_mask_i)
                hub_subpaths_q_valid_t.append(hub_subpaths_q_valid_i)
            
            node_pairs_s_valid.append(node_pairs_s_valid_t)
            node_pairs_q_valid.append(node_pairs_q_valid_t)
            subpaths_s_list.append(subpaths_s_list_t)
            subpaths_lens_s_list.append(subpaths_lens_s_list_t)
            subpaths_q_list.append(subpaths_q_list_t)
            subpaths_lens_q_list.append(subpaths_lens_q_list_t)
            
            hub_subpaths_s_list.append(hub_subpaths_s_list_t)
            hub_subpaths_s_mask.append(hub_subpaths_s_mask_t)
            hub_subpaths_s_valid.append(hub_subpaths_s_valid_t)
            hub_subpaths_q_list.append(hub_subpaths_q_list_t)
            hub_subpaths_q_mask.append(hub_subpaths_q_mask_t)
            hub_subpaths_q_valid.append(hub_subpaths_q_valid_t)
        
        subpaths_s_list_new, subpaths_s_nodes, subpaths_s_nodes_lens = processBatchToFilterAllNodes(subpaths_s_list)
        subpaths_q_list_new, subpaths_q_nodes, subpaths_q_nodes_lens = processBatchToFilterAllNodes(subpaths_q_list)
        
        hub_subpaths_s_list_new, hub_subpaths_s_nodes, hub_subpaths_s_nodes_lens = processBatchToFilterAllNodes_hub_subpaths(hub_subpaths_s_list)
        hub_subpaths_q_list_new, hub_subpaths_q_nodes, hub_subpaths_q_nodes_lens = processBatchToFilterAllNodes_hub_subpaths(hub_subpaths_q_list)
        
        subpaths_s_array = np.array(subpaths_s_list_new).astype(np.int32) 
        subpaths_lens_s_array = np.array(subpaths_lens_s_list).astype(np.float32)
        node_pairs_s_labels_array = np.array(node_pairs_s_labels).astype(np.float32)
        node_pairs_s_valid_array = np.array(node_pairs_s_valid).astype(np.float32)
        subpaths_s_nodes_array = np.array(subpaths_s_nodes).astype(np.int32)
        subpaths_s_nodes_lens_array = np.array(subpaths_s_nodes_lens).astype(np.int32)
          
        subpaths_q_array = np.array(subpaths_q_list_new).astype(np.int32)
        subpaths_lens_q_array = np.array(subpaths_lens_q_list).astype(np.float32)
        node_pairs_q_labels_array = np.array(node_pairs_q_labels).astype(np.float32)
        node_pairs_q_valid_array = np.array(node_pairs_q_valid).astype(np.float32)
        subpaths_q_nodes_array = np.array(subpaths_q_nodes).astype(np.int32)
        subpaths_q_nodes_lens_array = np.array(subpaths_q_nodes_lens).astype(np.int32)
        
        hub_subpaths_s_array = np.array(hub_subpaths_s_list_new).astype(np.int32) 
        hub_subpaths_lens_s_array = np.array(hub_subpaths_s_mask).astype(np.float32)
        hub_subpaths_s_valid_array = np.array(hub_subpaths_s_valid).astype(np.float32)
        hub_subpaths_s_nodes_array = np.array(hub_subpaths_s_nodes).astype(np.int32)
        hub_subpaths_s_nodes_lens_array = np.array(hub_subpaths_s_nodes_lens).astype(np.int32)
          
        hub_subpaths_q_array = np.array(hub_subpaths_q_list_new).astype(np.int32) 
        hub_subpaths_lens_q_array = np.array(hub_subpaths_q_mask).astype(np.float32)
        hub_subpaths_q_valid_array = np.array(hub_subpaths_q_valid).astype(np.float32)
        hub_subpaths_q_nodes_array = np.array(hub_subpaths_q_nodes).astype(np.int32)
        hub_subpaths_q_nodes_lens_array = np.array(hub_subpaths_q_nodes_lens).astype(np.int32)
        
        node_pairs_s_valid_array_shape = node_pairs_s_valid_array.shape
        node_pairs_s_valid_array_sum = np.sum(node_pairs_s_valid_array)
        
        x = 1
        for i in range(len(node_pairs_s_valid_array_shape)):
            x = x*node_pairs_s_valid_array_shape[i]
        s_subpaths_valid_count_ideal += x
        s_subpaths_valid_count += node_pairs_s_valid_array_sum
        
        
        node_pairs_q_valid_array_shape = node_pairs_q_valid_array.shape
        node_pairs_q_valid_array_sum = np.sum(node_pairs_q_valid_array)
        
        x = 1
        for i in range(len(node_pairs_q_valid_array_shape)):
            x = x*node_pairs_q_valid_array_shape[i]
        q_subpaths_valid_count_ideal += x
        q_subpaths_valid_count += node_pairs_q_valid_array_sum
        
        
        subpaths_s_array = subpaths_s_array.tobytes()
        subpaths_lens_s_array = subpaths_lens_s_array.tobytes()
        node_pairs_s_labels_array = node_pairs_s_labels_array.tobytes()
        node_pairs_s_valid_array = node_pairs_s_valid_array.tobytes()
        subpaths_s_nodes_array = subpaths_s_nodes_array.tobytes()
        subpaths_s_nodes_lens_array = subpaths_s_nodes_lens_array.tobytes()
          
        subpaths_q_array = subpaths_q_array.tobytes()
        subpaths_lens_q_array = subpaths_lens_q_array.tobytes()
        node_pairs_q_labels_array = node_pairs_q_labels_array.tobytes()
        node_pairs_q_valid_array = node_pairs_q_valid_array.tobytes()
        subpaths_q_nodes_array = subpaths_q_nodes_array.tobytes()
        subpaths_q_nodes_lens_array = subpaths_q_nodes_lens_array.tobytes()
        
        hub_subpaths_s_array = hub_subpaths_s_array.tobytes() 
        hub_subpaths_lens_s_array = hub_subpaths_lens_s_array.tobytes()
        hub_subpaths_s_valid_array = hub_subpaths_s_valid_array.tobytes()
        hub_subpaths_s_nodes_array = hub_subpaths_s_nodes_array.tobytes()
        hub_subpaths_s_nodes_lens_array = hub_subpaths_s_nodes_lens_array.tobytes()
          
        hub_subpaths_q_array = hub_subpaths_q_array.tobytes() 
        hub_subpaths_lens_q_array = hub_subpaths_lens_q_array.tobytes()
        hub_subpaths_q_valid_array = hub_subpaths_q_valid_array.tobytes()
        hub_subpaths_q_nodes_array = hub_subpaths_q_nodes_array.tobytes()
        hub_subpaths_q_nodes_lens_array = hub_subpaths_q_nodes_lens_array.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                    'subpaths_s_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_s_array])),
                    'subpaths_lens_s_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_lens_s_array])),
                    'node_pairs_s_labels_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[node_pairs_s_labels_array])),
                    'node_pairs_s_valid_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[node_pairs_s_valid_array])),
                    'subpaths_s_nodes_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_s_nodes_array])),
                    'subpaths_s_nodes_lens_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_s_nodes_lens_array])),
                    
                    'subpaths_q_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_q_array])),
                    'subpaths_lens_q_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_lens_q_array])),
                    'node_pairs_q_labels_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[node_pairs_q_labels_array])),
                    'node_pairs_q_valid_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[node_pairs_q_valid_array])),
                    'subpaths_q_nodes_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_q_nodes_array])),
                    'subpaths_q_nodes_lens_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[subpaths_q_nodes_lens_array])),
                    
                    'hub_subpaths_s_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_s_array])),
                    'hub_subpaths_lens_s_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_lens_s_array])),
                    'hub_subpaths_s_valid_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_s_valid_array])),
                    'hub_subpaths_s_nodes_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_s_nodes_array])),
                    'hub_subpaths_s_nodes_lens_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_s_nodes_lens_array])),
                    
                    'hub_subpaths_q_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_q_array])),
                    'hub_subpaths_lens_q_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_lens_q_array])),
                    'hub_subpaths_q_valid_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_q_valid_array])),
                    'hub_subpaths_q_nodes_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_q_nodes_array])),
                    'hub_subpaths_q_nodes_lens_array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hub_subpaths_q_nodes_lens_array])),
                    }))  
        serialized = example.SerializeToString() 
        writer.write(serialized)
    
    writer.flush()
    

def generateHubPahtsForEachNode(current_node, subpaths_map_node, hub_nodes, select_hub_nodes_num, hub_subpaths_num, hub_paths_len):
    valid_hub_nodes = list(subpaths_map_node.keys() & hub_nodes) 
    if len(valid_hub_nodes) == 0: 
        non_subpaths = [[[current_node for j in range(hub_paths_len)] for i in range(hub_subpaths_num)] for x in range(select_hub_nodes_num)] 
        non_subpaths_mask = [[[0. for j in range(hub_paths_len)] for i in range(hub_subpaths_num)] for x in range(select_hub_nodes_num)] 
        return non_subpaths, non_subpaths_mask, 0.
    
    valid_hub_nodes_array = np.array(valid_hub_nodes)
    valid_hub_nodes_index = np.arange(len(valid_hub_nodes))
    if len(valid_hub_nodes_index) >= select_hub_nodes_num: 
        indeces = np.random.choice(valid_hub_nodes_index, select_hub_nodes_num, replace=False) 
    else:
        indeces = np.random.choice(valid_hub_nodes_index, select_hub_nodes_num, replace=True) 
    ids = valid_hub_nodes_array[indeces] 
    hub_subpaths = [] 
    hub_subpaths_mask = []
    for id in ids: 
        subpaths_tmp = subpaths_map_node[id]
        subpaths_tmp_index = np.arange(len(subpaths_tmp)) 
        if len(subpaths_tmp) >= hub_subpaths_num: 
            subpaths_index1 = np.random.choice(subpaths_tmp_index, hub_subpaths_num, replace=False) 
        else:
            subpaths_index1 = np.random.choice(subpaths_tmp_index, hub_subpaths_num, replace=True) 
        subpaths_tmp_choice = [[int(nid) for nid in subpaths_tmp[i].split(" ")] for i in subpaths_index1] 
        subpaths_tmp_choice_mask = [[1. if xx<len(each_subpath) else 0. for xx in range(hub_paths_len)] for each_subpath in subpaths_tmp_choice]
        zeros = [[id for iii in range(len(each_subpath), hub_paths_len)] for each_subpath in subpaths_tmp_choice] 
        subpaths_tmp_choice_padding = [subpaths_tmp_choice[subpaths_index] + zeros[subpaths_index] for subpaths_index in range(hub_subpaths_num)] 
        hub_subpaths.append(subpaths_tmp_choice_padding)
        hub_subpaths_mask.append(subpaths_tmp_choice_mask)
    
    return hub_subpaths, hub_subpaths_mask, 1.
    

def processBatchToFilterAllNodes(subpaths_list):
    batch_size = len(subpaths_list)
    dim1 = len(subpaths_list[0])
    dim2 = len(subpaths_list[0][0])
    dim3 = len(subpaths_list[0][0][0])
    max_len = 0
    nodes_list_all = []
    nodes_list_map = []
    lens_list = []
    for i in range(batch_size): 
        nodes_task = subpaths_list[i] 
        nodes_array = np.array(nodes_task) 
        nodes_list_distinct = list(set(nodes_array.flatten().tolist())) 
        id2index = dict(zip(nodes_list_distinct, np.arange(len(nodes_list_distinct)))) 
        
        nodes_array_1 = np.reshape(nodes_array, -1) 
        nodes_list_1 = list(nodes_array_1) 
        nodes_list_1 = list(map(lambda x: id2index[x], nodes_list_1)) 
        nodes_array_1 = np.array(nodes_list_1)
        nodes_array_new = np.reshape(nodes_array_1, [dim1, dim2, dim3]) 
        nodes_list_new = nodes_array_new.tolist()
        nodes_list_all.append(nodes_list_new)
        nodes_list_map.append(nodes_list_distinct)
        lens_list.append(len(nodes_list_distinct))
    
    max_len = dim1 * dim2 * dim3 
    zeros = [[0 for iii in range(len(each_list), max_len)] for each_list in nodes_list_map] 
    nodes_list_map = [nodes_list_map[list_index] + zeros[list_index] for list_index in range(batch_size)] 
    
    return nodes_list_all, nodes_list_map, lens_list


def processBatchToFilterAllNodes_hub_subpaths(subpaths_list):
    batch_size = len(subpaths_list)
    dim1 = len(subpaths_list[0])
    dim2 = len(subpaths_list[0][0])
    dim3 = len(subpaths_list[0][0][0])
    dim4 = len(subpaths_list[0][0][0][0])
    max_len = 0
    nodes_list_all = []
    nodes_list_map = []
    lens_list = []
    for i in range(batch_size): 
        nodes_task = subpaths_list[i] 
        nodes_array = np.array(nodes_task) 
        nodes_list_distinct = list(set(nodes_array.flatten().tolist())) 
        id2index = dict(zip(nodes_list_distinct, np.arange(len(nodes_list_distinct)))) 
        
        nodes_array_1 = np.reshape(nodes_array, -1) 
        nodes_list_1 = list(nodes_array_1) 
        nodes_list_1 = list(map(lambda x: id2index[x], nodes_list_1)) 
        nodes_array_1 = np.array(nodes_list_1)
        nodes_array_new = np.reshape(nodes_array_1, [dim1, dim2, dim3, dim4]) 
        nodes_list_new = nodes_array_new.tolist()
        nodes_list_all.append(nodes_list_new)
        nodes_list_map.append(nodes_list_distinct)
        lens_list.append(len(nodes_list_distinct))
    
    max_len = dim1 * dim2 * dim3 * dim4 
    zeros = [[0 for iii in range(len(each_list), max_len)] for each_list in nodes_list_map] 
    nodes_list_map = [nodes_list_map[list_index] + zeros[list_index] for list_index in range(batch_size)] 
    
    return nodes_list_all, nodes_list_map, lens_list
    


def readSubpathsAndSaveAsNPY(subpaths_file, saveDir, min_id, max_id, max_subpaths_num, subpaths_num_per_nodePair):
    subpathsMap = {} 
    with open(subpaths_file) as f:
        for l in f:
            tmp=l.strip()
            if len(tmp)>0:
                arr = tmp.split()
                
                start = int(arr[0])
                end = int(arr[1])
                flag_0 = start < min_id or start >= max_id 
                flag_1 = end < min_id or end >= max_id 
                
                if flag_0 and flag_1: 
                    continue
                elif start >= min_id and start < max_id: 
                    if start in subpathsMap: 
                        if end in subpathsMap[start]: 
                            if len(subpathsMap[start][end])<max_subpaths_num:
                                subpathsMap[start][end].append(" ".join(arr[2:]))
                        else: 
                            subpathsMap[start][end] = [" ".join(arr[2:])] 
                    else: 
                        subpathsMap[start] = {}
                        subpathsMap[start][end] = [" ".join(arr[2:])]
                
                if flag_1: 
                    continue
                b = arr[:2]
                c = arr[2:]
                b.reverse()
                c.reverse()
                arr = b + c
                start = int(arr[0])
                end = int(arr[1])
                if start >= min_id and start < max_id: 
                    if start in subpathsMap: 
                        if end in subpathsMap[start]: 
                            if len(subpathsMap[start][end])<max_subpaths_num: 
                                subpathsMap[start][end].append(" ".join(arr[2:]))
                        else: 
                            subpathsMap[start][end] = [" ".join(arr[2:])] 
                    else: 
                        subpathsMap[start] = {}
                        subpathsMap[start][end] = [" ".join(arr[2:])]
    f.close()
    
    subpathsMap_new = {}
    for start in subpathsMap: 
        for end in subpathsMap[start]:
            tmp = np.array(subpathsMap[start][end])
            indeces = np.arange(len(subpathsMap[start][end])) 
            if len(subpathsMap[start][end]) > subpaths_num_per_nodePair: 
                ids = np.random.choice(indeces, subpaths_num_per_nodePair, replace=False) 
            else:
                ids = np.arange(len(subpathsMap[start][end])) 
            subpaths = [l for l in tmp[ids]]
            
            if start in subpathsMap_new:
                subpathsMap_new[start][end] = subpaths 
            else: 
                subpathsMap_new[start] = {}
                subpathsMap_new[start][end] = subpaths
        
        subpathsMap[start] = None
    
    print('Start to save the subpaths maps in this period, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    keys = subpathsMap_new.keys()
    keys = list(keys)
    for key in tqdm.tqdm(keys, 'For each map key, save subpaths to file'):
        value = subpathsMap_new[key] 
        np.save(saveDir + str(key) + '.npy', eval(str(value))) 
        
        
def readSubpathsForBatchNodes(nodesList, subpathsByNodes_dir):
    subpathsMap = {}
    nodesSet = set(nodesList) 
    for nodeid in tqdm.tqdm(nodesSet, 'Load subpaths for the nodes batch'):
        subpaths_file_path = subpathsByNodes_dir + str(nodeid) + '.npy'
        if not os.path.exists(subpaths_file_path): 
            continue
        subpaths_m = np.load(subpaths_file_path, allow_pickle=True)
        subpathsMap[nodeid] = subpaths_m.item() 
    
    return subpathsMap
        

def randomwalkSamplingAndSubpathsProcessWhole_train(root_dir, split_index, kshot, kquery, nway, batch_size, train_batch_num, val_batch_num, test_batch_num, metatrain_classes_file, metaval_classes_file, metatest_classes_file, randomWalkPathsFile, subpathsFile_train, subpathsFile_val, subpathsFile_test, sampling_batch_size, samplingTimesPerNode, samplingMaxLengthPerPath, node_pairs_process_batch_size, minLen_subpath, maxLen_subpath, max_subpaths_num, subpaths_num_per_nodePair, hub_nodes_topk_ratio, maxLen_subpath_hub, subpaths_ratio, TFRecord_batch_sz, select_hub_nodes_num_per_node, subpaths_num_per_hubnode):
    options = locals().copy() 
    start_time=time.time()
    
    preprocessed_data_save_file_train = options['root_dir'] + 'datasets-splits/trainTasksSplit_'+str(options['split_index'])+'-preprocessed.tfrecord'
    all_select_nodes_train, all_select_nodes_val, all_select_nodes_test = tasks_genetation_just_load(options)
    hub_nodes_array = np.load(root_dir + 'hub_nodes_array.npy', allow_pickle=True)
    hub_nodes = list(hub_nodes_array)
    hub_nodes_set = set(hub_nodes)
    
    subpathsSaveDir_train = root_dir + 'datasets-splits/train-dir-' + str(split_index) + '/'
    
    n_nodes = options['kshot'] + options['kquery'] 
    examples_per_task = n_nodes * options['nway'] 
    examples_per_batch = examples_per_task * options['batch_size'] 
    writer = tf.python_io.TFRecordWriter(preprocessed_data_save_file_train) 
    tfRecord_batchs_num = int(math.ceil(train_batch_num / TFRecord_batch_sz))
    for batch_index in tqdm.tqdm(range(tfRecord_batchs_num), 'Loop for each TFRecord batch'):
        print('----------------------------------------------------')
        select_nodes = all_select_nodes_train[batch_index*TFRecord_batch_sz*examples_per_batch : (batch_index+1)*TFRecord_batch_sz*examples_per_batch]
        print('Generate node pairs from current batch, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        current_batch_sz = int(len(select_nodes) / examples_per_batch) 
        print('Load subpaths and process, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        subpaths_map_batch = readSubpathsForBatchNodes(select_nodes, subpathsSaveDir_train)
        print('Prepare the subpaths for each node pairs and save data into tfrecord file, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        prepareSubpathsToNodePairsAndSaveMatch(options, subpaths_map_batch, writer, current_batch_sz, select_nodes, options['maxLen_subpath'], subpaths_num_per_nodePair, preprocessed_data_save_file_train, hub_nodes, hub_nodes_set, options['subpaths_ratio'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub'])
    
        select_nodes = None
        subpaths_map_batch = None
        
    writer.close()
    print('Finish the data preparation, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    end_time=time.time()
    print('All cost time =', end_time-start_time,' s')
    

def randomwalkSamplingAndSubpathsProcessWhole_val(root_dir, split_index, kshot, kquery, nway, batch_size, train_batch_num, val_batch_num, test_batch_num, metatrain_classes_file, metaval_classes_file, metatest_classes_file, randomWalkPathsFile, subpathsFile_train, subpathsFile_val, subpathsFile_test, sampling_batch_size, samplingTimesPerNode, samplingMaxLengthPerPath, node_pairs_process_batch_size, minLen_subpath, maxLen_subpath, max_subpaths_num, subpaths_num_per_nodePair, hub_nodes_topk_ratio, maxLen_subpath_hub, subpaths_ratio, TFRecord_batch_sz, select_hub_nodes_num_per_node, subpaths_num_per_hubnode):
    options = locals().copy() 
    start_time=time.time()
    
    preprocessed_data_save_file_val = options['root_dir'] + 'datasets-splits/valTasksSplit_'+str(options['split_index'])+'-preprocessed.tfrecord'
    all_select_nodes_train, all_select_nodes_val, all_select_nodes_test = tasks_genetation_just_load(options)
    hub_nodes_array = np.load(root_dir + 'hub_nodes_array.npy', allow_pickle=True)
    hub_nodes = list(hub_nodes_array)
    hub_nodes_set = set(hub_nodes)
    
    subpathsSaveDir_val = root_dir + 'datasets-splits/val-dir-' + str(split_index) + '/'
    
    n_nodes = options['kshot'] + options['kquery'] 
    examples_per_task = n_nodes * options['nway'] 
    examples_per_batch = examples_per_task * options['batch_size'] 
    writer = tf.python_io.TFRecordWriter(preprocessed_data_save_file_val) 
    tfRecord_batchs_num = int(math.ceil(val_batch_num / TFRecord_batch_sz))
    for batch_index in tqdm.tqdm(range(tfRecord_batchs_num), 'Loop for each TFRecord batch'):
        select_nodes = all_select_nodes_val[batch_index*TFRecord_batch_sz*examples_per_batch : (batch_index+1)*TFRecord_batch_sz*examples_per_batch]
        print('Generate node pairs from current batch, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        current_batch_sz = int(len(select_nodes) / examples_per_batch) 
        print('Load subpaths and process, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        subpaths_map_batch = readSubpathsForBatchNodes(select_nodes, subpathsSaveDir_val)
        print('Prepare the subpaths for each node pairs and save data into tfrecord file, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        prepareSubpathsToNodePairsAndSaveMatch(options, subpaths_map_batch, writer, current_batch_sz, select_nodes, options['maxLen_subpath'], subpaths_num_per_nodePair, preprocessed_data_save_file_val, hub_nodes, hub_nodes_set, options['subpaths_ratio'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub'])
    
        select_nodes = None
        subpaths_map_batch = None
        
    writer.close()
    print('Finish the data preparation, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    end_time=time.time()
    print('All cost time =', end_time-start_time,' s')


def randomwalkSamplingAndSubpathsProcessWhole_test(root_dir, split_index, kshot, kquery, nway, batch_size, train_batch_num, val_batch_num, test_batch_num, metatrain_classes_file, metaval_classes_file, metatest_classes_file, randomWalkPathsFile, subpathsFile_train, subpathsFile_val, subpathsFile_test, sampling_batch_size, samplingTimesPerNode, samplingMaxLengthPerPath, node_pairs_process_batch_size, minLen_subpath, maxLen_subpath, max_subpaths_num, subpaths_num_per_nodePair, hub_nodes_topk_ratio, maxLen_subpath_hub, subpaths_ratio, TFRecord_batch_sz, select_hub_nodes_num_per_node, subpaths_num_per_hubnode):
    options = locals().copy() 
    start_time=time.time()
    
    preprocessed_data_save_file_test = options['root_dir'] + 'datasets-splits/testTasksSplit_'+str(options['split_index'])+'-preprocessed.tfrecord'
    all_select_nodes_train, all_select_nodes_val, all_select_nodes_test = tasks_genetation_just_load(options)
    hub_nodes_array = np.load(root_dir + 'hub_nodes_array.npy', allow_pickle=True)
    hub_nodes = list(hub_nodes_array)
    hub_nodes_set = set(hub_nodes)
    
    subpathsSaveDir_test = root_dir + 'datasets-splits/test-dir-' + str(split_index) + '/'
    
    n_nodes = options['kshot'] + options['kquery'] 
    examples_per_task = n_nodes * options['nway'] 
    examples_per_batch = examples_per_task * options['batch_size'] 
    writer = tf.python_io.TFRecordWriter(preprocessed_data_save_file_test) 
    tfRecord_batchs_num = int(math.ceil(test_batch_num / TFRecord_batch_sz))
    for batch_index in tqdm.tqdm(range(tfRecord_batchs_num), 'Loop for each TFRecord batch'):
        select_nodes = all_select_nodes_test[batch_index*TFRecord_batch_sz*examples_per_batch : (batch_index+1)*TFRecord_batch_sz*examples_per_batch]
        print('Generate node pairs from current batch, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        current_batch_sz = int(len(select_nodes) / examples_per_batch) 
        print('Load subpaths and process, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        subpaths_map_batch = readSubpathsForBatchNodes(select_nodes, subpathsSaveDir_test)
        print('Prepare the subpaths for each node pairs and save data into tfrecord file, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        prepareSubpathsToNodePairsAndSaveMatch(options, subpaths_map_batch, writer, current_batch_sz, select_nodes, options['maxLen_subpath'], subpaths_num_per_nodePair, preprocessed_data_save_file_test, hub_nodes, hub_nodes_set, options['subpaths_ratio'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub'])
    
        select_nodes = None
        subpaths_map_batch = None
        
    writer.close()
    print('Finish the data preparation, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    end_time=time.time()
    print('All cost time =', end_time-start_time,' s')
    


def prepareRandomWalkAndSubpaths(root_dir, dataset, split_index, kshot, kquery, nway, batch_size, train_batch_num, val_batch_num, test_batch_num, metatrain_classes_file, metaval_classes_file, metatest_classes_file, randomWalkPathsFile, subpathsFile_train, subpathsFile_val, subpathsFile_test, sampling_batch_size, samplingTimesPerNode, samplingMaxLengthPerPath, node_pairs_process_batch_size, minLen_subpath, maxLen_subpath, max_subpaths_num, subpaths_num_per_nodePair, hub_nodes_topk_ratio, maxLen_subpath_hub, subpaths_ratio, TFRecord_batch_sz, subpathsSaveByIdsBatchSz):
    options = locals().copy() 
    start_time=time.time()
    print('Start to load nodes information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    node_file = root_dir + 'graph.node'
    edge_file = root_dir + 'graph.edge'
    if os.path.exists(root_dir + 'node_label.npy'): 
        node_label = np.load(root_dir + 'node_label.npy', allow_pickle=True)
        label_nodes = np.load(root_dir + 'label_nodes.npy', allow_pickle=True)
        features = np.load(root_dir + 'features.npy', allow_pickle=True)
    else: 
        node_label, label_nodes, features = processTools.readNodesFromFile(node_file, dataset)
        np.save(root_dir + 'node_label.npy', node_label)
        np.save(root_dir + 'label_nodes.npy', label_nodes)
        np.save(root_dir + 'features.npy', features)
    options['nodes_num'] = nodes_num = node_label.shape[0] 
    options['labels_num'] = labels_num = len(label_nodes) 
    options['features_num'] = features_num = features.shape[1] 
    all_labels_nodes_map = label_nodes
    print('Start to load edges information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if os.path.exists(root_dir + 'neighborsDictList.npy'):
        neighboursDict = list(np.load(root_dir + 'neighborsDictList.npy', allow_pickle=True)) 
    else:
        neighboursDict = processTools.processNodeNeighborsDict(edge_file, nodes_num)
        np.save(root_dir + 'neighborsDictList.npy', neighboursDict)
    
    hub_nodes_topk = int(nodes_num * hub_nodes_topk_ratio)
    hub_nodes = getHubNodes_by_pagerank(nodes_num, edge_file, hub_nodes_topk)
    hub_nodes_array = np.array(hub_nodes)
    np.save(root_dir + 'hub_nodes_array.npy', hub_nodes_array)
    
    print('Generate the batches and tasks, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    all_select_nodes_train, all_select_nodes_val, all_select_nodes_test = tasks_genetation_and_load(options, all_labels_nodes_map, metatrain_classes_file, metaval_classes_file, metatest_classes_file)
    print('Generate all node pairs in the generated batches and tasks in training, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    node_pairs_train = getAllNodePairs(options, options['train_batch_num'], all_select_nodes_train)
    print('Generate all node pairs in the generated batches and tasks in val, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    node_pairs_val = getAllNodePairs(options, options['val_batch_num'], all_select_nodes_val)
    print('Generate all node pairs in the generated batches and tasks in test, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    node_pairs_test = getAllNodePairs(options, options['test_batch_num'], all_select_nodes_test)
    print('Start to random walk sampling, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if not os.path.exists(randomWalkPathsFile): 
        randomWalkSampling(neighboursDict, options['nodes_num'], options['sampling_batch_size'], options['samplingTimesPerNode'], options['samplingMaxLengthPerPath'], randomWalkPathsFile)
    print('Extract subpaths from the sampled paths, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    if not os.path.exists(subpathsFile_train): 
        subpathsExtractionMatch(randomWalkPathsFile, options['node_pairs_process_batch_size'], options['minLen_subpath'], options['maxLen_subpath'], options['maxLen_subpath_hub'], subpathsFile_train, node_pairs_train, hub_nodes)
    if not os.path.exists(subpathsFile_val): 
        subpathsExtractionMatch(randomWalkPathsFile, options['node_pairs_process_batch_size'], options['minLen_subpath'], options['maxLen_subpath'], options['maxLen_subpath_hub'], subpathsFile_val, node_pairs_val, hub_nodes)
    if not os.path.exists(subpathsFile_test): 
        subpathsExtractionMatch(randomWalkPathsFile, options['node_pairs_process_batch_size'], options['minLen_subpath'], options['maxLen_subpath'], options['maxLen_subpath_hub'], subpathsFile_test, node_pairs_test, hub_nodes)
    
    return nodes_num    


def reprocessSubpathsAndSaveByNodeIds(nodes_num, root_dir, split_index, kshot, kquery, nway, batch_size, train_batch_num, val_batch_num, test_batch_num, metatrain_classes_file, metaval_classes_file, metatest_classes_file, randomWalkPathsFile, subpathsFile_train, subpathsFile_val, subpathsFile_test, sampling_batch_size, samplingTimesPerNode, samplingMaxLengthPerPath, node_pairs_process_batch_size, minLen_subpath, maxLen_subpath, max_subpaths_num, subpaths_num_per_nodePair, hub_nodes_topk_ratio, maxLen_subpath_hub, subpaths_ratio, TFRecord_batch_sz, subpathsSaveByIdsBatchSz):
    
    print('Start to reprepare the train subpaths and then save them by node ids, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    start_time=time.time()
    
    subpathsSaveDir_train = root_dir + 'datasets-splits/train-dir-' + str(split_index) + '/'
    if not os.path.exists(subpathsSaveDir_train): 
        os.makedirs(subpathsSaveDir_train)
    subpaths_save_batch_num  = math.ceil(nodes_num / subpathsSaveByIdsBatchSz)
    for i in tqdm.tqdm(range(subpaths_save_batch_num), 'Save all subpaths for each node id'):
        print('Start to read all subpaths and filter, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        readSubpathsAndSaveAsNPY(subpathsFile_train, subpathsSaveDir_train, i*subpathsSaveByIdsBatchSz, (i+1)*subpathsSaveByIdsBatchSz, max_subpaths_num, subpaths_num_per_nodePair)
    
    print('Start to reprepare the val subpaths and then save them by node ids, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    subpathsSaveDir_val = root_dir + 'datasets-splits/val-dir-' + str(split_index) + '/'
    if not os.path.exists(subpathsSaveDir_val): 
        os.makedirs(subpathsSaveDir_val)
    subpathsSaveByIdsBatchSz = subpathsSaveByIdsBatchSz * 10 
    subpaths_save_batch_num  = math.ceil(nodes_num / subpathsSaveByIdsBatchSz)
    for i in tqdm.tqdm(range(subpaths_save_batch_num), 'Save all subpaths for each node id'):
        print('Start to read all subpaths and filter, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        readSubpathsAndSaveAsNPY(subpathsFile_val, subpathsSaveDir_val, i*subpathsSaveByIdsBatchSz, (i+1)*subpathsSaveByIdsBatchSz, max_subpaths_num, subpaths_num_per_nodePair)
    
    print('Start to reprepare the test subpaths and then save them by node ids, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    subpathsSaveDir_test = root_dir + 'datasets-splits/test-dir-' + str(split_index) + '/'
    if not os.path.exists(subpathsSaveDir_test): 
        os.makedirs(subpathsSaveDir_test)
    subpathsSaveByIdsBatchSz = subpathsSaveByIdsBatchSz * 10 
    subpaths_save_batch_num  = math.ceil(nodes_num / subpathsSaveByIdsBatchSz)
    for i in tqdm.tqdm(range(subpaths_save_batch_num), 'Save all subpaths for each node id'):
        print('Start to read all subpaths and filter, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
        readSubpathsAndSaveAsNPY(subpathsFile_test, subpathsSaveDir_test, i*subpathsSaveByIdsBatchSz, (i+1)*subpathsSaveByIdsBatchSz, max_subpaths_num, subpaths_num_per_nodePair)
    
    end_time=time.time()
    print('method reprocessSubpathsAndSaveByNodeIds cost time =', end_time-start_time,' s')
    
    
    
    
    