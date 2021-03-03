#encoding=utf-8
import numpy as np
import os 
import configparser
import mamlTraining


cf = configparser.SafeConfigParser()
cf.read("paramsConfigPython")

root_dir = cf.get("param", "root_dir") 
dataset = cf.get("param", "dataset") # dataset name
root_dir = root_dir + dataset + '/'
gpu = cf.get("param", "gpu") # gpu id

os.environ["CUDA_VISIBLE_DEVICES"] = gpu 

split_index = cf.getint("param", "split_index")
inner_dim = cf.getint("param", "inner_dim") 
dropout = cf.getfloat("param", "dropout")
kshot = cf.getint("param", "kshot") 
kquery = cf.getint("param", "kquery")  
nway = cf.getint("param", "nway") 
batch_size = cf.getint("param", "batch_size") 
update_times = cf.getint("param", "update_times") 
val_epoch = cf.getint("param", "val_epoch")
train_batch_num = cf.getint("param", "train_batch_num") 
ifTrain = cf.getboolean("param", "ifTrain")
val_batch_num = cf.getint("param", "val_batch_num") 
test_batch_num = cf.getint("param", "test_batch_num") 
patience = cf.getint("param", "patience") 
meta_lr = cf.getfloat("param", "meta_lr")
inner_train_lr = cf.getfloat("param", "inner_train_lr")
num_heads = cf.getint("param", "num_heads")
l2_coef = cf.getfloat("param", "l2_coef")

max_degree = cf.getint("param", "max_degree")
max_degree_1_order = cf.getint("param", "max_degree_1_order")
max_degree_2_order = cf.getint("param", "max_degree_2_order")

maxLen_subpath = cf.getint("param", "maxLen_subpath")
subpaths_num_per_nodePair = cf.getint("param", "subpaths_num_per_nodePair")
alpha = cf.getfloat("param", "alpha")
clip_by_norm = cf.getfloat("param", "clip_by_norm")

select_hub_nodes_num_per_node = cf.getint("param", "select_hub_nodes_num_per_node")
subpaths_num_per_hubnode = cf.getint("param", "subpaths_num_per_hubnode")
maxLen_subpath_hub = cf.getint("param", "maxLen_subpath_hub")

mamlTraining.maml_main(
    root_dir, 
    dataset, 
    split_index,
    inner_dim, 
    max_degree,
    max_degree_1_order,
    max_degree_2_order,
    kshot, 
    kquery, 
    nway, 
    batch_size, 
    update_times, 
    val_epoch,
    train_batch_num,
    ifTrain,
    val_batch_num,
    test_batch_num,
    patience,
    dropout, 
    meta_lr,
    inner_train_lr,
    num_heads,
    l2_coef,
    maxLen_subpath,
    subpaths_num_per_nodePair,
    alpha,
    clip_by_norm,
    select_hub_nodes_num_per_node,
    subpaths_num_per_hubnode,
    maxLen_subpath_hub
    )