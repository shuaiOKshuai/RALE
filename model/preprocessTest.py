#encoding=utf-8

import os 
import configparser
import prepareDatasetMatch2


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
train_batch_num = cf.getint("param", "train_batch_num") 
ifTrain = cf.getboolean("param", "ifTrain")
val_batch_num = cf.getint("param", "val_batch_num") 
test_batch_num = cf.getint("param", "test_batch_num") 
meta_lr = cf.getfloat("param", "meta_lr")
inner_train_lr = cf.getfloat("param", "inner_train_lr")

max_degree = cf.getint("param", "max_degree")
max_degree_1_order = cf.getint("param", "max_degree_1_order")
max_degree_2_order = cf.getint("param", "max_degree_2_order")

metatrain_classes_file = root_dir + 'datasets-splits/train-class-'+str(split_index)
metaval_classes_file = root_dir + 'datasets-splits/val-class-'+str(split_index)
metatest_classes_file = root_dir + 'datasets-splits/test-class-'+str(split_index)
randomWalkPathsFile = root_dir + 'randomWalkPathsSaveFile'
subpathsFile_train = root_dir + 'datasets-splits/train-subpaths-'+str(split_index)
subpathsFile_val = root_dir + 'datasets-splits/val-subpaths-'+str(split_index)
subpathsFile_test = root_dir + 'datasets-splits/test-subpaths-'+str(split_index)

sampling_batch_size = cf.getint("param", "sampling_batch_size")
samplingTimesPerNode = cf.getint("param", "samplingTimesPerNode")
samplingMaxLengthPerPath = cf.getint("param", "samplingMaxLengthPerPath")
node_pairs_process_batch_size = cf.getint("param", "node_pairs_process_batch_size")
minLen_subpath = cf.getint("param", "minLen_subpath")
maxLen_subpath = cf.getint("param", "maxLen_subpath")
max_subpaths_num = cf.getint("param", "max_subpaths_num")
subpaths_num_per_nodePair = cf.getint("param", "subpaths_num_per_nodePair")
hub_nodes_topk_ratio = cf.getfloat("param", "hub_nodes_topk_ratio")
maxLen_subpath_hub = cf.getint("param", "maxLen_subpath_hub")
subpaths_ratio = cf.getfloat("param", "subpaths_ratio")
TFRecord_batch_sz = cf.getint("param", "TFRecord_batch_sz")
select_hub_nodes_num_per_node = cf.getint("param", "select_hub_nodes_num_per_node")
subpaths_num_per_hubnode = cf.getint("param", "subpaths_num_per_hubnode")

prepareDatasetMatch2.randomwalkSamplingAndSubpathsProcessWhole_test(
    root_dir, 
    split_index, 
    kshot, 
    kquery, 
    nway, 
    batch_size, 
    train_batch_num, 
    val_batch_num,
    test_batch_num, 
    metatrain_classes_file, 
    metaval_classes_file,
    metatest_classes_file, 
    randomWalkPathsFile, 
    subpathsFile_train, 
    subpathsFile_val,
    subpathsFile_test, 
    sampling_batch_size, 
    samplingTimesPerNode, 
    samplingMaxLengthPerPath,
    node_pairs_process_batch_size, 
    minLen_subpath, 
    maxLen_subpath, 
    max_subpaths_num, 
    subpaths_num_per_nodePair,
    hub_nodes_topk_ratio,
    maxLen_subpath_hub,
    subpaths_ratio,
    TFRecord_batch_sz,
    select_hub_nodes_num_per_node,
    subpaths_num_per_hubnode)