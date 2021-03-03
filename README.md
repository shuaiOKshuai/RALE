
# Relative and Absolute Location Embedding for Few-Shot Node Classification on Graph

We present the datasets and code for our paper "Relative and Absolute Location Embedding for Few-Shot Node Classification on Graph", which is published on AAAI-21.


## 1. Description for each file

	dataProcess:
		- preprocess_reddit.py : the reddit dataset could be processed by the functions inside this file.
		
	model:
		- paramsConfigPython : parameters setting file
		- mainEntry.py : the main entry of the model
		- modelTraining.py : the model training file 
		- amazon_reddit_data_generator.py : online pipeline dataset generator for Amazon and Reddit
		- email_data_generator.py : online pipeline dataset generator for Email dataset
		- amazon_reddit_maml.py : the main model for Amazon and Reddit
		- email_maml.py : the main model for Email
		- self_attentionModel.py : position-aware self-attention
		- preprocessPre.py : the offline random walk and segments sampling part
		- preprocessTrain.py : offline pipeline generation for training
		- preprocessVal.py : offline pipeline generation for validation
		- preprocessTest.py : offline pipeline generation for test
		- prepareDatasetMatch2.py : the main functions of offline pipeline generation
		- processTools.py : some tool functions
	
	datasets:
		- two datasets: Amazon and Email
		
		
## 2. Requirements (Environment)
	python-3.6.5
	tensorflow-1.13.1


## 3. How to run
- (1) First configure the parameters in file paramsConfigPython
	
	--- precomputation (offline) ---
- (2) Run 'python3 preprocessPre.py', to random walk and generate the segments
- (3) Run 'python3 preprocessTrain.py', to generate the offline pipeline for training
- (4) Run 'python3 preprocessVal.py', to generate the offline pipeline for validation
- (5) Run 'python3 preprocessTest.py', to generate the offline pipeline for test
	
	--- model evaluation (online) ---
- (6) Run 'python3 mainEntry.py' for training and validation, here we should set 'ifTrain = True' (meaning for training) in file paramsConfigPython before running this command
- (7) Run 'python3 mainEntry.py' for test, here we should set 'ifTrain = False' (meaning for test) in file paramsConfigPython before running this command

Note that, if you have already prepared the data (namely, the steps 1, 2, 3, 4 and 5 are finished), the prepared data would be saved on the disk (may occupy some space, according to the graph size). Thus, you can directly run steps 6 and 7.
	

## 4. Datasets

We include the Amazon and Email data in folder 'datasets'; while Reddit is too large so you can download it from link http://snap.stanford.edu/graphsage/, then you can use methods in 'preprocess_reddit.py' to process it into our format.
And you can also prepare your own datasets. The data format is as follows,
- (1) two files and a folder for class splits should be prepared: graph.node to describe the nodes, graph.edge to describe the edges, and datasets-splits to describe the class splits;
- (2) node file ( graph.node )
		The first row is the number of nodes + tab + the number of features
		In the following rows, each row represents a node: the first column is the node_id, the second column is the label_id of the node, and the third to the last columns are the features of this node. All these columns should be split by tabs.
- (3) edge file ( graph.edge )
		Each row is a directed edge, for example : 'a tab b' means a->b. We can add another line 'b tab a' to represent this is a bidirection edge. All these values should be split by tabs.
- (4) datasets-splits:
		All the classes should be split into train, val and test (each has its own file, e.g., file 'train-class-0', 'val-class-0' and 'test-class-0' are the 0-th split of classes for train, val and test, respectively). For example, in file 'test-class-0', each line is a class id of one class, and all the test classes (novel classes) are in this file.

## 5. Note
The different distribution of the sampled classes may result in high standard deviation. Thus, for each data split, please run several times and average the results. 
