#encoding=utf-8
import numpy as np
import os, sys
import random
import tensorflow as tf
import tqdm
import pickle
import processTools



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


class ar_DataGenerator:

	def __init__(self, options, main_dir, all_labels_nodes_map, neisArray):
		self.main_dir = main_dir
		self.meta_batch_size = options['batch_size'] 
		# number of nodes to sample per class
		self.nimg = options['kshot'] + options['kquery'] 
		self.nway = options['nway'] 
		self.train_batch_num = options['train_batch_num'] 
		self.test_batch_num = options['test_batch_num'] 
		self.dim_output = options['nway'] 
		self.split_index = options['split_index'] 
		self.all_labels_nodes_map = all_labels_nodes_map 
		self.neisArray = neisArray
		self.max_degree_1_order = options['max_degree_1_order']
		self.max_degree_2_order = options['max_degree_2_order']
		self.max_degree = options['max_degree']

		self.metatrain_classes_file = self.main_dir  + 'datasets-splits/train-class-' + str(self.split_index) 
		self.metatest_classes_file = self.main_dir + 'datasets-splits/test-class-' + str(self.split_index) 
		
		print('metatrain_classes_file: ', self.metatrain_classes_file)
		print('metatest_classes_file: ', self.metatest_classes_file)

	
	def prepare_data_tensor(self, training):
		if training: 
			conduct_file = self.metatrain_classes_file 
			num_total_batches = self.train_batch_num 
		else: 
			conduct_file = self.metatest_classes_file 
			num_total_batches = self.test_batch_num
		
		if training and os.path.exists(self.main_dir + 'datasets-splits/trainTasksSplit_'+str(self.split_index)+'.pkl'):
			select_labels_task = np.arange(self.nway).repeat(self.nimg).tolist() 
			with open(self.main_dir + 'datasets-splits/trainTasksSplit_'+str(self.split_index)+'.pkl', 'rb') as f: 
				all_select_nodes = pickle.load(f) 
				print('load training tasks splits for split--' + str(self.split_index) + ' from file, len == ', len(all_select_nodes))

		else: # test or not existed
			all_select_nodes = [] 
			classes = processTools.readAllClassIdsFromFile(conduct_file)
			for _ in tqdm.tqdm(range(num_total_batches), 'generating batches and tasks'): 
				sampled_classes = random.sample(classes, self.nway) 
				random.shuffle(sampled_classes) 
				labels_and_nodes_task = get_nodes(sampled_classes, self.all_labels_nodes_map, range(self.nway), nb_samples=self.nimg, shuffle=False) 
				select_labels_task = [li[0] for li in labels_and_nodes_task] 
				select_nodes_task = [li[1] for li in labels_and_nodes_task] 
				all_select_nodes.extend(select_nodes_task) 
			if training: # only save for training.
				with open(self.main_dir + 'datasets-splits/trainTasksSplit_'+str(self.split_index)+'.pkl', 'wb') as f:
					pickle.dump(all_select_nodes,f)
					print('save training tasks splits for split--' + str(self.split_index) + ' to ' + self.main_dir + 'datasets-splits/trainTasksSplit_'+str(self.split_index)+'.pkl')
		
	
	def make_data_tensor(self, options, node_pairs_subpaths_file):
		options['subpaths_num_per_nodePair'] = options['subpaths_num_per_nodePair']
		# make queue for tensorflow to read from
		print('creating pipeline ops ...')
		filename_queue = tf.train.string_input_producer([node_pairs_subpaths_file], shuffle=False) 
		reader = tf.TFRecordReader() 
		_, serialized_example = reader.read(filename_queue)  
		features = tf.parse_single_example(     
	        serialized_example,  
	        features={ 
	            'subpaths_s_array': tf.FixedLenFeature([], tf.string), 
	            'subpaths_lens_s_array': tf.FixedLenFeature([], tf.string),
	            'node_pairs_s_labels_array': tf.FixedLenFeature([], tf.string),
	            'node_pairs_s_valid_array': tf.FixedLenFeature([], tf.string),
	            'subpaths_s_nodes_array': tf.FixedLenFeature([], tf.string),
	            'subpaths_s_nodes_lens_array': tf.FixedLenFeature([], tf.string),
	            
	            'subpaths_q_array': tf.FixedLenFeature([], tf.string),
	            'subpaths_lens_q_array': tf.FixedLenFeature([], tf.string),
	            'node_pairs_q_labels_array': tf.FixedLenFeature([], tf.string),
	            'node_pairs_q_valid_array': tf.FixedLenFeature([], tf.string),
	            'subpaths_q_nodes_array': tf.FixedLenFeature([], tf.string),
	            'subpaths_q_nodes_lens_array': tf.FixedLenFeature([], tf.string),
	            
	            'hub_subpaths_s_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_lens_s_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_s_valid_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_s_nodes_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_s_nodes_lens_array': tf.FixedLenFeature([], tf.string),
	            
	            'hub_subpaths_q_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_lens_q_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_q_valid_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_q_nodes_array': tf.FixedLenFeature([], tf.string),
	            'hub_subpaths_q_nodes_lens_array': tf.FixedLenFeature([], tf.string)
	        }
    	)  
		subpaths_s_array = tf.decode_raw(features['subpaths_s_array'], tf.int32) 
		subpaths_s_array = tf.reshape(subpaths_s_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kshot']*options['nway']), options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']]) 
		subpaths_lens_s_array = tf.decode_raw(features['subpaths_lens_s_array'], tf.float32) 
		subpaths_lens_s_array = tf.reshape(subpaths_lens_s_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kshot']*options['nway']), options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']]) 
		node_pairs_s_labels_array = tf.decode_raw(features['node_pairs_s_labels_array'], tf.float32) 
		node_pairs_s_labels_array = tf.reshape(node_pairs_s_labels_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kshot']*options['nway'])]) 
		node_pairs_s_valid_array = tf.decode_raw(features['node_pairs_s_valid_array'], tf.float32) 
		node_pairs_s_valid_array = tf.reshape(node_pairs_s_valid_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kshot']*options['nway']), 2]) 
		subpaths_s_nodes_array = tf.decode_raw(features['subpaths_s_nodes_array'], tf.int32)
		subpaths_s_nodes_array = tf.reshape(subpaths_s_nodes_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kshot']*options['nway'])*options['subpaths_num_per_nodePair']*2*options['maxLen_subpath']]) 
		subpaths_s_nodes_lens_array = tf.decode_raw(features['subpaths_s_nodes_lens_array'], tf.int32) 
		subpaths_s_nodes_lens_array = tf.reshape(subpaths_s_nodes_lens_array, [options['batch_size'], ]) 
		
		subpaths_q_array = tf.decode_raw(features['subpaths_q_array'], tf.int32) 
		subpaths_q_array = tf.reshape(subpaths_q_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kquery']*options['nway']), options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']]) 
		subpaths_lens_q_array = tf.decode_raw(features['subpaths_lens_q_array'], tf.float32) 
		subpaths_lens_q_array = tf.reshape(subpaths_lens_q_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kquery']*options['nway']), options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']]) 
		node_pairs_q_labels_array = tf.decode_raw(features['node_pairs_q_labels_array'], tf.float32) 
		node_pairs_q_labels_array = tf.reshape(node_pairs_q_labels_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kquery']*options['nway'])]) 
		node_pairs_q_valid_array = tf.decode_raw(features['node_pairs_q_valid_array'], tf.float32)
		node_pairs_q_valid_array = tf.reshape(node_pairs_q_valid_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kquery']*options['nway']), 2]) 
		subpaths_q_nodes_array = tf.decode_raw(features['subpaths_q_nodes_array'], tf.int32) 
		subpaths_q_nodes_array = tf.reshape(subpaths_q_nodes_array, [options['batch_size'], (options['kshot']*options['nway'])*(options['kquery']*options['nway'])*options['subpaths_num_per_nodePair']*2*options['maxLen_subpath']]) 
		subpaths_q_nodes_lens_array = tf.decode_raw(features['subpaths_q_nodes_lens_array'], tf.int32) 
		subpaths_q_nodes_lens_array = tf.reshape(subpaths_q_nodes_lens_array, [options['batch_size'], ]) 
		
		hub_subpaths_s_array = tf.decode_raw(features['hub_subpaths_s_array'], tf.int32)
		hub_subpaths_s_array = tf.reshape(hub_subpaths_s_array, [options['batch_size'], options['kshot']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']]) 
		hub_subpaths_lens_s_array = tf.decode_raw(features['hub_subpaths_lens_s_array'], tf.float32) 
		hub_subpaths_lens_s_array = tf.reshape(hub_subpaths_lens_s_array, [options['batch_size'], options['kshot']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']]) 
		hub_subpaths_s_valid_array = tf.decode_raw(features['hub_subpaths_s_valid_array'], tf.float32) 
		hub_subpaths_s_valid_array = tf.reshape(hub_subpaths_s_valid_array, [options['batch_size'], options['kshot']*options['nway']]) 
		hub_subpaths_s_nodes_array = tf.decode_raw(features['hub_subpaths_s_nodes_array'], tf.int32) 
		hub_subpaths_s_nodes_array = tf.reshape(hub_subpaths_s_nodes_array, [options['batch_size'], options['kshot']*options['nway'] * options['select_hub_nodes_num_per_node'] * options['subpaths_num_per_hubnode'] * options['maxLen_subpath_hub']]) 
		hub_subpaths_s_nodes_lens_array = tf.decode_raw(features['hub_subpaths_s_nodes_lens_array'], tf.int32) 
		hub_subpaths_s_nodes_lens_array = tf.reshape(hub_subpaths_s_nodes_lens_array, [options['batch_size'], ]) 
		
		hub_subpaths_q_array = tf.decode_raw(features['hub_subpaths_q_array'], tf.int32)
		hub_subpaths_q_array = tf.reshape(hub_subpaths_q_array, [options['batch_size'], options['kquery']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']]) 
		hub_subpaths_lens_q_array = tf.decode_raw(features['hub_subpaths_lens_q_array'], tf.float32) 
		hub_subpaths_lens_q_array = tf.reshape(hub_subpaths_lens_q_array, [options['batch_size'], options['kquery']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']]) 
		hub_subpaths_q_valid_array = tf.decode_raw(features['hub_subpaths_q_valid_array'], tf.float32) 
		hub_subpaths_q_valid_array = tf.reshape(hub_subpaths_q_valid_array, [options['batch_size'], options['kquery']*options['nway']]) 
		hub_subpaths_q_nodes_array = tf.decode_raw(features['hub_subpaths_q_nodes_array'], tf.int32) 
		hub_subpaths_q_nodes_array = tf.reshape(hub_subpaths_q_nodes_array, [options['batch_size'], options['kquery']*options['nway'] * options['select_hub_nodes_num_per_node'] * options['subpaths_num_per_hubnode'] * options['maxLen_subpath_hub']]) 
		hub_subpaths_q_nodes_lens_array = tf.decode_raw(features['hub_subpaths_q_nodes_lens_array'], tf.int32) 
		hub_subpaths_q_nodes_lens_array = tf.reshape(hub_subpaths_q_nodes_lens_array, [options['batch_size'], ]) 
		
		batch_sz = 1
		subpaths_s_array_batch, subpaths_lens_s_array_batch, node_pairs_s_labels_array_batch, node_pairs_s_valid_array_batch, subpaths_s_nodes_array_batch, subpaths_s_nodes_lens_array_batch, \
		subpaths_q_array_batch, subpaths_lens_q_array_batch, node_pairs_q_labels_array_batch, node_pairs_q_valid_array_batch, subpaths_q_nodes_array_batch, subpaths_q_nodes_lens_array_batch, \
		hub_subpaths_s_array_batch, hub_subpaths_lens_s_array_batch, hub_subpaths_s_valid_array_batch, hub_subpaths_s_nodes_array_batch, hub_subpaths_s_nodes_lens_array_batch, \
		hub_subpaths_q_array_batch, hub_subpaths_lens_q_array_batch, hub_subpaths_q_valid_array_batch, hub_subpaths_q_nodes_array_batch, hub_subpaths_q_nodes_lens_array_batch = tf.train.batch(
			[subpaths_s_array, subpaths_lens_s_array, node_pairs_s_labels_array, node_pairs_s_valid_array, subpaths_s_nodes_array, subpaths_s_nodes_lens_array, subpaths_q_array, subpaths_lens_q_array, node_pairs_q_labels_array, node_pairs_q_valid_array, subpaths_q_nodes_array, subpaths_q_nodes_lens_array, hub_subpaths_s_array, hub_subpaths_lens_s_array, hub_subpaths_s_valid_array, hub_subpaths_s_nodes_array, hub_subpaths_s_nodes_lens_array, hub_subpaths_q_array, hub_subpaths_lens_q_array, hub_subpaths_q_valid_array, hub_subpaths_q_nodes_array, hub_subpaths_q_nodes_lens_array], 
			batch_size=batch_sz, 
			capacity=32, 
			num_threads=1) 
		
		# support paths
		subpaths_q_array_batch = tf.reshape(subpaths_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		subpaths_lens_q_array_batch = tf.reshape(subpaths_lens_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		node_pairs_q_labels_array_batch = tf.reshape(node_pairs_q_labels_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot']])
		node_pairs_q_valid_array_batch = tf.reshape(node_pairs_q_valid_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot'], 2])
		subpaths_q_nodes_array_batch = tf.reshape(subpaths_q_nodes_array_batch, [options['batch_size'], (options['kshot']*options['nway'])*(options['kquery']*options['nway'])*options['subpaths_num_per_nodePair']*2*options['maxLen_subpath']])
		subpaths_q_nodes_lens_array_batch = tf.reshape(subpaths_q_nodes_lens_array_batch, [options['batch_size'], ])
		
		subpaths_q_array_batch = tf.reshape(subpaths_q_array_batch, [options['batch_size'], options['nway'], options['kquery'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		subpaths_lens_q_array_batch = tf.reshape(subpaths_lens_q_array_batch, [options['batch_size'], options['nway'], options['kquery'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		node_pairs_q_labels_array_batch = tf.reshape(node_pairs_q_labels_array_batch, [options['batch_size'], options['nway'], options['kquery'], options['nway'], options['kshot']])
		node_pairs_q_valid_array_batch = tf.reshape(node_pairs_q_valid_array_batch, [options['batch_size'], options['nway'], options['kquery'], options['nway'], options['kshot'], 2])
		
		subpaths_q_array_batch = subpaths_q_array_batch[:, :, :options['kquery']]
		subpaths_lens_q_array_batch = subpaths_lens_q_array_batch[:, :, :options['kquery']]
		node_pairs_q_labels_array_batch = node_pairs_q_labels_array_batch[:, :, :options['kquery']]
		node_pairs_q_valid_array_batch = node_pairs_q_valid_array_batch[:, :, :options['kquery']]
		
		
		# hub paths
		hub_subpaths_q_array_batch = tf.reshape(hub_subpaths_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_lens_q_array_batch = tf.reshape(hub_subpaths_lens_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_q_valid_array_batch = tf.reshape(hub_subpaths_q_valid_array_batch, [options['batch_size'], options['kquery']*options['nway']])
		hub_subpaths_q_nodes_array_batch = tf.reshape(hub_subpaths_q_nodes_array_batch, [options['batch_size'], options['kquery']*options['nway'] * options['select_hub_nodes_num_per_node'] * options['subpaths_num_per_hubnode'] * options['maxLen_subpath_hub']])
		hub_subpaths_q_nodes_lens_array_batch = tf.reshape(hub_subpaths_q_nodes_lens_array_batch, [options['batch_size'], ])
		
		hub_subpaths_q_array_batch = tf.reshape(hub_subpaths_q_array_batch, [options['batch_size'], options['nway'], options['kquery'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_lens_q_array_batch = tf.reshape(hub_subpaths_lens_q_array_batch, [options['batch_size'], options['nway'], options['kquery'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_q_valid_array_batch = tf.reshape(hub_subpaths_q_valid_array_batch, [options['batch_size'], options['nway'], options['kquery']])
		
		hub_subpaths_q_array_batch = hub_subpaths_q_array_batch[:, :, :options['kquery']]
		hub_subpaths_lens_q_array_batch = hub_subpaths_lens_q_array_batch[:, :, :options['kquery']]
		hub_subpaths_q_valid_array_batch = hub_subpaths_q_valid_array_batch[:, :, :options['kquery']]
		
		
		subpaths_s_array_batch = tf.reshape(subpaths_s_array_batch, [options['batch_size'], options['kshot']*options['nway'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		subpaths_lens_s_array_batch = tf.reshape(subpaths_lens_s_array_batch, [options['batch_size'], options['kshot']*options['nway'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		node_pairs_s_labels_array_batch = tf.reshape(node_pairs_s_labels_array_batch, [options['batch_size'], options['kshot']*options['nway'], options['nway'], options['kshot']])
		node_pairs_s_valid_array_batch = tf.reshape(node_pairs_s_valid_array_batch, [options['batch_size'], options['kshot']*options['nway'], options['nway'], options['kshot'], 2])
		subpaths_s_nodes_array_batch = tf.reshape(subpaths_s_nodes_array_batch, [options['batch_size'], (options['kshot']*options['nway'])*(options['kshot']*options['nway'])*options['subpaths_num_per_nodePair']*2*options['maxLen_subpath']])
		subpaths_s_nodes_lens_array_batch = tf.reshape(subpaths_s_nodes_lens_array_batch, [options['batch_size'], ])
		maxlen = tf.reduce_max(subpaths_s_nodes_lens_array_batch) 
		subpaths_s_nodes_array_batch = subpaths_s_nodes_array_batch[:, :maxlen]
		
		subpaths_q_array_batch = tf.reshape(subpaths_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		subpaths_lens_q_array_batch = tf.reshape(subpaths_lens_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot'], options['subpaths_num_per_nodePair']*2, options['maxLen_subpath']])
		node_pairs_q_labels_array_batch = tf.reshape(node_pairs_q_labels_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot']])
		node_pairs_q_valid_array_batch = tf.reshape(node_pairs_q_valid_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['nway'], options['kshot'], 2])
		maxlen = tf.reduce_max(subpaths_q_nodes_lens_array_batch) 
		subpaths_q_nodes_array_batch = subpaths_q_nodes_array_batch[:, :maxlen]
		
		subpaths_s_array_batch_neis_1_order = tf.nn.embedding_lookup(self.neisArray, subpaths_s_nodes_array_batch)
		indeces_1 = tf.random_uniform(shape=[self.max_degree_1_order,], maxval=self.max_degree, dtype=tf.int32) 
		neis_1_order = tf.gather(subpaths_s_array_batch_neis_1_order, indeces_1, axis=-1) 
		subpaths_s_array_batch_neis_1_order = tf.concat([tf.reshape(subpaths_s_nodes_array_batch, [tf.shape(subpaths_s_nodes_array_batch)[0], tf.shape(subpaths_s_nodes_array_batch)[1], 1]), neis_1_order[:, :, 1:]], axis=-1) 
		
		nodes_s_neis = subpaths_s_array_batch[:,:,0,0,0,0] 
		nodes_s_neis_list = [] 
		for i in range(options['batch_size']):
			nodes_s_neis_list.append(tf.nn.embedding_lookup(subpaths_s_nodes_array_batch[i], nodes_s_neis[i])[None, :]) 
		nodes_s_neis = tf.concat(nodes_s_neis_list, axis=0) 
		nodes_s_1_order = tf.nn.embedding_lookup(self.neisArray, nodes_s_neis) 
		nodes_s_1_order = tf.gather(nodes_s_1_order, indeces_1, axis=-1) 
		nodes_s_1_order = tf.concat([tf.reshape(nodes_s_neis, [tf.shape(nodes_s_neis)[0], tf.shape(nodes_s_neis)[1], 1]), nodes_s_1_order[:, :, 1:]], axis=-1) 
		nodes_s_2_order = tf.nn.embedding_lookup(self.neisArray, nodes_s_1_order) 
		indeces_2 = tf.random_uniform(shape=[self.max_degree_2_order,], maxval=self.max_degree, dtype=tf.int32) 
		neis_2_order = tf.gather(nodes_s_2_order, indeces_2, axis=-1) 
		nodes_s_2_order = tf.concat([tf.reshape(nodes_s_1_order, [tf.shape(nodes_s_1_order)[0], tf.shape(nodes_s_1_order)[1], tf.shape(nodes_s_1_order)[2], 1]), neis_2_order[:, :, :, 1:]], axis=-1) 
		
		
		subpaths_q_array_batch_neis_1_order = tf.nn.embedding_lookup(self.neisArray, subpaths_q_nodes_array_batch) 
		neis_1_order = tf.gather(subpaths_q_array_batch_neis_1_order, indeces_1, axis=-1) 
		subpaths_q_array_batch_neis_1_order = tf.concat([tf.reshape(subpaths_q_nodes_array_batch, [tf.shape(subpaths_q_nodes_array_batch)[0], tf.shape(subpaths_q_nodes_array_batch)[1], 1]), neis_1_order[:, :, 1:]], axis=-1) 
		
		nodes_q_neis = subpaths_q_array_batch[:,:,0,0,0,0] 
		nodes_q_neis_list = [] 
		for i in range(options['batch_size']):
			nodes_q_neis_list.append(tf.nn.embedding_lookup(subpaths_q_nodes_array_batch[i], nodes_q_neis[i])[None, :])
		nodes_q_neis = tf.concat(nodes_q_neis_list, axis=0) 
		nodes_q_1_order = tf.nn.embedding_lookup(self.neisArray, nodes_q_neis) 
		nodes_q_1_order = tf.gather(nodes_q_1_order, indeces_1, axis=-1) 
		nodes_q_1_order = tf.concat([tf.reshape(nodes_q_neis, [tf.shape(nodes_q_neis)[0], tf.shape(nodes_q_neis)[1], 1]), nodes_q_1_order[:, :, 1:]], axis=-1) 
		nodes_q_2_order = tf.nn.embedding_lookup(self.neisArray, nodes_q_1_order) 
		neis_2_order = tf.gather(nodes_q_2_order, indeces_2, axis=-1) 
		nodes_q_2_order = tf.concat([tf.reshape(nodes_q_1_order, [tf.shape(nodes_q_1_order)[0], tf.shape(nodes_q_1_order)[1], tf.shape(nodes_q_1_order)[2], 1]), neis_2_order[:, :, :, 1:]], axis=-1)
		
		node_pairs_s_labels_array_batch = tf.reduce_max(node_pairs_s_labels_array_batch, axis=-1) 
		node_pairs_q_labels_array_batch = tf.reduce_max(node_pairs_q_labels_array_batch, axis=-1) 
		
		hub_subpaths_s_array_batch = tf.reshape(hub_subpaths_s_array_batch, [options['batch_size'], options['kshot']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_lens_s_array_batch = tf.reshape(hub_subpaths_lens_s_array_batch, [options['batch_size'], options['kshot']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_s_valid_array_batch = tf.reshape(hub_subpaths_s_valid_array_batch, [options['batch_size'], options['kshot']*options['nway']])
		hub_subpaths_s_nodes_array_batch = tf.reshape(hub_subpaths_s_nodes_array_batch, [options['batch_size'], options['kshot']*options['nway'] * options['select_hub_nodes_num_per_node'] * options['subpaths_num_per_hubnode'] * options['maxLen_subpath_hub']])
		hub_subpaths_s_nodes_lens_array_batch = tf.reshape(hub_subpaths_s_nodes_lens_array_batch, [options['batch_size'], ])
		maxlen = tf.reduce_max(hub_subpaths_s_nodes_lens_array_batch) 
		hub_subpaths_s_nodes_array_batch = hub_subpaths_s_nodes_array_batch[:, :maxlen]
		
		hub_subpaths_q_array_batch = tf.reshape(hub_subpaths_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_lens_q_array_batch = tf.reshape(hub_subpaths_lens_q_array_batch, [options['batch_size'], options['kquery']*options['nway'], options['select_hub_nodes_num_per_node'], options['subpaths_num_per_hubnode'], options['maxLen_subpath_hub']])
		hub_subpaths_q_valid_array_batch = tf.reshape(hub_subpaths_q_valid_array_batch, [options['batch_size'], options['kquery']*options['nway']])
		maxlen = tf.reduce_max(hub_subpaths_q_nodes_lens_array_batch) 
		hub_subpaths_q_nodes_array_batch = hub_subpaths_q_nodes_array_batch[:, :maxlen]
		
		hub_subpaths_s_nodes_array_batch_neis_1_order = tf.nn.embedding_lookup(self.neisArray, hub_subpaths_s_nodes_array_batch) 
		neis_1_order = tf.gather(hub_subpaths_s_nodes_array_batch_neis_1_order, indeces_1, axis=-1) 
		hub_subpaths_s_nodes_array_batch_neis_1_order = tf.concat([tf.reshape(hub_subpaths_s_nodes_array_batch, [tf.shape(hub_subpaths_s_nodes_array_batch)[0], tf.shape(hub_subpaths_s_nodes_array_batch)[1], 1]), neis_1_order[:, :, 1:]], axis=-1) 
		
		hub_subpaths_q_nodes_array_batch_neis_1_order = tf.nn.embedding_lookup(self.neisArray, hub_subpaths_q_nodes_array_batch) 
		neis_1_order = tf.gather(hub_subpaths_q_nodes_array_batch_neis_1_order, indeces_1, axis=-1)
		hub_subpaths_q_nodes_array_batch_neis_1_order = tf.concat([tf.reshape(hub_subpaths_q_nodes_array_batch, [tf.shape(hub_subpaths_q_nodes_array_batch)[0], tf.shape(hub_subpaths_q_nodes_array_batch)[1], 1]), neis_1_order[:, :, 1:]], axis=-1) 
		
		return subpaths_s_array_batch, subpaths_lens_s_array_batch, node_pairs_s_valid_array_batch, node_pairs_s_labels_array_batch, subpaths_s_array_batch_neis_1_order, subpaths_s_nodes_lens_array_batch, nodes_s_2_order, subpaths_q_array_batch, subpaths_lens_q_array_batch, node_pairs_q_valid_array_batch, node_pairs_q_labels_array_batch, subpaths_q_array_batch_neis_1_order, subpaths_q_nodes_lens_array_batch, nodes_q_2_order, \
			hub_subpaths_s_array_batch, hub_subpaths_lens_s_array_batch, hub_subpaths_s_valid_array_batch, hub_subpaths_s_nodes_array_batch_neis_1_order, hub_subpaths_s_nodes_lens_array_batch, hub_subpaths_q_array_batch, hub_subpaths_lens_q_array_batch, hub_subpaths_q_valid_array_batch, hub_subpaths_q_nodes_array_batch_neis_1_order, hub_subpaths_q_nodes_lens_array_batch
		
		
		

