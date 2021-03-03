#encoding=utf-8
import os
import numpy as np
import argparse
import random
import tensorflow as tf

from amazon_reddit_data_generator import ar_DataGenerator
from email_data_generator import email_DataGenerator
from amazon_reddit_maml import ar_MAML
from email_maml import email_MAML
import processTools
import datetime, time
import scipy.sparse as sp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


def train(model, val_epoch, train_batch_num, val_batch_num, patience, saver, sess):
	"""
	the training
	"""
	support_loss_value = 0.0
	support_acc_value = 0.0
	prelosses, postlosses, preaccs, postaccs = [], [], [], []
	pre_preds, post_preds, pre_labels, post_labels = [],[],[],[]
	best_acc = 0
	curr_step = 0
	max_epoch = 0

	# train for meta_iteartion epoches
	for iteration in range(train_batch_num): 
		# this is the main op
		ops = [model.meta_op] 

		# add summary and print op
		if iteration % 200 == 0: 
			ops.extend([model.summ_op, 
			            model.query_losses[0], model.query_losses[-1],
			            model.query_accs[0], model.query_accs[-1],
			            model.query_preds[0], model.query_preds[-1], 
			            model.query_labels[0], model.query_labels[-1], 
			            
						model.support_loss,
						model.support_acc,
			            ]) 

		# run all ops
		result = sess.run(ops) 

		# summary 
		if iteration % 200 == 0:
			prelosses.append(result[2])
			postlosses.append(result[3])
			preaccs.append(result[4])
			postaccs.append(result[5])
			
			pre_preds.append(result[6])
			post_preds.append(result[7])
			pre_labels.append(result[8])
			post_labels.append(result[9])
			
			support_loss_value = result[10]
			support_acc_value = result[11]
			
			print(iteration, '\tsupport-loss: ', support_loss_value, ', support-acc: ', support_acc_value, '\tquery loss:', np.mean(prelosses), '=>', np.mean(postlosses),
			      '\t\tacc:', np.mean(preaccs), '=>', np.mean(postaccs))
			prelosses, postlosses, preaccs, postaccs = [], [], [], []

		if iteration % val_epoch == 0:
			acc1s, acc2s = [], []
			for _ in range(val_batch_num):
				acc1, acc2 = sess.run([model.test_query_accs[0], model.test_query_accs[-1]])
				acc1s.append(acc1)
				acc2s.append(acc2)
				
			print(' ++++++++++++++++ ',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
			acc1 = np.mean(acc1s) 
			acc2 = np.mean(acc2s) 
			print('>>>>\t\tValidation accs: [1] == ', acc1, ', [K] == ', acc2, ', best:', best_acc, '\t\t<<<<')

			acc = acc2
			if acc > best_acc: 
				saver.save(sess, os.path.join('ckpt', 'mini.mdl'))
				best_acc = acc
				curr_step = 0
				max_epoch = iteration
				print('saved into ckpt:', acc)
				print('------------------------------------------------------------------------------------')
			else: 
				curr_step += 1
				if curr_step == patience: 
					print('Early stop! -------------------------------------')
					print('Early stop model max epoch: ', max_epoch)
					print('Early stop model validation accuracy: ', best_acc)
					break
	print('------------------------------------------------------------------------------------')
	print('Finished the training ......')
	print('Stop model max epoch: ', max_epoch)
	print('Stop model validation accuracy: ', best_acc)
	print('------------------------------------------------------------------------------------')


def test(model, test_batch_num, sess):
	"""
	the test
	"""
	np.random.seed(1)
	random.seed(1)

	# repeat test accuracy for 600 times
	test_accs = []
	for i in range(test_batch_num): 
		if i % 100 == 1:
			print(i)
		ops = [model.test_support_acc]
		ops.extend(model.test_query_accs)
		result = sess.run(ops)
		test_accs.append(result) 

	test_accs = np.array(test_accs)
	means = np.mean(test_accs, 0)
	stds = np.std(test_accs, 0) 
	ci95 = 1.96 * stds / np.sqrt(test_batch_num)

	print('[support_t0, query_t0 - \t\t\tK] ')
	print('mean:', means)
	print('stds:', stds)
	print('ci95:', ci95)



def maml_main(
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
	    maxLen_subpath_hub):
	
	options = locals().copy() 
	
	training = ifTrain 
	
	# load dataset
	node_file = root_dir + 'graph.node'
	edge_file = root_dir + 'graph.edge'
	
	print('Start to load nodes information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
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
	print('Start to load edges information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
	if os.path.exists(root_dir + 'adj.npz'):
		adj = sp.load_npz(root_dir + 'adj.npz')
		neis_nums = np.load(root_dir + 'neis_nums.npy', allow_pickle=True)
	else:
		adj, neis_nums = processTools.readEdgesFromFile_sparse(edge_file, nodes_num) 
		sp.save_npz(root_dir + 'adj.npz', adj)
		np.save(root_dir + 'neis_nums.npy', neis_nums)
	
	print(neis_nums.shape)
	print('max number of neighbours == ', np.max(neis_nums), ' -----------------------')
	
	print('Start to load adj-self information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
	if os.path.exists(root_dir + 'adj_self.npz'):
		adj_self = sp.load_npz(root_dir + 'adj_self.npz')
		adj_self_nor = sp.load_npz(root_dir + 'adj_self_nor.npz')
	else:
		adj_self, adj_self_nor = processTools.preprocess_adj_normalization_sparse(adj)
		sp.save_npz(root_dir + 'adj_self.npz', adj_self)
		sp.save_npz(root_dir + 'adj_self_nor.npz', adj_self_nor)
	print('Start to load neighbours information, time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
	if os.path.exists(root_dir + 'neisArray.npy'):
		neisArray = np.load(root_dir + 'neisArray.npy', allow_pickle=True)
	else:
		neisArray = processTools.processNodeInfo_sparse_dense_maxDegree(adj_self, adj_self_nor, nodes_num, max_degree) 
		np.save(root_dir + 'neisArray.npy', neisArray)
	
	features_tensor = tf.convert_to_tensor(features)
	neighbours_tensor = tf.convert_to_tensor(neisArray)

	# 1. construct MAML model
	model = None
	if dataset in {'email'}: # for email
		model = email_MAML(options, features_tensor)
		db = email_DataGenerator(options, root_dir, label_nodes, neighbours_tensor) 
	else: # for others (amazon and reddit) 
		model = ar_MAML(options, features_tensor)
		db = ar_DataGenerator(options, root_dir, label_nodes, neighbours_tensor) 

	# load pipelines
	if  training:  
		preprocessed_data_save_file = options['root_dir'] + 'datasets-splits/trainTasksSplit_'+str(options['split_index'])+'-preprocessed.tfrecord'
		subpaths_s_array_batch, subpaths_lens_s_array_batch, node_pairs_s_valid_array_batch, node_pairs_s_labels_array_batch, subpaths_s_array_batch_neis_1_order, subpaths_s_nodes_lens_array_batch, nodes_s_2_order, subpaths_q_array_batch, subpaths_lens_q_array_batch, node_pairs_q_valid_array_batch, node_pairs_q_labels_array_batch, subpaths_q_array_batch_neis_1_order, subpaths_q_nodes_lens_array_batch, nodes_q_2_order, hub_subpaths_s_array_batch, hub_subpaths_lens_s_array_batch, hub_subpaths_s_valid_array_batch, hub_subpaths_s_nodes_array_batch_neis_1_order, hub_subpaths_s_nodes_lens_array_batch, hub_subpaths_q_array_batch, hub_subpaths_lens_q_array_batch, hub_subpaths_q_valid_array_batch, hub_subpaths_q_nodes_array_batch_neis_1_order, hub_subpaths_q_nodes_lens_array_batch = db.make_data_tensor(options, preprocessed_data_save_file) 
		
		preprocessed_data_save_file_val = options['root_dir'] + 'datasets-splits/valTasksSplit_'+str(options['split_index'])+'-preprocessed.tfrecord'
		subpaths_s_array_batch_val, subpaths_lens_s_array_batch_val, node_pairs_s_valid_array_batch_val, node_pairs_s_labels_array_batch_val, subpaths_s_array_batch_neis_1_order_val, subpaths_s_nodes_lens_array_batch_val, nodes_s_2_order_val, subpaths_q_array_batch_val, subpaths_lens_q_array_batch_val, node_pairs_q_valid_array_batch_val, node_pairs_q_labels_array_batch_val, subpaths_q_array_batch_neis_1_order_val, subpaths_q_nodes_lens_array_batch_val, nodes_q_2_order_val, hub_subpaths_s_array_batch_val, hub_subpaths_lens_s_array_batch_val, hub_subpaths_s_valid_array_batch_val, hub_subpaths_s_nodes_array_batch_neis_1_order_val, hub_subpaths_s_nodes_lens_array_batch_val, hub_subpaths_q_array_batch_val, hub_subpaths_lens_q_array_batch_val, hub_subpaths_q_valid_array_batch_val, hub_subpaths_q_nodes_array_batch_neis_1_order_val, hub_subpaths_q_nodes_lens_array_batch_val = db.make_data_tensor(options, preprocessed_data_save_file_val) 
		
	preprocessed_data_save_file_test = options['root_dir'] + 'datasets-splits/testTasksSplit_'+str(options['split_index'])+'-preprocessed.tfrecord'
	subpaths_s_array_batch_test, subpaths_lens_s_array_batch_test, node_pairs_s_valid_array_batch_test, node_pairs_s_labels_array_batch_test, subpaths_s_array_batch_neis_1_order_test, subpaths_s_nodes_lens_array_batch_test, nodes_s_2_order_test, subpaths_q_array_batch_test, subpaths_lens_q_array_batch_test, node_pairs_q_valid_array_batch_test, node_pairs_q_labels_array_batch_test, subpaths_q_array_batch_neis_1_order_test, subpaths_q_nodes_lens_array_batch_test, nodes_q_2_order_test, hub_subpaths_s_array_batch_test, hub_subpaths_lens_s_array_batch_test, hub_subpaths_s_valid_array_batch_test, hub_subpaths_s_nodes_array_batch_neis_1_order_test, hub_subpaths_s_nodes_lens_array_batch_test, hub_subpaths_q_array_batch_test, hub_subpaths_lens_q_array_batch_test, hub_subpaths_q_valid_array_batch_test, hub_subpaths_q_nodes_array_batch_neis_1_order_test, hub_subpaths_q_nodes_lens_array_batch_test = db.make_data_tensor(options, preprocessed_data_save_file_test) 
		

	# construct metatrain_, metaval_ and metatest_
	if  training: 
		# training
		model.build(subpaths_s_array_batch, subpaths_lens_s_array_batch, node_pairs_s_valid_array_batch, node_pairs_s_labels_array_batch, subpaths_s_array_batch_neis_1_order, subpaths_s_nodes_lens_array_batch, nodes_s_2_order, subpaths_q_array_batch, subpaths_lens_q_array_batch, node_pairs_q_valid_array_batch, node_pairs_q_labels_array_batch, subpaths_q_array_batch_neis_1_order, subpaths_q_nodes_lens_array_batch, nodes_q_2_order, hub_subpaths_s_array_batch, hub_subpaths_lens_s_array_batch, hub_subpaths_s_valid_array_batch, hub_subpaths_s_nodes_array_batch_neis_1_order, hub_subpaths_s_nodes_lens_array_batch, hub_subpaths_q_array_batch, hub_subpaths_lens_q_array_batch, hub_subpaths_q_valid_array_batch, hub_subpaths_q_nodes_array_batch_neis_1_order, hub_subpaths_q_nodes_lens_array_batch,
				update_times, batch_size, mode='train')
		# validation
		model.build(subpaths_s_array_batch_val, subpaths_lens_s_array_batch_val, node_pairs_s_valid_array_batch_val, node_pairs_s_labels_array_batch_val, subpaths_s_array_batch_neis_1_order_val, subpaths_s_nodes_lens_array_batch_val, nodes_s_2_order_val, subpaths_q_array_batch_val, subpaths_lens_q_array_batch_val, node_pairs_q_valid_array_batch_val, node_pairs_q_labels_array_batch_val, subpaths_q_array_batch_neis_1_order_val, subpaths_q_nodes_lens_array_batch_val, nodes_q_2_order_val, hub_subpaths_s_array_batch_val, hub_subpaths_lens_s_array_batch_val, hub_subpaths_s_valid_array_batch_val, hub_subpaths_s_nodes_array_batch_neis_1_order_val, hub_subpaths_s_nodes_lens_array_batch_val, hub_subpaths_q_array_batch_val, hub_subpaths_lens_q_array_batch_val, hub_subpaths_q_valid_array_batch_val, hub_subpaths_q_nodes_array_batch_neis_1_order_val, hub_subpaths_q_nodes_lens_array_batch_val,
				update_times, batch_size, mode='val')
	else: # test
		model.build(subpaths_s_array_batch_test, subpaths_lens_s_array_batch_test, node_pairs_s_valid_array_batch_test, node_pairs_s_labels_array_batch_test, subpaths_s_array_batch_neis_1_order_test, subpaths_s_nodes_lens_array_batch_test, nodes_s_2_order_test, subpaths_q_array_batch_test, subpaths_lens_q_array_batch_test, node_pairs_q_valid_array_batch_test, node_pairs_q_labels_array_batch_test, subpaths_q_array_batch_neis_1_order_test, subpaths_q_nodes_lens_array_batch_test, nodes_q_2_order_test, hub_subpaths_s_array_batch_test, hub_subpaths_lens_s_array_batch_test, hub_subpaths_s_valid_array_batch_test, hub_subpaths_s_nodes_array_batch_neis_1_order_test, hub_subpaths_s_nodes_lens_array_batch_test, hub_subpaths_q_array_batch_test, hub_subpaths_lens_q_array_batch_test, hub_subpaths_q_valid_array_batch_test, hub_subpaths_q_nodes_array_batch_neis_1_order_test, hub_subpaths_q_nodes_lens_array_batch_test,
				update_times, batch_size, mode='test')
	model.summ_op = tf.summary.merge_all() 
	
	all_vars = filter(lambda x: 'meta_optim' not in x.name, tf.trainable_variables()) 
	for p in all_vars: 
		print(p)

	# config
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

	tf.global_variables_initializer().run()
	tf.train.start_queue_runners() 

	# restore for test
	if not training and os.path.exists(os.path.join('ckpt', 'checkpoint')): 
		model_file = tf.train.latest_checkpoint('ckpt')
		print("Restoring model weights from ", model_file)
		saver.restore(sess, model_file)

	start_time=time.time()
	print('Start time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
	if training: # train the model
		train(model, val_epoch, train_batch_num, val_batch_num, patience, saver, sess)
	else: # test the model
		test(model, test_batch_num, sess)
	end_time=time.time()
	print('End time ==', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
	print('All cost time =', end_time-start_time,' s')


