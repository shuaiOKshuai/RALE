#encoding=utf-8
import numpy as np
import tensorflow as tf
import self_attentionModel

class email_MAML:
	def __init__(self, options, features_tensor):
		self.features_tensor = features_tensor
		self.inner_dim = options['inner_dim']
		self.features_num = options['features_num']
		self.nway = options['nway']
		self.dropout = options['dropout']
		self.meta_lr = options['meta_lr']
		self.train_lr = options['inner_train_lr']
		self.subpaths_max_len = options['maxLen_subpath'] 
		self.hub_subpaths_max_len = options['maxLen_subpath_hub'] 
		self.subpaths_num_per_nodePair = options['subpaths_num_per_nodePair'] 
		self.alpha = options['alpha']
		self.clip_by_norm = options['clip_by_norm']
		self.emb_avg_value = 1.0 / options['inner_dim']
		self.num_heads = options['num_heads']
		self.l2_coef = options['l2_coef']
		
		self.max_degree_1_order = options['max_degree_1_order']
		self.max_degree_2_order = options['max_degree_2_order']

		self.positionInfo_unit_subpath = self_attentionModel.positional_encoding(1, options['maxLen_subpath'], options['inner_dim']) 
		self.positionInfo_unit_subpath_hub = self_attentionModel.positional_encoding(1, options['maxLen_subpath_hub'], options['inner_dim']) 
		
		print('meta-lr:', self.meta_lr, 'train-lr:', self.train_lr )


	def build(self, support_subpaths_array, support_subpaths_lens, support_node_pair_valids, support_node_pair_labels, support_nodes_neis_1, support_nodes_lens, support_nodes_2_order, query_subpaths_array, query_subpaths_lens, query_node_pair_valids, query_node_pair_labels, query_nodes_neis_1, query_nodes_lens, query_nodes_2_order, hub_subpaths_s_array, hub_subpaths_lens_s_array, hub_subpaths_s_valid_array, hub_subpaths_s_nodes_1_order, hub_subpaths_s_nodes_lens, hub_subpaths_q_array, hub_subpaths_lens_q_array, hub_subpaths_q_valid_array, hub_subpaths_q_nodes_array_1_order, hub_subpaths_q_nodes_lens, K, meta_batchsz, mode='train'):
		"""
		build the model
		"""
		self.weights = self.init_variables() 
		training = True if mode is 'train' else False 

		def meta_task(input):
			"""
			build model for each meta task
			"""
			s_subpaths_array, s_subpaths_lens, s_node_pair_valids, s_node_pair_labels, s_subpaths_neis_1, s_nodes_lens, s_nodes_2_order, q_subpaths_array, q_subpaths_lens, q_node_pair_valids, q_node_pair_labels, q_subpaths_neis_1, q_nodes_lens, q_nodes_2_order, h_subpaths_s_array, h_subpaths_lens_s, h_subpaths_s_valid, h_subpaths_s_nodes_1_order, h_subpaths_s_nodes_lens, h_subpaths_q, h_subpaths_lens_q, h_subpaths_q_valid, h_subpaths_q_nodes_array_1_order, h_subpaths_q_nodes_lens = input
			query_preds, query_losses, query_accs, query_labels = [], [], [], []

			support_pred = self.forward(s_subpaths_array, s_subpaths_lens, s_node_pair_valids, s_subpaths_neis_1, s_nodes_lens, s_nodes_2_order, s_nodes_2_order, h_subpaths_s_array, h_subpaths_lens_s, h_subpaths_s_valid, h_subpaths_s_nodes_1_order, h_subpaths_s_nodes_lens, self.weights, training) 
			support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=s_node_pair_labels)
			support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, axis=1), axis=1),
			                                             tf.argmax(s_node_pair_labels, axis=1))
			grads = tf.gradients(support_loss, list(self.weights.values()))
			gvs = dict(zip(self.weights.keys(), grads))
			fast_weights = dict(zip(self.weights.keys(), [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
			query_pred = self.forward(q_subpaths_array, q_subpaths_lens, q_node_pair_valids, q_subpaths_neis_1, q_nodes_lens, q_nodes_2_order, s_nodes_2_order, h_subpaths_q, h_subpaths_lens_q, h_subpaths_q_valid, h_subpaths_q_nodes_array_1_order, h_subpaths_q_nodes_lens, fast_weights, training) 
			query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=q_node_pair_labels) 
			if training:
				query_loss += tf.add_n([tf.nn.l2_loss(v) for v in self.weights.values()]) * self.l2_coef
			query_preds.append(query_pred)
			query_losses.append(query_loss)
			query_labels.append(q_node_pair_labels)

			for iter in range(1, K):
				loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.forward(s_subpaths_array, s_subpaths_lens, s_node_pair_valids, s_subpaths_neis_1, s_nodes_lens, s_nodes_2_order, s_nodes_2_order, h_subpaths_s_array, h_subpaths_lens_s, h_subpaths_s_valid, h_subpaths_s_nodes_1_order, h_subpaths_s_nodes_lens, fast_weights, training),
				                                               labels=s_node_pair_labels)
				grads = tf.gradients(loss, list(fast_weights.values())) 
				gvs = dict(zip(fast_weights.keys(), grads))
				fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key] for key in fast_weights.keys()]))
				query_pred = self.forward(q_subpaths_array, q_subpaths_lens, q_node_pair_valids, q_subpaths_neis_1, q_nodes_lens, q_nodes_2_order, s_nodes_2_order, h_subpaths_q, h_subpaths_lens_q, h_subpaths_q_valid, h_subpaths_q_nodes_array_1_order, h_subpaths_q_nodes_lens, fast_weights, training) 
				query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=q_node_pair_labels)
				if training:
					query_loss += tf.add_n([tf.nn.l2_loss(v) for v in self.weights.values()]) * self.l2_coef
				query_preds.append(query_pred)
				query_losses.append(query_loss)
				query_labels.append(q_node_pair_labels)

			for i in range(K):
				query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], axis=1), axis=1),
					                                            tf.argmax(q_node_pair_labels, axis=1)))
			result = [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs, query_labels]

			return result
		
		out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
		result = tf.map_fn(meta_task, elems=(support_subpaths_array, support_subpaths_lens, support_node_pair_valids, support_node_pair_labels, support_nodes_neis_1, support_nodes_lens, support_nodes_2_order, query_subpaths_array, query_subpaths_lens, query_node_pair_valids, query_node_pair_labels, query_nodes_neis_1, query_nodes_lens, query_nodes_2_order, hub_subpaths_s_array, hub_subpaths_lens_s_array, hub_subpaths_s_valid_array, hub_subpaths_s_nodes_1_order, hub_subpaths_s_nodes_lens, hub_subpaths_q_array, hub_subpaths_lens_q_array, hub_subpaths_q_valid_array, hub_subpaths_q_nodes_array_1_order, hub_subpaths_q_nodes_lens),
		                   dtype=out_dtype, parallel_iterations=meta_batchsz, name='map_fn')
		support_pred_tasks, support_loss_tasks, support_acc_tasks, \
			query_preds_tasks, query_losses_tasks, query_accs_tasks, query_labels_tasks = result

		self.query_preds = query_preds_tasks 
		self.query_labels = query_labels_tasks
		if mode is 'train':
			self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
			self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
			                                        for j in range(K)]
			self.support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
			self.query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
			                                        for j in range(K)]

			optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
			gvs = optimizer.compute_gradients(self.query_losses[-1]) 
			gvs = [(tf.clip_by_norm(grad, self.clip_by_norm), var) for grad, var in gvs] 
			self.meta_op = optimizer.apply_gradients(gvs) 

		else: 
			self.test_support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
			self.test_query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
			                                        for j in range(K)]
			self.test_support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
			self.test_query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
			                                        for j in range(K)]

		tf.summary.scalar(mode + '：support loss', support_loss)
		tf.summary.scalar(mode + '：support acc', support_acc)
		for j in range(K):
			tf.summary.scalar(mode + '：query loss, step ' + str(j + 1), query_losses[j])
			tf.summary.scalar(mode + '：query acc, step ' + str(j + 1), query_accs[j])



	def init_variables(self):
		variables = {}
		he_initializer = tf.contrib.layers.variance_scaling_initializer(seed=123)
		with tf.variable_scope('MAML', reuse= tf.AUTO_REUSE): 

			variables['W_2'] = tf.get_variable('W_2', [2 * self.features_num, self.inner_dim], initializer=he_initializer)
			variables['W_1'] = tf.get_variable('W_1', [2 * self.inner_dim, self.inner_dim], initializer=he_initializer)
			
			variables['W_Q'] = tf.get_variable('W_Q', [self.inner_dim, self.inner_dim], initializer=he_initializer)
			variables['W_K'] = tf.get_variable('W_K', [self.inner_dim, self.inner_dim], initializer=he_initializer)
			variables['W_V'] = tf.get_variable('W_V', [self.inner_dim, self.inner_dim], initializer=he_initializer)
			
			variables['b_Q'] = tf.get_variable('b_Q', [self.inner_dim, ], initializer=he_initializer)
			variables['b_K'] = tf.get_variable('b_K', [self.inner_dim, ], initializer=he_initializer)
			variables['b_V'] = tf.get_variable('b_V', [self.inner_dim, ], initializer=he_initializer)
			
			variables['multi_head_W_1'] = tf.get_variable('multi_head_W_1', [self.inner_dim, self.inner_dim], initializer=he_initializer)
			variables['multi_head_W_2'] = tf.get_variable('multi_head_W_2', [self.inner_dim, self.inner_dim], initializer=he_initializer)
			variables['multi_head_b_1'] = tf.get_variable('multi_head_b_1', [self.inner_dim, ], initializer=he_initializer)
			variables['multi_head_b_2'] = tf.get_variable('multi_head_b_2', [self.inner_dim, ], initializer=he_initializer)
			
			variables['W_atten_support'] = tf.get_variable('W_atten_support', [self.inner_dim, self.inner_dim], initializer=he_initializer)
			variables['b_atten_support'] = tf.get_variable('b_atten_support', [self.inner_dim, ], initializer=he_initializer)
			variables['eta_atten_support'] = tf.get_variable('eta_atten_support', [self.inner_dim, ], initializer=he_initializer)
			
			variables['W_atten_hub'] = tf.get_variable('W_atten_hub', [self.inner_dim, self.inner_dim], initializer=he_initializer)
			variables['b_atten_hub'] = tf.get_variable('b_atten_hub', [self.inner_dim, ], initializer=he_initializer)
			variables['eta_atten_hub'] = tf.get_variable('eta_atten_hub', [self.inner_dim, ], initializer=he_initializer)
			
			variables['W_classify'] = tf.get_variable('W_classify', [self.inner_dim*3, self.nway], initializer=he_initializer)
			
			self.gcnVariables = {'W_2'}
			self.rnnVariables = {'W_Q', 'W_K', 'W_V', 'b_Q', 'b_K', 'b_V', 'multi_head_W_1', 'multi_head_W_2', 'multi_head_b_1', 'multi_head_b_2', 'beta', 'gamma'}
			self.attentionVariables = {'W_atten_support', 'b_atten_support', 'eta_atten_support', 'W_atten_hub', 'b_atten_hub', 'eta_atten_hub'}
			self.commonVariables = {'W_classify'}
			
		return variables
	
	
	def GCN_layer_1(self, input, features, W=None, dropout=0.0, training=True):
		self_2 = tf.nn.embedding_lookup(features, input[:,0]) 
		neis_2 = tf.nn.embedding_lookup(features, input[:,1:]) 
		neis_2 = tf.reduce_mean(neis_2, axis=-2) 
		neis_2 = tf.concat([self_2, neis_2], axis=-1) 
		ret = tf.matmul(neis_2, W) 
		if training:
			ret = tf.nn.dropout(ret, 1.0 - dropout) 
		ret = tf.nn.elu(ret)
		ret = tf.nn.l2_normalize(ret, axis=-1)
		return ret
	
	
	def GCN_layer_2(self, input, features, W=None, layer=2, dropout=0.0, training=True):
		if layer==2:
			self_2 = tf.nn.embedding_lookup(features, input[:,:,0]) 
			neis_2 = tf.nn.embedding_lookup(features, input[:,:,1:]) 
			neis_2 = tf.reduce_mean(neis_2, axis=-2) 
			neis_2 = tf.concat([self_2, neis_2], axis=-1) 
			tmp1 = tf.reshape(neis_2, [-1, tf.shape(neis_2)[-1]])
			tmp2 = tf.matmul(tmp1, W) 
			ret = tf.reshape(tmp2, [tf.shape(neis_2)[0], tf.shape(neis_2)[1], tf.shape(W)[1]]) 
			if training:
				ret = tf.nn.dropout(ret, 1.0 - dropout) 
			ret = tf.nn.elu(ret)
			ret = tf.nn.l2_normalize(ret, axis=-1)
		if layer==1: 
			self_1 = input[:,0] 
			neis_1 = input[:, 1:] 
			neis_1 = tf.reduce_mean(neis_1, axis=-2) 
			tmp1 = tf.concat([self_1, neis_1], axis=-1) 
			tmp2 = tf.matmul(tmp1, W) 
			ret = tf.reshape(tmp2, [tf.shape(neis_1)[0], tf.shape(W)[1]]) 
			if training:
				ret = tf.nn.dropout(ret, 1.0 - dropout) 
			ret = tf.nn.elu(ret)
			ret = tf.nn.l2_normalize(ret, axis=-1)
		return ret
	
	
	def position_aware_self_attention_cal(self, paths_emb, masks, variables, is_hub = False, causality=False, dropout_rate=0., training=True):
		paths_emb_reshape = tf.reshape(paths_emb, [-1, tf.shape(paths_emb)[-2], tf.shape(paths_emb)[-1]]) 
		masks_reshape = tf.reshape(masks, [-1, tf.shape(masks)[-1]]) 
		paths_emb_reshape *= self.inner_dim**0.5
		position_info_unit_subpath = None
		if is_hub: 
			position_info_unit_subpath = self.positionInfo_unit_subpath_hub
		else:
			position_info_unit_subpath = self.positionInfo_unit_subpath
		position_info = tf.tile(position_info_unit_subpath, [tf.shape(paths_emb_reshape)[0], 1, 1]) 
		paths_emb_reshape += position_info 
		
		paths_emb_tmp = tf.reshape(paths_emb_reshape, [-1, tf.shape(paths_emb_reshape)[-1]]) 
		Q = tf.matmul(paths_emb_tmp, variables['W_Q']) + variables['b_Q'] 
		K = tf.matmul(paths_emb_tmp, variables['W_K']) + variables['b_K']
		V = tf.matmul(paths_emb_tmp, variables['W_V']) + variables['b_V']
		Q_reshape = tf.reshape(Q, [tf.shape(paths_emb_reshape)[0], tf.shape(paths_emb_reshape)[1], tf.shape(Q)[-1]]) 
		K_reshape = tf.reshape(K, [tf.shape(paths_emb_reshape)[0], tf.shape(paths_emb_reshape)[1], tf.shape(K)[-1]])
		V_reshape = tf.reshape(V, [tf.shape(paths_emb_reshape)[0], tf.shape(paths_emb_reshape)[1], tf.shape(V)[-1]])
		
		attention_result = self_attentionModel.multihead_attention(Q_reshape, K_reshape, V_reshape, masks_reshape, paths_emb_reshape, variables, self.num_heads, dropout_rate, training)
		result = tf.reshape(attention_result, tf.shape(paths_emb)) 
		
		neg_masks = 1.0 - masks 
		padding_num = -2 ** 6 + 1
		result_m = result + tf.expand_dims(neg_masks * padding_num, axis=-1)
		embedding = tf.reduce_max(result_m, axis=-2)
		
		return embedding
	
			
	def forward(self, subpaths_array, subpaths_lens, node_pair_valids, neis_2_order, nodes_len, node_s_2_order, support_neis_2, 
			hub_subpaths_array, hub_subpaths_lens, hub_subpaths_valids, hub_subpaths_nodes_2_order, hub_subpaths_nodes_len,
			variables, training):
		dropout = self.dropout
		
		neis_2_order = neis_2_order[:nodes_len] 
		neis_1_order = self.GCN_layer_2(neis_2_order, self.features_tensor, W=variables['W_2'], layer=2, dropout=dropout, training=training)
		h_gcn = self.GCN_layer_2(neis_1_order, self.features_tensor, W=variables['W_1'], layer=1, dropout=dropout, training=training) 
		
		subpaths_array_embs = tf.nn.embedding_lookup(h_gcn, subpaths_array) 

		h_rnn= self.position_aware_self_attention_cal(subpaths_array_embs, subpaths_lens, variables, is_hub=False, dropout_rate=dropout, training=training)

		subpaths_lens_weight = tf.reduce_sum(subpaths_lens, axis=-1) 
		subpaths_lens_weight = self.exp_weight(subpaths_lens_weight, self.alpha) 
		h_rnn = h_rnn * subpaths_lens_weight[:,:,:,:,None] 

		oneDirection = tf.reduce_mean(h_rnn[:,:,:,:self.subpaths_num_per_nodePair,:], axis=-2)
		anotherDirection = tf.reduce_mean(h_rnn[:,:,:,self.subpaths_num_per_nodePair:,:], axis=-2) 
		combine = tf.concat([(oneDirection * node_pair_valids[:,:,:,0][:,:,:,None])[:,:,:,None,:], (anotherDirection * node_pair_valids[:,:,:,1][:,:,:,None])[:,:,:,None,:]], axis=-2)
		combine_sum, node_pair_valids_sum = self.mean_pool_weight(combine, node_pair_valids)
		nodePairEmbs = combine_sum / node_pair_valids_sum[:,:,:,None]
		
		node_pair_valids_1 = tf.reduce_max(node_pair_valids, axis=-1) 
		nodePairEmbs_weight = nodePairEmbs * node_pair_valids_1[:,:,:,None]
		nodePairEmbs_weight_sum, node_pair_valids_1_sum = self.mean_pool_weight(nodePairEmbs_weight, node_pair_valids_1)
		connectionsEmb = nodePairEmbs_weight_sum / node_pair_valids_1_sum[:,:,None] 
		
		hub_neis_2_order = hub_subpaths_nodes_2_order[:hub_subpaths_nodes_len] 
		hub_neis_1_order = self.GCN_layer_2(hub_neis_2_order, self.features_tensor, W=variables['W_2'], layer=2, dropout=dropout, training=training)
		hub_h_gcn = self.GCN_layer_2(hub_neis_1_order, self.features_tensor, W=variables['W_1'], layer=1, dropout=dropout, training=training) 
		
		hub_h_gcn_1_layer = hub_neis_1_order[:,0,:]
		hub_subpaths_array_embs_1_layer = tf.nn.embedding_lookup(hub_h_gcn_1_layer, hub_subpaths_array)
		center_node_embs = hub_subpaths_array_embs_1_layer[:,0,0,0,:]
		
		hub_subpaths_array_embs = tf.nn.embedding_lookup(hub_h_gcn, hub_subpaths_array) 
		hub_h_rnn= self.position_aware_self_attention_cal(hub_subpaths_array_embs, hub_subpaths_lens, variables, is_hub=True, dropout_rate=dropout, training=training)
		
		hub_node_embs = hub_subpaths_array_embs[:,:,0,-1,:] 
		
		hub_subpaths_lens_weight = tf.reduce_sum(hub_subpaths_lens, axis=-1) 
		hub_subpaths_lens_weight = self.exp_weight(hub_subpaths_lens_weight, self.alpha) 
		hub_h_rnn = hub_h_rnn * hub_subpaths_lens_weight[:,:,:,None] 
		hub_paths_embs = tf.reduce_mean(hub_h_rnn, axis=-2) 
		
		connectionsEmb_reshape = tf.reshape(connectionsEmb, [tf.shape(connectionsEmb)[0]*tf.shape(connectionsEmb)[1], tf.shape(connectionsEmb)[2]]) 
		tmp1 = tf.nn.leaky_relu(tf.matmul(connectionsEmb_reshape, variables['W_atten_support']) + variables['b_atten_support']) 
		tmp2 = tf.reduce_sum(tf.multiply(tmp1, variables['eta_atten_support']), axis=-1) 
		tmp3 = tf.reshape(tmp2, [tf.shape(connectionsEmb)[0], tf.shape(connectionsEmb)[1]]) 
		if training:
			connectionsEmb = tf.nn.dropout(connectionsEmb, 1.0 - dropout) 
		node_pair_valids_mask = tf.reduce_max(node_pair_valids, axis=-1)
		node_pair_valids_mask = tf.reduce_max(node_pair_valids_mask, axis=-1) 
		tmp4 = self.addMask(tmp3, node_pair_valids_mask)
		
		connectionsWeights = tf.nn.softmax(tmp4, axis=-1)
		support_nodeEmb = tf.reduce_sum(tf.multiply(connectionsEmb, connectionsWeights[:,:,None]), axis=-2) 
		
		hub_connectionsEmb_reshape = tf.reshape(hub_paths_embs, [tf.shape(hub_paths_embs)[0]*tf.shape(hub_paths_embs)[1], tf.shape(hub_paths_embs)[2]]) 
		tmp1 = tf.nn.leaky_relu(tf.matmul(hub_connectionsEmb_reshape, variables['W_atten_hub']) + variables['b_atten_hub']) 
		tmp2 = tf.reduce_sum(tf.multiply(tmp1, variables['eta_atten_hub']), axis=-1) 
		tmp3 = tf.reshape(tmp2, [tf.shape(hub_paths_embs)[0], tf.shape(hub_paths_embs)[1]]) 
		if training:
			hub_paths_embs = tf.nn.dropout(hub_paths_embs, 1.0 - dropout) 
		hub_connectionsWeights = tf.nn.softmax(tmp3, axis=-1) 
 		
		hub_pathEmb = tf.reduce_sum(tf.multiply(hub_paths_embs, hub_connectionsWeights[:,:,None]), axis=-2) 
		hub_pathEmb = tf.multiply(hub_pathEmb, hub_subpaths_valids[:,None])
		
		nodeEmbs = tf.concat([support_nodeEmb, hub_pathEmb, center_node_embs], axis=-1) 
		classPreb = tf.matmul(nodeEmbs, variables['W_classify']) 
		classPreb = tf.nn.elu(classPreb)
		
		return classPreb
		

	
	def exp_weight(self, M, alpha):
		return tf.exp(-1. * alpha * M)
		
	
	def mean_pool_weight(self, M, weights):
		M_sum = tf.reduce_sum(M, axis=-2)
		weights_sum = tf.reduce_sum(weights, axis=-1)
		weights_sum += 1e-6
		return M_sum, weights_sum
		
		
	def addMask(self, inputs, key_masks=None):
		"""Masks paddings on keys or queries to inputs
		inputs: 2d tensor. (batch, nway)
		key_masks: 2d tensor. (batch, nway)
		"""
		key_masks = 1.0 - key_masks
		padding_num = -2 ** 32 + 1
		outputs = inputs + key_masks * padding_num 
		
		return outputs
