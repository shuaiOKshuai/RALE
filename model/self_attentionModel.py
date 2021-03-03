#encoding=utf-8
import numpy as np
import tensorflow as tf


def multihead_attention(Q, K, V, key_masks, queries, variables,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    d_model = queries.get_shape().as_list()[-1]
    
    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) 
    outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, queries, causality, dropout_rate, training)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) 
    outputs += queries
    outputs = tf.nn.l2_normalize(outputs, axis=-1) 
    
    outputs_tmp = tf.reshape(outputs, [-1, tf.shape(outputs)[-1]]) 
    tmp1 = tf.nn.relu(tf.matmul(outputs_tmp, variables['multi_head_W_1']) + variables['multi_head_b_1']) 
    tmp2 = tf.matmul(tmp1, variables['multi_head_W_2']) + variables['multi_head_b_2']
    outputs_tmp2 = tmp2 + outputs_tmp
    outputs_tmp3 = tf.reshape(outputs_tmp2, [tf.shape(outputs)[0], tf.shape(outputs)[1], tf.shape(variables['multi_head_W_2'])[1]])
    outputs = tf.nn.l2_normalize(outputs_tmp3, axis=-1) 
 
    return outputs
    

def scaled_dot_product_attention(Q, K, V, key_masks, paths_emb_reshape,
                                 causality=False, dropout_rate=0.,
                                 training=True
                                 ):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
#     with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    d_k = Q.get_shape().as_list()[-1] 
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) 
 
    outputs /= d_k ** 0.5
     
    outputs = mask(outputs, key_masks=key_masks)
 
    outputs = tf.nn.softmax(outputs)
 
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
    outputs = tf.matmul(outputs, V) 
     
    return outputs



def mask(inputs, key_masks=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)。
    type: string. "key" | "future"， 
    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    key_masks = 1.0 - key_masks
    padding_num = -2 ** 32 + 1
    key_masks = tf.to_float(key_masks) 
    key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) 
    key_masks = tf.expand_dims(key_masks, axis = 1)  
    outputs = inputs + key_masks * padding_num 
    return outputs


def positional_encoding(paths_num, path_len, emb_dim):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    N = paths_num
    T = path_len
    E = emb_dim
    maxlen = path_len
    position_ind = tf.tile(tf.expand_dims(tf.range(T), axis = 0), [N, 1]) 

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
        for pos in range(maxlen)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  
    position_enc = tf.convert_to_tensor(position_enc, tf.float32) 

    outputs = tf.nn.embedding_lookup(position_enc, position_ind) 

    return tf.to_float(outputs)


def ln(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs