import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]  # (1, max_position_encoding, d_model)
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q,k,v,mask=None):
  """
  Args:
  q: shape == (...,seq_len_q, depth)
  k: shape == (...,seq_len_k, depth)
  v: shape == (...,seq_len_v, depth)
  mask: shape == (...,seq_lenk, seq_len_v), default - None

  Returns:
    output, attention_weights
  """
  matmul_op = tf.matmul(q,k,transpose_b= True)  # (..., seq_len_q, seq_len_k)
  dk = tf.cast(tf.shape(q)[-1], tf.float32)
  scaled_op = matmul_op/tf.math.sqrt(dk)
  
  if mask is not None:
    scaled_op += mask*(-1e9)
  
  softmax_op = tf.nn.softmax(scaled_op, axis= -1)
  output = tf.matmul(softmax_op, v)

  return output, softmax_op

def create_encoder_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq,0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask

def create_mel_padding_mask(seq):
    #seq = tf.reduce_sum(tf.math.abs(seq), axis=-1)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, y, x)
