import tensorflow as tf
import numpy as np

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg


#def create_diagonal_attention_mask(T,S,b):
#    np_arr = np.zeros((int(T),int(S)))
#    k = tf.math.divide(S,T, name= 'Attention_rate_slope')
#    for t in range(int(T)):
#      l = max(0, int(k*t-b)) 
#      r = min(int(k*t+b),int(S-1)) 
#      np_arr[t,l:r+1] = 1  
#
#    mask = my_func(np_arr)
#    return tf.transpose(mask)  # (mel_seq_len, text_seq_len) 

## Below was working earlier 6 Nov
#def create_diagonal_attention_mask(T,S,b):
#    np_arr = np.zeros((int(T),int(S)))
#    k = tf.math.divide(S,T, name= 'Attention_rate_slope')
#    for t in range(int(T)):
#      l = max(0, int(k*t-b)) 
#      r = min(int(k*t+b),int(S-1)) 
#      np_arr[t,l:r+1] = 1  
#
#    mask = my_func(np_arr)
#    return tf.transpose(mask) # (mel_seq_len, text_seq_len)


def create_diagonal_attention_mask(T,S,b,mel_len,text_len):
    np_arr = np.zeros((int(T),int(S)))
    diag_mask = []
    for seq in tf.range(tf.shape(mel_len)[0]):
        k = tf.math.divide(int(mel_len[seq]), int(text_len[seq]), name= 'Attention_rate_slope')
        for t in range(int(text_len)):
          l = max(0, int(k*t-b)) 
          r = min(int(k*t+b),int(mel_len-1)) 
          np_arr[t,l:r+1] = 1  

        mask = tf.transpose(my_func(np_arr))
        mask= tf.stack([mask, mask]) # expected_shape == (2, S, T)
        diag_mask.append(mask)
    diag_mask = tf.stack(diag_mask) # expected_shape == (B,2,S,T)

    return diag_mask # (mel_seq_len, text_seq_len)


def apply_mask(attention, mask):
  return tf.math.multiply(attention, mask)

#Below was working earlier 6 Nov
#def compute_diagonal_rate(decoder_attention, diagonal_mask):
#  masked_attention = tf.math.multiply(decoder_attention, diagonal_mask)
#  x = tf.math.reduce_sum(masked_attention, [2,3])
#  x = tf.math.divide(x, tf.cast(tf.shape(decoder_attention)[2], tf.float32))
#  return tf.math.reduce_mean(x)


def compute_diagonal_rate(decoder_attention, diagonal_mask, num_layers):
  r = []
  for i in tf.range(num_layers): 
      masked_attention = tf.math.multiply(decoder_attention[f'decoder_layer{i+1}_block2'], diagonal_mask)
      x = tf.math.reduce_sum(masked_attention, [2,3])
      x = tf.math.divide(x, tf.cast(tf.shape(decoder_attention[f'decoder_layers{num_layers}_block2'])[2], tf.float32))
      r.append(tf.math.reduce_mean(x, [1]))
  r = my_func(r) 
  return tf.reduce_mean(r)

def masked_mean_squared_error(targets: tf.Tensor, logits: tf.Tensor, decoder_attention: tf.Tensor, Ldc: bool, mel_len: tf.Tensor, phon_len: tf.Tensor, b: int= 50, regul_coeff: int= 0.01, num_layers: int= 4) -> tf.Tensor:
    mse = tf.keras.losses.MeanSquaredError()
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int32)
    mask = tf.reduce_max(mask, axis=-1)
    # diagonal attention rate ...
    # decoder_attention : (batch_size, num_heads, mel_seq_len, text_seq_len) 
    #6Nov#S = tf.cast(tf.shape(decoder_attention)[2], dtype= tf.float32) #change it 
    #6Nov#T = tf.cast(tf.shape(decoder_attention)[-1], dtype= tf.float32) # change it
    
    #S = tf.cast(tf.shape(decoder_attention[f'decoder_layer{num_layers}_block2'])[2], dtype= tf.float32) #change it 
    #T = tf.cast(tf.shape(decoder_attention[f'decoder_layer{num_layers}_block2'])[-1], dtype= tf.float32) # change it
    #mel_len = tf.cast(mel_len, dtype= tf.float32)
    #phon_len = tf.cast(phon_len, dtype= tf.float32)
    #diagonal_mask = create_diagonal_attention_mask(T,S,b)
    #diagonal_mask = tf.py_function(func=create_diagonal_attention_mask, inp= [T,S,tf.cast(b, tf.float32),mel_len,phon_len], Tout= tf.float32) 
    r = 0 #compute_diagonal_rate(decoder_attention, diagonal_mask, num_layers)

    mse_loss = mse(targets, logits, sample_weight=mask) 
    if Ldc:
        L_dc = -r
        total_mse_loss = mse_loss + regul_coeff * L_dc
    else:
        total_mse_loss = mse_loss
    return total_mse_loss, mse_loss, r


def new_scaled_crossentropy(index=2, scaling=1.0):
    """
    Returns masked crossentropy with extra scaling:
    Scales the loss for given stop_index by stp_scaling
    """
    def masked_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        padding_mask = tf.math.equal(targets, 0)
        padding_mask = tf.math.logical_not(padding_mask)
        padding_mask = tf.cast(padding_mask,dtype= tf.float32)
        stop_mask = tf.math.equal(targets, index)
        stop_mask = tf.cast(stop_mask, dtype= tf.float32) * (scaling - 1.)
        combined_mask = padding_mask + stop_mask
        loss = crossentropy(targets, logits, sample_weight= combined_mask)
        return loss
    return masked_crossentropy


def crossentropy_loss(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction= 'none')
    mask = tf.math.logical_not(tf.math.equal(targets, 0)) 
    loss_ = crossentropy(targets, logits)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def model_loss(target, pred):
    loss = crossentropy_loss(target, pred)

    return loss
