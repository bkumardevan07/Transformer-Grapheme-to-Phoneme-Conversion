import sys
import numpy as np
import tensorflow as tf
from model.transformer_utils import create_encoder_padding_mask, create_mel_padding_mask, create_look_ahead_mask
#from preprocessing.text import Pipeline
from model.layers import Encoder, Decoder
from utils.losses import model_loss, crossentropy_loss
import data_utils 

class Transformer(tf.keras.models.Model):

    def __init__(self,
            encoder_model_dimension : int,
            decoder_model_dimension : int,
            encoder_num_heads : int,
            decoder_num_heads : int,
            encoder_num_layers : int,
            decoder_num_layers : int,
            encoder_maximum_position_encoding : int,
            decoder_maximum_position_encoding : int,
            encoder_feed_forward_dimension : int,
            decoder_feed_forward_dimension : int,
            dropout_rate : int,
            encoder_vocab_size : int,
            decoder_vocab_size : int,
            debug : bool,
            diagonal_bandwidth_b : int,
            diagonal_rate_regul_coeff : float,
            layernorm : bool,
            Ldc : bool,
            buckets: list,
            training: bool,
            epoch_path: str,
            **kwargs
            ):
    
      super(Transformer, self).__init__(**kwargs)
      self.Ldc = Ldc
      self.buckets = buckets
      self.b = diagonal_bandwidth_b
      self.lamda = diagonal_rate_regul_coeff
      self.isTraining = training
      #if self.isTraining:
      #    try:
      #        print("\n\n")
      #        with open(epoch_path, 'r') as f:
      #            self.epoch =  int(f.readlines()[0])
      #            print("Successfully loaded epoch count.")
      #    except:
      #        print("Could not load epoch path : %s" % epoch_path)
      #        print("Creating a file with default epoch = 1")
      #        with open(epoch_path, 'w+') as f:
      #            f.write(str(1))
      #        print("Successfully created epoch file.")
      #        self.epoch = 1
      #self.epoch_path = epoch_path

      self.encoder = Encoder(d_model = encoder_model_dimension,
                             num_heads = encoder_num_heads,
                             num_layers = encoder_num_layers,
                             dff = encoder_feed_forward_dimension,
                             input_vocab_size = encoder_vocab_size, #change
                             maximum_position_encoding = encoder_maximum_position_encoding,
                             layernorm= layernorm,
                             rate = dropout_rate,
                             name = 'Encoder')

      self.decoder = Decoder(d_model = decoder_model_dimension,
                             num_heads = decoder_num_heads,
                             dff = decoder_feed_forward_dimension,
                             maximum_position_encoding = decoder_maximum_position_encoding,
                             layernorm= layernorm,
                             output_vocab_size = decoder_vocab_size, #change
                             num_layers = decoder_num_layers,
                             rate = dropout_rate,
                             name = 'Decoder')
      
      self.decoder_layers = decoder_num_layers
      self.regul_coeff = diagonal_rate_regul_coeff
      self.loss_bandwidth = diagonal_bandwidth_b
      self.decoder_model_dim = decoder_model_dimension
      self.linear = tf.keras.layers.Dense(decoder_vocab_size, name= 'linear')
      ## remaining ##
      self.training_input_signature = [
        tf.TensorSpec(shape=(None, None), dtype = tf.int32),
        tf.TensorSpec(shape=(None), dtype = tf.int64),
        tf.TensorSpec(shape=(None, None), dtype = tf.int32),
        tf.TensorSpec(shape=(None), dtype = tf.int64),

      ]

      self.forward_input_signature = [
        tf.TensorSpec(shape=(None, None), dtype= tf.int32),
        tf.TensorSpec(shape=(None, 1), dtype= tf.int32),
        #tf.TensorSpec(shape=(None, None, mel_channels), dtype = tf.float32)
      ]

      self.encoder_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)
      ]

      self.decoder_signature = [
        tf.TensorSpec(shape=(None, None, encoder_model_dimension), dtype = tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
      ]

      self.debug = debug
      self._apply_all_signatures()

    @property
    def step(self):
      return int(self.optimizer.iterations)

    def _apply_signature(self, function, signature):
      if self.debug:
        return function
      else:
        return tf.function(input_signature=signature)(function)
    def increment_epoch(self):
        self.epoch += 1
        with open(self.epoch_path,'w') as f:
            f.write(str(self.epoch))

    def call(self):
        self._apply_all_signature()

    def _apply_all_signatures(self):
      #self.forward = self._apply_signature(self._forward, self.forward_input_signature)
      self.train_step = self._apply_signature(self._train_step, self.training_input_signature)
      self.val_step = self._apply_signature(self._val_step, self.training_input_signature)
      self.forward_encoder = self._apply_signature(self._forward_encoder, self.encoder_signature)
      self.forward_decoder = self._apply_signature(self._forward_decoder, self.decoder_signature)

    def _call_encoder(self, inputs, training):
      padding_mask = create_encoder_padding_mask(inputs)
      enc_input = inputs 
      enc_output, attn_weights = self.encoder(enc_input,
                                              training= training,
                                              mask = padding_mask)
      return enc_output, padding_mask, attn_weights

    def _call_decoder(self, dec_input, enc_output, enc_padding_mask, training):
      dec_target_padding_mask = create_mel_padding_mask(dec_input)
      look_ahead_mask = create_look_ahead_mask(tf.shape(dec_input)[1])
      combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
      dec_output, attention_weights = self.decoder(x = dec_input,
                                                   enc_output = enc_output,
                                                   training = training,
                                                   look_ahead_mask = combined_mask,
                                                   padding_mask = enc_padding_mask
                                                   )
      linear = self.linear(dec_output) 
      model_out = {'linear':linear, 'decoder_attention':attention_weights, 'decoder_output':dec_output}
      return model_out

    def _forward(self, inp, output): # not getting used
      model_out = self.__call__(inputs = inp,
                                speaker_input = sp_id,
                                targets = output,
                                training = False) 
    
      return model_out

    def _forward_encoder(self, inputs):
      return self._call_encoder(inputs, training = False)
    
    def _forward_decoder(self, encoder_output, targets, encoder_padding_mask):
      return self._call_decoder(targets, encoder_output, encoder_padding_mask, training = False) 

    def _gta_forward(self, encoder_inputs, seq_len, decoder_inputs, seq_len_target, training):
      tar_inp = decoder_inputs[:,:-1]
      tar_real = decoder_inputs[:,1:]

      seq_len = int(tf.shape(tar_inp)[1]) 

      with tf.GradientTape() as tape:
        model_out = self.__call__(inputs= encoder_inputs,
                                  targets = tar_inp,
                                  training = training)

        loss = model_loss(tar_real, 
                          model_out['linear']
                         )
        model_out.update({'loss' : loss})
        model_out.update({'target': tar_inp}) 
        return model_out, tape

    def _train_step(self, encoder_inputs, seq_len, _decoder_inputs, seq_len_target): 
        model_out, tape = self._gta_forward(encoder_inputs, seq_len, _decoder_inputs, seq_len_target, training= True)
        gradients = tape.gradient(model_out['loss'], self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return model_out

    def _val_step(self, encoder_inputs, seq_len, decoder_inputs, seq_len_target):
        model_out, _ = self._gta_forward(encoder_inputs, seq_len, decoder_inputs, seq_len_target, training= False)
        return model_out

    def _compile(self, optimizer):
        self.compile(loss = crossentropy_loss,
                            optimizer = optimizer)

    def call(self, inputs, targets, training): 
        encoder_output, encoder_padding_mask, encoder_attention = self._call_encoder(inputs, training=training) 
        model_out = self._call_decoder(targets, encoder_output, encoder_padding_mask, training= training)
        model_out.update({'encoder_attention' : encoder_attention})
        return model_out

    def _compute_centroid(self, attention, win_c, s):
        attention = tf.reduce_mean(attention, axis=1)
        if s >= attention.shape[-1]:
            return win_c
        C_s = tf.cast(tf.reduce_sum(attention, 1).numpy()[:, s], tf.int32)
        if win_c + 3 * self.r >= C_s:
            return win_c + 3*self.r
        return win_c

    def predict(self, encoder_inp, max_length = 20, verbose = True):
        #print(inp.shape)
        start_vec = tf.convert_to_tensor(tf.constant([data_utils.GO_ID]), dtype= tf.int32)
        encoder_inp = tf.cast(tf.expand_dims(encoder_inp, 0), tf.int32) 
        output = tf.cast(tf.expand_dims(start_vec, 0), tf.int32)
        output_concat = tf.cast(tf.expand_dims(start_vec, 0), tf.int32) 
        out_dict = {}
        encoder_output, encoder_padding_mask, encoder_attention  = self.forward_encoder(encoder_inp) 
        
        for i in range(max_length + 1):
          model_out = self.forward_decoder(encoder_output, output, encoder_padding_mask)

          predictions = model_out['linear'][:,-1:,:]
          prediction_id = tf.cast(tf.argmax(predictions, axis = -1), dtype= tf.int32)
          #concat_vec = tf.convert_to_tensor(tf.constant([]), dtype= tf.int32)
          concat_vec = tf.cast(prediction_id, tf.int32)
          output = tf.concat([output, concat_vec], axis=-1)  
          output_concat = tf.concat([tf.cast(output_concat, tf.int32), concat_vec],
                                    axis=-1) ####### UNCLEAR -SELF.R ##########
          out_dict = {'linear': output_concat,
                      'decoder_attention': model_out['decoder_attention'],
                      'encoder_attention': encoder_attention}
          if verbose:
            sys.stdout.write(f'\rpred word phoneme: {i}')
          if prediction_id == data_utils.EOS_ID:
            if verbose:
              print('Stopping')
            break
        
        return out_dict

    def set_constants(self, learning_rate: float= None):
        if learning_rate is not None:
           self.optimizer.lr.assign(learning_rate)

    def get_batch(self, data, bucket_id=None):
        """Prepare minibatch from given data.
        Args:
            data: A list of datapoints (all from same bucket).
            bucket_id: Bucket ID of data. This is irrevelant for training but
                for evaluation we can limit the padding by the bucket size.
        Returns:
            Batched input IDs, input sequence length, output IDs & output
            sequence length
        """
        if not self.isTraining:
            # During evaluation the bucket size limits the amount of padding
            _, decoder_size = self.buckets[bucket_id]

        encoder_inputs, decoder_inputs = [], []
        batch_size = len(data)

        seq_len = np.zeros((batch_size), dtype=np.int64)
        seq_len_target = np.zeros((batch_size), dtype=np.int64)

        for i, sample in enumerate(data):
            encoder_input, decoder_input = sample
            seq_len[i] = len(encoder_input)
            if not self.isTraining:
                seq_len_target[i] = decoder_size
            else:
                # 1 is added to output sequence length because the EOS token is
                # crucial to "halt" the decoder. Consider it the punctuation
                # mark of a English sentence. Both are necessary.
                seq_len_target[i] = len(decoder_input) + 1

        # Maximum input and output length which limit the padding till them
        max_len_source = max(seq_len)
        max_len_target = max(seq_len_target)

        for i, sample in enumerate(data):
            encoder_input, decoder_input = sample
            # Encoder inputs are padded and then reversed.
            encoder_pad_size = max_len_source - len(encoder_input)
            encoder_pad = [data_utils.PAD_ID] * encoder_pad_size
            # Encoder input is reversed - https://arxiv.org/abs/1409.3215
            #encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad) 

            encoder_inputs.append(encoder_input + encoder_pad) # removed reversed
            # 1 is added to decoder_input because GO_ID is considered a part of
            # decoder input. While EOS_ID is also added, it's really used by
            # the target tensor (self.tensor) in the core code above.
            decoder_pad_size = max_len_target - (len(decoder_input) + 1)
            decoder_inputs.append([data_utils.GO_ID] +
                                  decoder_input +
                                  [data_utils.EOS_ID] +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Both the id sequences are made time major via transpose
        encoder_inputs = np.asarray(encoder_inputs, dtype=np.int32)# changed transpose
        decoder_inputs = np.asarray(decoder_inputs, dtype=np.int32)# same
        return tf.convert_to_tensor(encoder_inputs, dtype= tf.int32), tf.convert_to_tensor(seq_len, dtype= tf.int64), tf.convert_to_tensor(decoder_inputs,dtype= tf.int32),tf.convert_to_tensor(seq_len_target, dtype= tf.int64)


