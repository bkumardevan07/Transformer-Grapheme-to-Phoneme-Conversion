import tensorflow as tf

class Dataset:
    def __init__(self, data, batch_size, isTraining= True, bucket_id= None, drop_remainder= True):
        self.epoch = 0
        self.data = data
        self.batch_size = batch_size
        self.bucket_id = bucket_id
        self._random = random
        self.isTraining = isTraining
        self.buckets = data_utils._buckets
        output_types = (tf.int32, tf.int64, tf.int32, tf.int64)
        padded_shapes = ([-1], 1, [-1], 1)
        dataset = tf.data.Dataset.from_generator(lambda: self.get_batch(),
                output_types=output_types)
        dataset = dataset.padded_batch(batch_size,
                padded_shapes=padded_shapes,
                drop_remainder=drop_remainder)
        dataset = dataset.prefetch(2)
        self.dataset = dataset
        self.data_iter = iter(dataset.repeat(-1))


    def next_batch(self):
        return next(self.data_iter)

    def all_batches(self):
        return iter(self.dataset)

    def get_batch(self):
        """Prepare minibatch from given data.
        Args:
            data: A list of datapoints (all from same bucket).
            bucket_id: Bucket ID of data. This is irrevelant for training but
                for evaluation we can limit the padding by the bucket size.
        Returns:
            Batched input IDs, input sequence length, output IDs & output
            sequence length
        """
        self.epoch += 1
        self._random.shuffle(self.data)
        if not self.isTraining:
            # During evaluation the bucket size limits the amount of padding
            _, decoder_size = self.buckets[self.bucket_id]

        encoder_inputs, decoder_inputs = [], []
        batch_size = len(self.data)

        seq_len = np.zeros((batch_size), dtype=np.int64)
        seq_len_target = np.zeros((batch_size), dtype=np.int64)

        for i, sample in enumerate(self.data):
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

        for i, sample in enumerate(self.data):
            encoder_input, decoder_input = sample
            # Encoder inputs are padded and then reversed.
            encoder_pad_size = max_len_source - len(encoder_input)
            encoder_pad = [data_utils.PAD_ID] * encoder_pad_size
            # Encoder input is reversed - https://arxiv.org/abs/1409.3215

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
        seq_len = np.expand_dims(seq_len,-1).astype(np.int64)
        seq_len_target = np.expand_dims(seq_len_target,-1).astype(np.int64)
        i = -1
        while i<batch_size-1:
            i += 1
            yield tf.convert_to_tensor(encoder_inputs[i,:], dtype= tf.int32), tf.convert_to_tensor([seq_len[i,0]], dtype= tf.int64), tf.convert_to_tensor(decoder_inputs[i,:],dtype= tf.int32),tf.convert_to_tensor([seq_len_target[i,0]], dtype= tf.int64)

