import os
import argparse
import traceback
import subprocess
import shutil
import random

import tensorflow as tf
import numpy as np
from tqdm import trange
from datetime import date
#from bunch import bunchify
import editdistance as ed
import time
import ruamel.yaml
import operator

import data_utils
from model.models import *
from utils.logging import SummaryManager
from dataset import Dataset

import tqdm

FLAGS = object()
_buckets = data_utils._buckets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True

np.random.seed(42)  
tf.random.set_seed(42)

# dynamically allocate GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[1],'GPU')
    #for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpus, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs', '\nUsing GPU : ', gpus[1])
  except Exception:
    traceback.print_exc()

def evaluate(checkpoint_path: str = None, on_test= True, verbose= True):
    model = get_model(False)
    model._compile(optimizer= tf.keras.optimizers.Adam(FLAGS['learning_rate'], beta_1= 0.9, beta_2= 0.98, epsilon= 1e-9) )
    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, FLAGS['weights_dir'],
                                        max_to_keep=None)

    if checkpoint_path:
      ckpt.restore(checkpoint_path)
      if verbose:
        print(f'restored weights from {checkpoint_path} at step {model.step}')
    else:
      if manager.latest_checkpoint is None:
        print(f"WARNING: could not find weights file. Trying to load from \n {FLAGS['weights_dir']}.")
        print('Edit data_config.yaml to point at the right log directory.')
        ckpt.restore(manager.latest_checkpoint)
      if verbose:
        print(f'restored weights from {manager.latest_checkpoint} at step {model.step}')
    if on_test: 
        test_set = data_utils.read_and_bucket_data(
                    os.path.join(FLAGS['data_dir'], "test.pkl"))
        wer, per, total_words = calc_levenshtein_loss(model, test_set)
        print('TOTAL_WORDS: ', total_words,'\nWER: ', wer, '\nPER: ', per)
    else:
        inp = str(input('\nENTER WORD : ')).rstrip().lstrip()
        vocab_phon_to_id, phon_list = initialize_vocabulary(os.path.join(FLAGS['data_dir'], "vocab.phone"))
        encoder_inp = []
        for phon in inp.split():
            encoder_inp.append(vocab_phon_to_id[char])
        encoder_inp = tf.constant(encoder_inp)
        encoder_inp = tf.convert_to_tensor(encoder_inp, dtype=tf.int32)
        
        out_dict = model.predict(encoder_inp, max_length= FLAGS['max_prediction_len'])
        output = out_dict['linear'],numpy()
        output = np.argmax(output, axis= -1)
        output = np.squeeze(output, axis=0).tolist()

        if data_utils.EOS_ID in output:
            output = output[:output.index(data_utils.EOS_ID)]

        vocab_id_to_phon = dict((v,k) for k,v in vocab_phon_to_id.items())

        phoneme = []
        for _id in output:
            phoneme.append(vocab_id_to_phon[_id])

        print ('Predicted Phoneme : ', phoneme)

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', default= os.getcwd())
    parser.add_argument('--session_name', dest='session_name', default="default_run", type=str)
    parser.add_argument("--eval",
                        default=False, action="store_true",
                        help="Evaluate using the last saved model")
    parser.add_argument("--data_dir", default="./CHK_DATA", type= str)
    parser.add_argument("-sv_file", "--source_vocab_file",
                        default="vocab.char", type=str,
                        help="Vocabulary file for characters")
    parser.add_argument("-tv_file", "--target_vocab_file",
                        default="vocab.phone", type=str,
                        help="Vocabulary file for phonemes")
    parser.add_argument("--restart", dest= 'restart_training', default= False, action= "store_true",
                        help="Restart session from scratch.")
    parser.add_argument("-run_id", "--run_id",
                        default=0, type=int,
                        dest= 'run_id',
                        help="Run ID parameter to distinguish diff. runs")

    args = parser.parse_args()
    arg_dict = vars(args)
    
    #if not arg_dict['eval']:
    if True:
        config_path = arg_dict['config_path'] 
        with open(str(config_path +'/'+ 'data_config.yaml'), 'rb') as data_yaml:
            data_config = ruamel.yaml.YAML().load(data_yaml)
        with open(str(config_path +'/'+ f'model_config.yaml'), 'rb') as model_yaml:
            model_config = ruamel.yaml.YAML().load(model_yaml)
        
        arg_dict.update(model_config)
        arg_dict.update(data_config)

        try:
            if not os.path.exists(arg_dict['train_base_dir']):
                os.makedirs(arg_dict['train_base_dir'])
        except:
            print ("Could not create base folder %s that contains model "
                   "directories for different runs" % arg_dict['train_base_dir'])
            traceback.print_exc()
            sys.exit(1)    
        
        try:
            train_dir =arg_dict['session_name'] or str(date.today().strftime("%b-%d-%Y")+f"-runs-{args.run_id}")
            arg_dict['session_name'] = train_dir 
            arg_dict['train_dir'] = os.path.join(arg_dict['train_base_dir'], train_dir)    

            if arg_dict['restart_training']:
                if arg_dict['train_dir']:
                    shutil.rmtree(arg_dict['train_dir'])#, ignore_errors= True)
                    print()

            if not os.path.exists(arg_dict['train_dir']):
                os.makedirs(arg_dict['train_dir'])
        except:
            print ("Could not create model directory %s to store checkpoints"
                   % (arg_dict['train_dir']))
            traceback.print_exc()
            sys.exit(1)


        parameter_file = 'parameters.txt'
        sorted_args = sorted(arg_dict.items(), key=operator.itemgetter(0))

        with open(os.path.join(arg_dict['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                g.write(arg + " :\t" + str(arg_val) + "\n")
    
    source_vocab_path = os.path.join(arg_dict['data_dir'],
                                     arg_dict['source_vocab_file'])
    target_vocab_path = os.path.join(arg_dict['data_dir'],
                                     arg_dict['target_vocab_file'])
    source_vocab, _ = data_utils.initialize_vocabulary(source_vocab_path)
    target_vocab, _ = data_utils.initialize_vocabulary(target_vocab_path)
    
    arg_dict['source_vocab_size'] = len(source_vocab)
    arg_dict['target_vocab_size'] = len(target_vocab)

    return arg_dict

def get_model(train):
    return Transformer(training = train,
                       epoch_path = FLAGS['epoch_path'],
                       buckets= data_utils._buckets,
                       encoder_model_dimension = FLAGS['encoder_model_dimension'],
                       decoder_model_dimension = FLAGS['decoder_model_dimension'],
                       encoder_num_heads = FLAGS['encoder_num_heads'],
                       decoder_num_heads = FLAGS['decoder_num_heads'],
                       encoder_num_layers = FLAGS['encoder_num_layers'],
                       decoder_num_layers = FLAGS['decoder_num_layers'],
                       encoder_maximum_position_encoding = FLAGS['encoder_maximum_position_encoding'],
                       decoder_maximum_position_encoding = FLAGS['decoder_maximum_position_encoding'],
                       encoder_feed_forward_dimension = FLAGS['encoder_feed_forward_dimension'],
                       decoder_feed_forward_dimension = FLAGS['decoder_feed_forward_dimension'],
                       dropout_rate = FLAGS['dropout_rate'],
                       encoder_vocab_size = FLAGS['encoder_vocab_size'],
                       decoder_vocab_size = FLAGS['decoder_vocab_size'],
                       debug = FLAGS['debug'],
                       diagonal_bandwidth_b = FLAGS['diagonal_bandwidth_b'],
                       diagonal_rate_regul_coeff = FLAGS['diagonal_rate_regul_coeff'],
                       layernorm = FLAGS['layernorm'],
                       Ldc = FLAGS['Ldc']
                       )

def calc_levenshtein_loss(model, eval_set, summary_manager=None, step=None):
    """Calculate the actual loss function for G2P.
    Args:
        model: Seq2SeqModel instance
        sess: Tensorflow session with the model compuation graph
        eval_set: Bucketed evaluation set
    Returns:
        wer: Word Error Rate
        per: Phoneme Error Rate
    """
    total_words = 0
    total_phonemes = 0
    wer = 0
    per = 0
    edit_distances = []
    #sample_pts = np.random.randint(10, size= 5)
    gd_sampled = []
    op_sampled = []
    inp_sampled = []
    count = 0

    for bucket_id in range(len(data_utils._buckets)):
        cur_data = eval_set[bucket_id]
        val_loss = 0.
        val_batch_step = 1.
        for batch_offset in range(0, len(cur_data), FLAGS['batch_size']):# batch size check remaining..## ##done##
            batch = cur_data[batch_offset:batch_offset + FLAGS['batch_size']]
            num_instances = len(batch)
            
            inp_ids = [inst[0] for inst in batch]
            gt_ids = [inst[1] for inst in batch]
            encoder_inputs, seq_len, decoder_inputs, seq_len_target = \
                model.get_batch(batch, bucket_id=bucket_id) ## remaining ## #done#
            # Run the model to get output_logits of shape BxTx|V|
            model_out = model.val_step(encoder_inputs, seq_len,
                                       decoder_inputs, seq_len_target) ## validate ...remaining ## ##done##
            val_loss += model_out['loss']
            val_batch_step += 1.

            output_logits = model_out['linear'].numpy()
            # This is a greedy decoder and output is just argmax at each timestep
            outputs = np.argmax(output_logits, axis=-1) ## remaining ## ##check in future##              ## axis check remaining here ##
            #print(output_logits[0,:,:], output_logits.shape)
            #print(outputs[0,:], outputs.shape)
            # Reshape the output and make it batch major via transpose
            #outputs = np.reshape(outputs, (max(seq_len_target), num_instances)).T ###commented###
            for idx in range(num_instances):
                cur_output = list(outputs[idx])
                if data_utils.EOS_ID in cur_output:
                    cur_output = cur_output[:cur_output.index(data_utils.EOS_ID)]

                gt = gt_ids[idx]
                inp = inp_ids[idx]
                # Calculate the edit distance from ground truth
                distance = ed.eval(gt, cur_output)
                edit_distances.append((inp_ids[idx], distance, len(gt)))
            if count<5:
                if np.random.randint(2,size=1)==1:
                    count+=1
                    gd_sampled.append(gt)
                    op_sampled.append(cur_output)
                    inp_sampled.append(inp)

        if summary_manager:
            summary_manager.display_attention_heads(model_out, tag='ValidationAttentionHeads')

    edit_distances.sort()
    if summary_manager is not None:

        vocab_phon_to_id, phon_list = data_utils.initialize_vocabulary(os.path.join(FLAGS['data_dir'], "vocab.phone"))
        vocab_char_to_id, char_list = data_utils.initialize_vocabulary(os.path.join(FLAGS['data_dir'], "vocab.char"))

        vocab_id_to_phon = dict((v,k) for k,v in vocab_phon_to_id.items())
        vocab_id_to_char = dict((v,k) for k,v in vocab_char_to_id.items())

        gd_phon = []
        op_phon = []
        inp_char = []
        for i in range(len(gd_sampled)):
            phon = []
            for _id in gd_sampled[i]:
                phon.append(vocab_id_to_phon[_id])
            gd_phon.append(phon)
            phon = []

            for _id in op_sampled[i]:
                phon.append(vocab_id_to_phon[_id])
            op_phon.append(phon)

            char = []        
            for _id in inp_sampled[i]:
                char.append(vocab_id_to_char[_id])
            inp_char.append(char)
        gd = gd_phon
        op = op_phon
        inp = inp_char
        summary_manager.display_text(inp,gd,op,tag='Validation-text', step=step)

    # Aggregate the edit distances for each word
    word_to_edit = {}
    for edit_distance in edit_distances:
        word, distance, num_phonemes = edit_distance
        word = tuple(word) #here not numpy 
        if word in word_to_edit:
            word_to_edit[word].append((distance, num_phonemes))
        else:
            word_to_edit[word] = [(distance, num_phonemes)]

    total_words = len(word_to_edit)
    for word in word_to_edit:
        # Pick the ground truth that's closest to output since their can be
        # multiple pronunciations
        distance, num_phonemes = min(word_to_edit[word])
        if distance != 0:
            wer += 1
            per += distance
        total_phonemes += num_phonemes

    try:
        wer = float(wer)/float(total_words)
    except ZeroDivisionError:
        print ("0 words in evaluation set")
        wer = 1.0
    try:
        per = float(per)/float(total_phonemes)
    except ZeroDivisionError:
        print ("0 phones in evaluation set")
        per = 1.0
    if not summary_manager:        
        return wer, per, total_words
    else:
        return wer, per, val_loss/val_batch_step

def _print_dict_values(values, key_name, level= 0, tab_size = 2):
    tab = level * tab_size * ' '
    print(tab + '-', key_name, ':', values)

def _print_dictionary(dictionary, recursion_level = 0):
    for key in dictionary.keys():
        if isinstance(key, dict):
            recursion_level += 1
            _print_dictionary(dictionary[key], recursion_level)
        else:
            _print_dict_values(dictionary[key], key_name = key, level = recursion_level)
            
def print_config():
    print('\nCONFIGURATION', FLAGS['session_name'])
    _print_dictionary(FLAGS)

def train():
    print_config()
    model = get_model(train=True)
    lr = FLAGS['learning_rate']
    model._compile(optimizer= tf.keras.optimizers.Adam(lr, beta_1= 0.9, beta_2= 0.98, epsilon= 1e-9))



    if not os.path.isfile(FLAGS['learning_rate_path']):
        try:
            with open(FLAGS['learning_rate_path'], 'w+') as f:
                f.write(str(FLAGS['learning_rate']))
        except:
            print("Could not create learning rate backup %s" % FLAGS['learning_rate_path'])
            traceback.print_exc()
            sys.exit(1)

    if not os.path.isfile(FLAGS['epoch_path']):
        try:
            with open(FLAGS['epoch_path'], 'w+') as f:
                f.write(str(0))
        except:
            print("Could not create epoch count file %s"% FLAGS['epoch_path'])
            traceback.print_exc()
            sys.exit(1)

    try:
        if not os.path.exists(FLAGS['weights_dir']):
            os.makedirs(FLAGS['weights_dir'])
    except:
        print ("Could not weights folder %s that contains model "
                "directories for trained weights " % FLAGS['weights_dir'])
        traceback.print_exc()
        sys.exit(1)    

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                 optimizer=model.optimizer,
                                 net=model) 
    manager = tf.train.CheckpointManager(checkpoint, str(FLAGS['weights_dir']),   #### remaining  #### ##done##
                                 max_to_keep=FLAGS['keep_n_weights'],
                                 keep_checkpoint_every_n_hours=FLAGS['keep_checkpoint_every_n_hours'])
    summary_manager = SummaryManager(model=model, log_dir=FLAGS['train_dir'], config=FLAGS)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f'\nresuming training from step {model.step} ({manager.latest_checkpoint})')
        try:
            with open(FLAGS['learning_rate_path'], 'r') as f:
                lr = float(f.readlines()[0])
                print("Successfully loaded learning rate.")
        except:
            print("Could not load learning rate file %s"% FLAGS['learning_rate_path'])
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f'\nStarting training from scratch ...')
    if FLAGS['debug'] is True:
        print('\nWARNING: DEBUG is set to True. Training in eager mode.')

    print('\nTRAINING...')

    train_data = data_utils.read_and_bucket_data(
            os.path.join(FLAGS['data_dir'], "train.pkl"))
    dev_set = data_utils.read_and_bucket_data(os.path.join(FLAGS['data_dir'], "dev.pkl"))

    train_dataset = Dataset(train_data[0], FLAGS['batch_size'], isTraining= True, bucket_id= None, drop_remainder= True)

    temp = []
    
    val_wer_window = []
    window_size = 3
    validation_improvement_threshold = FLAGS['valid_thresh'] # per_threshold for validation improvement

    step_time, loss = 0.0, 0.0
    #previous_losses = []
    steps_done = model.step
    val_losses = []

    if steps_done > 0: ## remaining## ##done##
            # The model saved would have wer and per better than 1.0
            best_wer, _ = calc_levenshtein_loss(model, dev_set)  ## remaining## ##done##
    else:
            best_wer = 1.0

    # _ = train_dataset.next_batch()
    epoch_id = train_dataset.epoch #remaining#
    t = trange(model.step, FLAGS['max_steps'], leave=True) ## implement model.epoch #### replace 3 with model.epoch# ##done##
    c = epoch_id
    steps = 0
    for _ in t:
      #current_temp = subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader'])
      t.set_description(f'Step {model.step}')
      #batch_data = data_utils.batch_bucketed_data(train_data, FLAGS['batch_size'])
      #for batch in tqdm.tqdm(batch_data):
      #start_time = time.time()
      encoder_inputs, seq_len, decoder_inputs, seq_len_target = train_dataset.next_batch() #model.get_batch(batch) ## to be implemented ## ## done ##
      model_out = model.train_step(encoder_inputs, seq_len, decoder_inputs, seq_len_target) ## remaining ## ##done##
      step_loss = model_out['loss']
      loss += step_loss
      steps += 1
      t.display(f'epoch : {train_dataset.epoch}', pos= 2)
      if model.step % FLAGS['train_images_plotting_frequency'] == 0:
          summary_manager.display_attention_heads(model_out, tag='TrainAttentionHeads')
      #model.increment_epoch()
      if c+1==train_dataset.epoch: #change in epoch
          c = train_dataset.epoch
          loss /=steps
          summary_manager.display_scalar(tag='Meta/epoch', scalar_value= c, plot_all=True)
          summary_manager.display_loss(loss, tag='Train', plot_all= True)
          #summary_manager.display_loss(loss, tag='Validation', plot)



          perplexity = np.exp(loss) if loss < 300 else float('inf')
          t.display("Epoch %d"
                   " perplexity %.4f" % (train_dataset.epoch,
                                         perplexity), pos=3)
          
          steps= 0
          loss = 0
          # Calculate validation result
          val_wer, val_per, val_loss = calc_levenshtein_loss(model, dev_set, summary_manager=summary_manager, step=model.step)
          val_losses.append(val_per)
          summary_manager.display_loss(val_loss, tag='Validation-loss', plot_all= True)
          summary_manager.display_loss(perplexity, tag='Validation-perplexity', plot_all= True)
          summary_manager.display_loss(val_per, tag= 'Validation-per', plot_all= True)
          summary_manager.display_loss(val_wer, tag= 'Validation-wer', plot_all= True)
          summary_manager.display_scalar(tag='Meta/learning_rate', scalar_value=model.optimizer.lr, plot_all= True)


          #validation_improvement_threshold
          t.display("Validation WER: %.5f, PER: %.5f" % (val_wer, val_per), pos= 4)
          if len(val_losses) >= 50:
              global_avg = sum(val_losses[-50:]) / 50.0
              last_10_avg = sum(val_losses[-10:]) / 10.0
              if global_avg-last_10_avg <validation_improvement_threshold:
                  lr *= 0.2
                  t.display("Learning rate updated.", pos= 5)
                  model.set_constants(learning_rate= lr)
                  with open(FLAGS['learning_rate_path'],'w') as f:
                      f.write(str(lr))
          # Validation WER is a moving window, we add the new entry and pop the oldest one
          val_wer_window.append(val_wer) ## confirm from this paper
          if len(val_wer_window) > window_size:
              val_wer_window.pop(0)
              avg_wer = sum(val_wer_window)/float(len(val_wer_window))
              t.display("Average Validation WER %.5f" % (avg_wer), pos= 6)
              # The best model is decided based on average validation WER to remove noisy cases of one off validation success
              if best_wer > avg_wer: ## saving criteria is different ## #done
                  # Save the best model
                  best_wer = avg_wer
                  t.display("Saving Updated Model", pos= 7)
                  save_path = manager.save()
            
      
      print()

if __name__ == "__main__":
    FLAGS = parse_options()
    FLAGS['weights_dir'] = os.path.join(FLAGS['train_dir'], 'model_weights')
    FLAGS['epoch_path'] = os.path.join(FLAGS['train_dir'], 'epoch_count.txt')
    FLAGS['learning_rate_path'] = os.path.join(FLAGS['train_dir'], 'cur_lr.txt')
    if FLAGS['eval']:
        evaluate()  # remaining# ##done##
    else:
        train()
