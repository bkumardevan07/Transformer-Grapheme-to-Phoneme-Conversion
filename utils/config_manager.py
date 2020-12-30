import subprocess
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import ruamel.yaml

from model.models import AutoregressiveTransformer
from utils.scheduling import piecewise_linear_schedule, reduction_schedule

class ConfigManager:
  def __init__(self, config_path: str, model_kind: str, session_name: str = None):
    if model_kind not in ['autoregressive']:
      raise TypeError(f"model kind must be in {['autoregressive']}")
    self.config_path = Path(config_path)
    self.model_kind = model_kind
    self.yaml = ruamel.yaml.YAML()
    self.config, self.data_config, self.model_config = self._load_config()
    self.git_hash = self._get_git_hash()
    if session_name is None:
      if self.config['session_name'] is None:
        session_name = self.git_hash
    self.session_name = '_'.join(filter(None, [self.config_path.name, session_name]))
    self.base_dir, self.log_dir, self.train_datadir, self.weights_dir = self._make_folder_paths()
    self.learning_rate = np.array(self.config['learning_rate_schedule'])[0,1].astype(np.float32)  # set this up
    if model_kind == 'autoregressive':
      self.max_r = np.array(self.config['reduction_factor_schedule'])[0,1].astype(np.int32)  # set this up
      self.stop_scaling = self.config.get('stop_loss_scaling', 1.)  # set this up

  def _load_config(self):
    with open(str(self.config_path / 'data_config.yaml'), 'rb') as data_yaml:
      data_config = self.yaml.load(data_yaml)
    with open(str(self.config_path / f'{self.model_kind}_config.yaml'), 'rb') as model_yaml:
      model_config = self.yaml.load(model_yaml)
    all_config = {}
    all_config.update(model_config)
    all_config.update(data_config)
    return all_config, data_config, model_config
    
  @staticmethod
  def _get_git_hash():
    try:
      return subprocess.check_output(['git', 'describe', '--always']).strip().decode()
    except Exception as e:
      print(f'WARNING: could not retrieve git hash. {e}')

  def _check_hash(self):
    try:
      git_hash = subprocess.check_output(['git', 'describe', '--always']).strip().decode()
      if self.config['git_hash'] != git_hash:
        print(f"WARNING: git hash mismatch. Current: {git_hash}. Config hash: {self.config['git_hash']}")
    except Exception as e:
      print(f'WARNING: could not check git hash. {e}')

  def _make_folder_paths(self):
    base_dir = Path(self.config['log_directory']) / self.session_name
    log_dir = base_dir / f'{self.model_kind}_logs'
    weights_dir = base_dir / f'{self.model_kind}_weights'
    train_datadir = self.config['train_data_directory']   # set this up
    if train_datadir is None:
      train_datadir = self.config['data_directory']  # remember
    train_datadir = Path(train_datadir)
    return base_dir, log_dir, train_datadir, weights_dir

  @staticmethod
  def _print_dict_values(values, key_name, level= 0, tab_size = 2):
    tab = level * tab_size * ' '
    print(tab + '-', key_name, ':', values)

  def _print_dictionary(self, dictionary, recursion_level = 0):
    for key in dictionary.keys():
      if isinstance(key, dict):
        recursion_level += 1
        self._print_dictionary(dictionary[key], recursion_level)
      else:
        self._print_dict_values(dictionary[key], key_name = key, level = recursion_level)

  def print_config(self):
    print('\nCONFIGURATION', self.session_name)
    self._print_dictionary(self.config)

  def update_config(self):
    self.config['git_hash'] = self.git_hash
    self.model_config['git_hash'] = self.git_hash
    self.data_config['session_name'] = self.session_name
    self.model_config['session_name'] = self.session_name
    self.config['session_name'] = self.session_name

  def get_model(self, ignore_hash= False):
    if not ignore_hash:
      self._check_hash()
    if self.model_kind == 'autoregressive':
        #print(self.config['train_data_dir']y
        return AutoregressiveTransformer(encoder_model_dimension = self.config['encoder_model_dimension'],
                                       decoder_model_dimension = self.config['decoder_model_dimension'],
                                       encoder_num_heads = self.config['encoder_num_heads'],
                                       decoder_num_heads = self.config['decoder_num_heads'],
                                       encoder_num_layers = self.config['encoder_num_layers'],
                                       decoder_num_layers = self.config['decoder_num_layers'],
                                       encoder_maximum_position_encoding = self.config['encoder_maximum_position_encoding'],
                                       decoder_maximum_position_encoding = self.config['decoder_maximum_position_encoding'],
                                       encoder_feed_forward_dimension = self.config['encoder_feed_forward_dimension'],
                                       decoder_feed_forward_dimension = self.config['decoder_feed_forward_dimension'],
                                       dropout_rate = self.config['dropout_rate'],
                                       mel_start_value = self.config['mel_start_value'],
                                       mel_end_value = self.config['mel_end_value'],
                                       mel_channels = self.config['mel_channels'],
                                       encoder_vocab_size = self.config['encoder_vocab_size'],
                                       speaker_vocab_size = self.config['speaker_vocab_size'],
                                       speaker_feed_forward_dimension = self.config['speaker_feed_forward_dimension'],
                                       speaker_embedding_dimension = self.config['speaker_embedding_dimension'],
                                       decoder_prenet_dimensions = self.config['decoder_prenet_dimensions'],
                                       decoder_prenet_dropout = self.config['decoder_prenet_dropout'],
                                       stop_linear_dimension = self.config['stop_linear_dimension'],
                                       max_r = self.max_r, ###observe this ###
                                       phoneme_language= self.config['phoneme_language'],
                                       with_stress= self.config['with_stress'],
                                       debug = self.config['debug'],
                                       token_id_path = self.config['train_data_directory'],
                                       diagonal_bandwidth_b = self.config['diagonal_bandwidth_b'],
                                       diagonal_rate_regul_coeff = self.config['diagonal_rate_regul_coeff'],
                                       loss_weights = self.config['loss_weights'],
                                       layernorm = self.config['layernorm'],
                                       Ldc = self.config['Ldc'],
                                       attention_window = self.config['attention_window'],
                                       regul_coeff = self.config['diagonal_rate_regul_coeff'],
                                       diag_bandwidth = self.config['diagonal_bandwidth_b']
                                      )

  def compile_model(self, model):
    if self.model_kind == 'autoregressive':
      model._compile(stop_scaling = self.stop_scaling, optimizer = self.new_adam(self.learning_rate))

  @staticmethod
  def new_adam(learning_rate):
    return tf.keras.optimizers.Adam(learning_rate,
                                    beta_1= 0.9,
                                    beta_2= 0.98,
                                    epsilon= 1e-9)

  def dump_config(self):
    self.update_config()
    with open(self.base_dir / f'{self.model_kind}_config_yaml', 'w') as model_yaml:
      self.yaml.dump(self.model_config, model_yaml)
    with open(self.base_dir / 'data_config.yaml', 'w') as data_yaml:
      self.yaml.dump(self.data_config, data_yaml)
    
  def create_remove_dirs(self, clear_dir: False, clear_logs: False, clear_weights: False):
    self.base_dir.mkdir(exist_ok = True)
    if clear_dir:
      delete = input(f'Delete {self.log_dir} AND {self.weights_dir}? (y/[n])')
      if delete == 'y':
        shutil.rmtree(self.log_dir, ignore_errors= True)
        shutil.rmtree(self.weights_dir, ignore_errors= True)
    if clear_logs:
      delete = input(f'Delete {self.log_dir}? (y/[n])')
      if delete == 'y':
        shutil.rmtree(self.weights_dir, ignore_errors= True)
    if clear_weights:
      delete = input(f'Delete {self.weights_dir}? (y/[n])')
      if delete == 'y':
        shutil.rmtree(self.weights_dir, ignore_errors=True)
    self.log_dir.mkdir(exist_ok=True)
    self.weights_dir.mkdir(exist_ok=True)


  def load_model(self, checkpoint_path: str = None, verbose= True):
    model = self.get_model()
    self.compile_model(model)
    ckpt = tf.train.Checkpoint(net= model)
    manager = tf.train.CheckpointManager(ckpt, self.weights_dir,
                                        max_to_keep= None)
    if checkpoint_path:
      ckpt.restore(checkpoint_path)
      if verbose:
        print(f'restored weights from {checkpoint_path} at step {model.step}')
    else:
      if manager.latest_checkpoint is None:
        print(f'WARNING: could not find weights file. Trying to load from \n {self.weigths_dir}.')
        print('Edit data_config.yaml to point at the right log directory.')
        ckpt.restore(manager.latest_checkpoint)
      if verbose:
        print(f'restored weights from {manager.latest_checkpoint} at step {model.step}')
    
    #decoder_prenet_dropout = piecewise_linear_schedule(model.step, self.config['decoder_prenet_dropout_schedule']) # check this
    reduction_factor = None
    if self.model_kind == 'autoregressive': #### CHECK THIS ALSO ####
      reduction_factor = reduction_schedule(model.step, self.config['reduction_factor_schedule'])
    #model.set_constants(reduction_factor=reduction_factor, decoder_prenet_dropout=decoder_prenet_dropout) ### check this ###
    return model











  
