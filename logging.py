from pathlib import Path

import tensorflow as tf

from utils.audio import Audio
from utils.display import tight_grid, buffer_image
from utils.vec_ops import norm_tensor
from utils.decorators import ignore_exception

def control_frequency(f):
    def apply_func(*args, **kwargs):
        # args[0] is self
        plot_all = ('plot_all' in kwargs) and kwargs['plot_all']
        if (args[0].global_step % args[0].plot_frequency == 0) or plot_all:
            result = f(*args, **kwargs)
            return result
        else:
            return None
    
    return apply_func

class SummaryManager:
  """
  Writes tensorboard logs during training.

  :arg model: model object that is trained
  :arg log_dir: base directory where logs of a config are created
  :arg config: configuration dictionary
  :arg max_plot_frequency: every how many steps to plot
  """

  def __init__(self, 
               model: tf.keras.models.Model,
               log_dir: str,
               config: dict,
               max_plot_frequency= 10,
               default_writer = 'train_base_dir'):
    self.model = model
    self.log_dir = Path(log_dir)
    self.config = config
    self.plot_frequency = max_plot_frequency
    self.default_writer = default_writer
    self.writers = {}
    self.add_writer(tag= default_writer, path= self.log_dir, default= True)

  def add_writer(self, path, tag= None, default= False):
    """
     Adds a writer to self.writers if the writer does not exist already.
    To avoid spamming on disk.

      : return the writer on path with tag or path
      
    """

    if not tag:
      tag = path
    if tag not in self.writers.keys():
      self.writers[tag] = tf.summary.create_file_writer(str(path))
    if default:
      self.default_writer = tag
    return self.writers[tag]

  @property
  def global_step(self):
    return self.model.step

  def add_scalars(self, tag, dictionary):
    for k in dictionary.keys():
      with self.add_writer(str(self.log_dir / k)).as_default():
        tf.summary.scalar(name=tag, data=dictionary[k], step=self.global_step)

  def add_text(self,tag, inp, gd, op, step):
      #text -->list
      with self.writers[self.default_writer].as_default():
          for i in range(len(gd)):
              tf.summary.text(f'{tag}/{i+1}/inp',tf.convert_to_tensor(" ".join(inp[i]),dtype= tf.string), step=step)
              tf.summary.text(f'{tag}/{i+1}/gt_phon',tf.convert_to_tensor(" ".join(gd[i]),dtype= tf.string),step=step)
              tf.summary.text(f'{tag}/{i+1}/out_phon',tf.convert_to_tensor(" ".join(op[i]),dtype= tf.string), step=step)
  def add_scalar(self, tag, scalar_value):
    with self.writers[self.default_writer].as_default():
      tf.summary.scalar(name=tag, data=scalar_value, step=self.global_step)

  def add_image(self, tag, image, step= None):
    if step is None:
      step = self.global_step
    with self.writers[self.default_writer].as_default():
      tf.summary.image(name=tag, data=image, step=step, max_outputs=4)

  def add_histogram(self, tag, values, buckets= None):
    with self.writers[self.default_writer].as_default():
      tf.summary.histogram(name=tag, data=values, step=self.global_step, buckets=buckets)

  def add_audio(self, tag, wav, sr):
    with self.writers[self.default_writer].as_default():
      tf.summary.audio(name=tag,
                       data=wav,
                       sample_rate=sr,
                       step=self.global_step)

  @ignore_exception
  def display_attention_heads(self, outputs, tag= ''):    #### CHeck this ####
    for layer in ['encoder_attention', 'decoder_attention']:
      for k in outputs[layer].keys():
        image = tight_grid(norm_tensor(outputs[layer][k][0]))
        # dim 0 of image_batch is now number of heads
        batch_plot_path = f'{tag}/{layer}/{k}'
        self.add_image(str(batch_plot_path), tf.expand_dims(tf.expand_dims(image, 0), -1))

  @ignore_exception
  def display_mel(self, mel, tag= ''):
    img = tf.transpose(mel)
    figure = self.audio.display_mel(img, is_normal=True)
    buf = buffer_image(figure)
    img_tf = tf.image.decode_png(buf.getvalue(), channels=3)
    self.add_image(tag, tf.expand_dims(img_tf, 0))

  @control_frequency
  @ignore_exception
  def display_loss(self, loss, tag= '', plot_all= False):
    #self.add_scalars(tag=f'{tag}/losses', dictionary=output['losses'])
    self.add_scalar(tag=f'{tag}/loss', scalar_value= loss)

  @control_frequency
  @ignore_exception
  def display_scalar(self, tag, scalar_value, plot_all= False):
    self.add_scalar(tag=tag, scalar_value=scalar_value)

  @ignore_exception
  def display_text(self,inp,gd,op,tag,step):
      self.add_text(tag=tag,inp=inp, gd=gd, op=op, step=step)
  @ignore_exception
  def display_audio(self, tag, mel):
    wav = tf.transpose(mel)
    wav = self.audio.reconstruct_waveform(wav)
    wav = tf.expand_dims(wav, 0)
    wav = tf.expand_dims(wav, -1)
    self.add_audio(tag, wav.numpy(), sr=self.config['sampling_rate'])

    






