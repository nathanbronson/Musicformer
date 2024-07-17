"""
    Musicformer: a neural network for unsupervised embeddings
    Copyright (C) 2023  Nathan Bronson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import tensorflow as tf
import numpy as np

#SAVE_MEMORY = True
#BENCHMARK = False
#LOAD_TF_RECORDS = True
#CONVERT_TO_TFRECORDS = False
#TEST_PROP = .2
MAX_TOKENS = MAX_LEN = 1291
#BUFFER_SIZE = 5000
#BATCH_SIZE = 7 if not TPU else 8 * tpu_strategy.num_replicas_in_sync#16 * tpu_strategy.num_replicas_in_sync#9 * tpu_strategy.num_replicas_in_sync#20 * tpu_strategy.num_replicas_in_sync
num_layers = 7
decoder_layers = 1
d_model = 128
N_MELS = 64
dff = 512#216
num_heads = 8
dropout_rate = 0.1
d_latent = 64
decoder_d_model = 64
decoder_dff = 216
CHECKPOINT_PATH = "./checkpoints/checkpoint"

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(length=MAX_LEN, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class PositionalLinear(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Dense(d_model)#MASK ZERO?
    self.pos_encoding = positional_encoding(length=MAX_LEN, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    if vocab_size is not None:
      print("a value was specified for the encoder's vocab_size, but this variable is unused, so the value will have no effect")

    self.pos_embedding = PositionalLinear(d_model)#PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = GlobalSelfAttention(##CausalSelfAttention( #default is second
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalLinear(d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x

class BlindDecoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(BlindDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(1292, d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = tf.repeat(tf.expand_dims(tf.range(1, 1292), 0), tf.shape(x)[0], axis=0)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x

class DenseDeconvBottleneck(tf.keras.layers.Layer):
  def __init__(self, size, d_model):
    super().__init__()
    self.deconv_layers = tf.keras.Sequential([
        tf.keras.layers.Dense(size * d_model),
        tf.keras.layers.Reshape((size, d_model))
    ])
    self.relu = tf.keras.layers.ReLU()

  def call(self, x):
    return self.relu(self.deconv_layers(x))

class DeconvBottleneck(tf.keras.layers.Layer):
  def __init__(self, size, d_model):
    super().__init__()
    deconv_layers = []
    in_d = 64
    for i in range(3):
      out_d = (size - in_d) // 4 + in_d
      stride = 8#4 - i (sqrt(d_model=64))
      kernel_size = 11
      if i == 2:
        out_d = size
        stride = 1
      deconv_layers.append(tf.keras.Sequential([
        tf.keras.layers.Conv1DTranspose(out_d, kernel_size, strides=stride, padding="same")
      ]))
      in_d = out_d
    self.deconv_layers = deconv_layers
    self.relu = tf.keras.layers.ReLU()
    self.linear = tf.keras.layers.Dense(d_model)
    self.reshape = tf.keras.layers.Reshape((1, d_model))
    self.permute = tf.keras.layers.Permute((2, 1))

  def call(self, x):
    x = self.relu(self.linear(x))
    x = self.reshape(x)
    for deconv in self.deconv_layers:
      x = self.relu(deconv(x))
    x = self.permute(x)
    return x

class ConvBottleneck(tf.keras.layers.Layer):
  def __init__(self, size, d_latent):
    super().__init__()
    conv_layers = []
    first = True
    in_d = size
    for i in range(3):#MIGHT NEED SOME STRIDE
      out_d = int((in_d - 64) // 2 + 64)
      if first:
        kernel_size = 9
        first = False
      else:
        kernel_size = 8
      if i == 2:
        out_d = 64
      conv_layers.append(tf.keras.Sequential([
        tf.keras.layers.Conv1D(out_d, kernel_size),
        tf.keras.layers.MaxPool1D()
      ]))
      in_d = out_d
    self.conv_layers = conv_layers
    self.relu = tf.keras.layers.ReLU()
    self.flat = tf.keras.layers.Flatten()
    self.mu_dense = tf.keras.layers.Dense(d_latent)
    self.logvar_dense = tf.keras.layers.Dense(d_latent)

  def call(self, x):
    for conv in self.conv_layers:
      x = self.relu(conv(x))
    x = self.flat(x)
    mu = self.mu_dense(x)
    logvar = self.logvar_dense(x)
    return mu, logvar

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1, d_latent=64,
               decoder_d_model=None, decoder_layers=None, decoder_dff=None):
    super().__init__()
    self.normalization_layer = tf.keras.layers.Normalization(axis=-1)
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers if decoder_layers is None else decoder_layers,
                           d_model=d_model if decoder_d_model is None else decoder_d_model,
                           num_heads=num_heads, dff=dff if decoder_dff is None else decoder_dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.conv_bottleneck = ConvBottleneck(d_model, d_latent)
    self.deconv_bottleneck = DenseDeconvBottleneck(MAX_LEN, d_model if decoder_d_model is None else decoder_d_model)#maybe decoder_d_model
    #self.cross_prepare = tf.keras.Sequential([
    #    tf.keras.layers.Reshape((d_latent, 1)),
    #    tf.keras.layers.Dense(d_model if decoder_d_model is None else decoder_d_model)
    #])
    self.cross_prepare = tf.keras.Sequential([
        tf.keras.layers.Dense(d_latent * (d_model if decoder_d_model is None else decoder_d_model)),
        tf.keras.layers.Reshape((d_latent, d_model if decoder_d_model is None else decoder_d_model)),
    ])

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    self.denormalization_layer = tf.keras.layers.Normalization(axis=-1, invert=True)

    self.loss_fn = None
    self.last_met = {}

    self.loss_tracker = tf.keras.metrics.Mean(name="loss")#tf.keras.metrics.Sum("loss")
    #self.loss_den = tf.keras.metrics.Sum("dloss")
    self.accuracy = tf.keras.metrics.MeanAbsoluteError(name="mae")
    self.mse = tf.keras.metrics.Mean(name="mse")
    self.kl = tf.keras.metrics.Mean(name="kl")

  @property
  def metrics(self):
    return [self.loss_tracker, self.accuracy, self.mse, self.kl]

  def set_loss_fn(self, loss_fn):
    self.loss_fn = loss_fn

  def adapt(self, ds=None, mean=None, variance=None):
    if ds is not None:
        self.normalization_layer.adapt(ds)
        del self.denormalization_layer
        self.denormalization_layer = tf.keras.layers.Normalization(axis=-1, invert=True, mean=self.normalization_layer.mean, variance=self.normalization_layer.variance)
    else:
        del self.normalization_layer
        del self.denormalization_layer
        self.normalization_layer = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance)
        self.denormalization_layer = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance, invert=True)

  def encode(self, x):
    x = self.normalization_layer(x)
    #print(tf.shape(x))#tf.print(tf.shape(x), output_stream=sys.stdout)
    x = self.encoder(x)
    #print(tf.shape(x))#tf.print(tf.shape(x), output_stream=sys.stdout)
    mu, logvar = self.conv_bottleneck(x)
    #print(tf.shape(mu))#tf.print(tf.shape(mu), output_stream=sys.stdout)
    #print(tf.shape(logvar))#tf.print(tf.shape(logvar), output_stream=sys.stdout)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return eps * tf.exp(logvar * .5) + mu

  def decode(self, z):
    x = self.deconv_bottleneck(z)
    #print(tf.shape(x))##tf.print(tf.shape(x), output_stream=sys.stdout)
    z = self.cross_prepare(z)
    #print(tf.shape(z))#tf.print(tf.shape(z), output_stream=sys.stdout)
    x = self.decoder(x, z)
    #print(tf.shape(x))#tf.print(tf.shape(x), output_stream=sys.stdout)
    x = self.final_layer(x)
    #print(tf.shape(x))#tf.print(tf.shape(x), output_stream=sys.stdout)
    x = self.denormalization_layer(x)
    #print(tf.shape(x))#tf.print(tf.shape(x), output_stream=sys.stdout)
    try:
      del x._keras_mask
    except AttributeError:
      pass
    return x

  def call(self, inputs, training=True):
    x, _  = inputs
    mu, logvar = self.encode(x)
    if training:
      z = self.reparameterize(mu, logvar)
    else:
      z = mu
    x = self.decode(z)
    return x, mu, logvar, z

  def train_step(self, data):
    x, y = data

    with tf.GradientTape() as tape:
      y_pred, mu, logvar, z = self(x, training=True)
      loss, mse, kl = self.loss_fn(y, y_pred, mu, logvar, z, step=self.optimizer.iterations, mse_divide=1.)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    for metric in self.metrics:
      if metric.name == "loss":
        metric.update_state(loss)
      elif metric.name == "mse":
        metric.update_state(mse)
      elif metric.name == "kl":
        metric.update_state(kl)
      elif metric.name == "dloss":
        metric.update_state(1)
      else:
        metric.update_state(y, y_pred)

    met = {m.name: m.result() for m in self.metrics}#{m.name: m.result() / (self.metrics[-1].result() if m.name == "loss" else 1) for m in self.metrics}
    #met = {k: tf.cond(tf.math.is_nan(v), lambda: tf.convert_to_tensor(self.last_met.get(k, np.nan)), lambda: v) for k, v in met.items()}#{k: self.last_met[k] if tf.math.is_nan(v) and k in self.last_met.keys() else v for k, v in met.items()}
    #self.last_met = {k: v.numpy() for k, v in met.items()}

    return met

  def test_step(self, data):
    x, y = data

    y_pred, mu, logvar, z = self(x, training=False)

    loss, mse, kl = self.loss_fn(y, y_pred, mu, logvar, z, mse_divide=1.)

    for metric in self.metrics:
      if metric.name == "loss":
        metric.update_state(loss)
      elif metric.name == "mse":
        metric.update_state(mse)
      elif metric.name == "kl":
        metric.update_state(kl)
      elif metric.name == "dloss":
        metric.update_state(1)
      else:
        metric.update_state(y, y_pred)

    met = {m.name: m.result() for m in self.metrics}
    #met = {k: tf.cond(tf.math.is_nan(v), lambda: tf.convert_to_tensor(self.last_met.get(k, np.nan)), lambda: v) for k, v in met.items()}
    #self.last_met = {k: v.numpy() for k, v in met.items()}

    return met

def make_transformer():
    return Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=N_MELS,
        target_vocab_size=N_MELS,
        dropout_rate=dropout_rate,
        d_latent=d_latent,
        decoder_layers=decoder_layers,
        decoder_d_model=decoder_d_model,
        decoder_dff=decoder_dff
    )

if __name__ == "__main__":
    transformer = make_transformer()

    transformer.load_weights(CHECKPOINT_PATH)