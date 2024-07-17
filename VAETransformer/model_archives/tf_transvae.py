import tensorflow as tf
import numpy as np

MAX_LEN = 1291
##### AT SOME POINT TRY TO USE LEARNED EMBEDDING FOR FIRST IN INPUT SEQUENCE #####
##### (PROBABLY WITH BASE THRU MODE FOR WHICH WE HAVE THE MOST DATA)


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

class EncoderDecoder(tf.keras.Model):
  def __init__(self, d_model, enc_layers, dec_layers, num_heads, d_ff, d_latent, bypass_bottleneck, eps_scale=1, dropout_rate=0.1):
    super().__init__()
    self.normalization_layer = tf.keras.layers.Normalization(axis=-1)
    
    self.encoder = VAEEncoder(enc_layers, num_heads, d_model, d_ff, d_latent, bypass_bottleneck=bypass_bottleneck, eps_scale=eps_scale, dropout_rate=dropout_rate)
    self.decoder = VAEDecoder(dec_layers, num_heads, d_model, d_ff, d_latent, bypass_bottleneck=bypass_bottleneck, dropout_rate=dropout_rate)
    self.src_embed = PositionalLinear(d_model)
    self.tgt_embed = PositionalLinear(d_model)
    self.generator = Generator(d_model)

    self.denormalization_layer = tf.keras.layers.Normalization(axis=-1, invert=True)

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

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x = inputs #src, tgt = inputs

    context = self.normalization_layer(context)
    context = self.src_embed(context)

    x = self.normalization_layer(x)
    x = self.tgt_embed(x)

    mem, mu, logvar = self.encoder(context)
    x = self.decoder(x, mem)

    # Final linear layer output.
    x = self.generator(x)  # (batch_size, target_len, target_vocab_size)

    x = self.denormalization_layer(x)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del x._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return x, mu, logvar

class Generator(tf.keras.layers.Layer):
  def __init__(self, target_vocab_size):
    super().__init__()
    self.proj = tf.keras.layers.Dense(target_vocab_size)

  def call(self, x):
    return self.proj(x)

class VAEDecoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, num_heads, d_model, d_ff, d_latent, bypass_bottleneck, dropout_rate=0.1):
    super().__init__()
    self.final_encodes = [VAEEncoderLayer(num_heads, d_model, d_ff, dropout_rate=dropout_rate)]
    self.layers = [VAEDecoderLayer(num_heads, d_model, d_ff, dropout_rate=dropout_rate) for _ in range(num_layers)]
    self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.bypass_bottleneck = bypass_bottleneck
    self.relu = tf.keras.layers.ReLU()

    self.linear = tf.keras.layers.Dense(64)
    self.reshape = tf.keras.layers.Reshape((1, 64))
    self.permute = tf.keras.layers.Permute((2, 1))
    self.deconv_bottleneck = DeconvBottleneck(MAX_LEN)##NOT SURE WHETHER THIS IS 1290 or 64

  def call(self, x, mem):
    #print(mem.shape)
    if not self.bypass_bottleneck:
      mem = self.relu(self.linear(mem))
      #print(mem.shape)
      mem = self.reshape(mem)
      #print(mem.shape)
      mem = self.deconv_bottleneck(mem)
      #print(mem.shape)
      mem = self.permute(mem)
      #print(mem.shape)
    
    for final_encode in self.final_encodes:
      mem = final_encode(mem)
    
    mem = self.norm(mem)

    for attn_layer in self.layers:
      x = attn_layer(x, mem, mem)
    
    return self.norm(x)

class DeconvBottleneck(tf.keras.layers.Layer):
  def __init__(self, size):
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
  
  def call(self, x):
    for deconv in self.deconv_layers:
      x = self.relu(deconv(x))
    return x

class VAEDecoderLayer(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.1):
    super().__init__()
    self.self_attn = SelfAttentionLayer(num_heads, d_model, dropout_rate=dropout_rate, causal=True)
    self.sublayer = [SublayerConnection(dropout_rate=dropout_rate) for _ in range(3)]
    self.src_attn = SrcAttentionLayer(num_heads, d_model, dropout_rate=dropout_rate)
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

  def call(self, x, memory_key, memory_val):
    x = self.sublayer[0](x, self.self_attn)
    x = self.sublayer[1](x, lambda x: self.src_attn(x, memory_key, memory_val))
    return self.sublayer[2](x, self.feed_forward)

class SrcAttentionLayer(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, dropout_rate=0.1):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

  def call(self, x, m_key, m_val):
    return self.mha(x, m_key, m_val)

class SelfAttentionLayer(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, dropout_rate=0.1, causal=False):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
    self.causal = causal

  def call(self, x):
    return self.mha(x, x, x, use_causal_mask=self.causal)

class VAEEncoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, num_heads, d_model, d_ff, d_latent, bypass_bottleneck, eps_scale=1, dropout_rate=0.1):
    super().__init__()
    self.enc_layers = [VAEEncoderLayer(num_heads, d_model, d_ff, dropout_rate=dropout_rate) for _ in range(num_layers)]
    self.conv_bottleneck = ConvBottleneck(d_model)
    self.z_means, self.z_var = tf.keras.layers.Dense(d_latent), tf.keras.layers.Dense(d_latent)
    self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.bypass_bottleneck = bypass_bottleneck
    self.eps_scale = eps_scale
    self.permute = tf.keras.layers.Permute((2, 1))
    self.flat = tf.keras.layers.Flatten()

  def reparameterize(self, mu, logvar, eps_scale=1):
    std = tf.exp(.5 * logvar)
    eps = tf.random.normal(shape=tf.shape(std)) * eps_scale
    return mu + eps*std
  
  def call(self, x):
    for attn_layer in self.enc_layers:
      x = attn_layer(x)
    
    mem = self.norm(x)

    if self.bypass_bottleneck:
      mu, logvar = tf.convert_to_tensor(0.), tf.convert_to_tensor(0.)
    else:
      mem = self.permute(mem)
      mem = self.conv_bottleneck(mem)
      mem = self.flat(mem)##MAYBE NEED CONTIGUOUS
      mu, logvar = self.z_means(mem), self.z_var(mem)
      mem = self.reparameterize(mu, logvar)
    
    return mem, mu, logvar

class ConvBottleneck(tf.keras.layers.Layer):
  def __init__(self, size):
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
  
  def call(self, x):
    for conv in self.conv_layers:
      x = self.relu(conv(x))
    return x

class VAEEncoderLayer(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, d_ff, dropout_rate=0.1):
    super().__init__()
    self.self_attn = SelfAttentionLayer(num_heads, d_model, dropout_rate=dropout_rate)
    self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate=0.1)
    self.sublayer = [SublayerConnection(dropout_rate=dropout_rate) for _ in range(2)]
  
  def call(self, x):
    x = self.sublayer[0](x, self.self_attn)
    return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(tf.keras.layers.Layer):
  def __init__(self, dropout_rate=0.1):
    super().__init__()
    self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.add = tf.keras.layers.Add()

  def call(self, x, sublayer):
    _x = self.norm(x)
    _x = sublayer(_x)
    _x = self.dropout(_x)
    return self.add([x, _x])

class LayerNorm(tf.keras.layers.Layer):
  def __init__(self, eps=1e-6, axis=-1):
    super().__init__()
    self.eps = eps
    self.axis = axis

  def build(self, input_shape):
    self.a_2 = self.add_weight("a_2", input_shape[self.axis], initializer="one", trainable=True)
    self.b_2 = self.add_weight("b_2", input_shape[self.axis], initializer="zero", trainable=True)

  def call(self, x):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    std = tf.math.reduce_std(x, axis=-1, keepdims=True)
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, d_ff, dropout_rate=0.1):
    super().__init__()
    self.w_1 = tf.keras.layers.Dense(d_ff)
    self.w_2 = tf.keras.layers.Dense(d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.relu = tf.keras.layers.ReLU()

  def call(self, x):
    return self.w_2(self.dropout(self.relu(self.w_1(x))))

"""
class MultiHeadedAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, dropout_rate=0.1):
    super().__init__()
    assert d_model % num_heads == 0
    self.d_k = d_model // num_heads
    self.num_heads = num_heads
    self.linears = [tf.keras.layers.Dense(d_model) for _ in range(4)]
    self.attn = None
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, query, key, value):
    tf.keras.layers.MultiHeadAttention()
"""
