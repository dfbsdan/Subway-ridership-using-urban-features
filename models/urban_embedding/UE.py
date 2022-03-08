import tensorflow as tf
import numpy as np

# generates a random sample z given a normal distribution's MEAN and LOGVAR
# using the reparameterization trick 
def reparameterize(mean: tf.Tensor, logvar: tf.Tensor):
  eps = tf.random.normal(shape=tf.shape(mean))
  return mean + eps * tf.exp(logvar * .5)

class VAE_Encoder(tf.keras.Model):
  def __init__(self, D: int, F: int, DH: int) -> None:
    # D: Latent space dimension
    # F: Input space dimension
    # DH: Hidden dimension
    super().__init__()
    assert D > 0 and F > 0 and DH > 0
    self.main = tf.keras.Sequential([
      tf.keras.Input(shape=(None, F)),
      tf.keras.layers.Dense(DH, activation='relu', kernel_regularizer='l2', activity_regularizer='l1'), # shape: (batch, P, F) -> (batch, P, DH)
      tf.keras.layers.Dense(DH, activation='relu', kernel_regularizer='l2', activity_regularizer='l1'), # shape -> shape
      # No activation
      tf.keras.layers.Dense(2*D, kernel_regularizer='l2', activity_regularizer='l1'), # shape -> (batch, P, 2*D)
    ])
  
  def encode(self, x: tf.Tensor):
    mean, logvar = tf.split(self.main(x), num_or_size_splits=2, axis=2)
    y = reparameterize(mean, logvar)
    return mean, logvar, y # (batch, P, F) -> (batch, P, D), (batch, P, D), (batch, P, D)

class VAE_Decoder(tf.keras.Model):
  def __init__(self, D: int, F: int, DH: int) -> None:
    # D: Latent space dimension
    # F: Input space dimension
    # DH: Hidden dimension
    super().__init__()
    assert D > 0 and F > 0 and DH > 0
    self.main = tf.keras.Sequential([
      tf.keras.Input(shape=(None, D)),
      tf.keras.layers.Dense(DH, activation='relu'), # shape: (batch, P+Q, D) -> (batch, P+Q, DH)
      tf.keras.layers.Dense(DH, activation='relu'), # shape -> shape
      # No activation
      tf.keras.layers.Dense(F, kernel_regularizer='l2'), # shape -> (batch, P+Q, F)
    ])
  
  def call(self, x: tf.Tensor):
    return self.main(x) # (batch, P+Q, D) -> (batch, P+Q, F)

class FC(tf.keras.Model):
  def __init__(self, units: list, activations: list, input_shape: tuple, drop_rate: float, use_bias: bool) -> None:
    super().__init__()
    assert drop_rate >=0 and drop_rate < 1 and len(input_shape) == 3
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    for num_unit, activation in zip(units, activations):
      model.add(tf.keras.layers.Dropout(drop_rate))
      model.add(tf.keras.layers.Conv2D(
        num_unit, kernel_size=(1,1), strides=(1,1), padding='valid', 
        use_bias=use_bias, activation=activation))
      '''model.add(tf.keras.layers.BatchNormalization(
        momentum=0.9, epsilon=1e-3
      ))'''
    self.main = model

  def call(self, x: tf.Tensor):
    return self.main(x)

def positional_encoding(seq_len: int, dim: int):
  def __get_angles(pos, i, D):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(D))
    return pos * angle_rates

  angle_rads = __get_angles(np.arange(seq_len)[:, np.newaxis], np.arange(dim)[np.newaxis, :], dim)
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32) # shape: (1, seq_len, dim)

class Tr_EncoderBlock(tf.keras.layers.Layer):
  def __init__(self, K: int, d: int, D: int, drop_rate=0.1):
    # K：number of attention heads
    # d：dimension of each attention head outputs
    # DH: Hidden dimension
    # D: Latent space dimension
    super(Tr_EncoderBlock, self).__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=K, key_dim=d, value_dim=d)
    self.ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(D, activation='relu'),
      tf.keras.layers.Dense(D)  # (batch, seq_len, D)
    ])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(drop_rate)
    self.dropout2 = tf.keras.layers.Dropout(drop_rate)

  def call(self, x: tf.Tensor): # (batch, P, D) -> (batch, P, D)
    attn = self.mha(x, x)  # shape: (batch, P, D) -> shape
    attn = self.dropout1(attn) # shape -> shape
    x = self.layernorm1(x + attn) # shape: (batch, P, D)
    ffn = self.ffn(x)  # (batch, P, D)
    ffn = self.dropout2(ffn)
    x = self.layernorm2(x + ffn)  # (batch, P, D)
    return x

class Tr_DecoderBlock(tf.keras.layers.Layer):
  def __init__(self, K: int, d: int, D: int, drop_rate=0.1):
    super(Tr_DecoderBlock, self).__init__()
    self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=K, key_dim=d, value_dim=d)
    self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=K, key_dim=d, value_dim=d)
    self.ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(D, activation='relu'),
      tf.keras.layers.Dense(D)  # (batch, seq_len, D)
    ])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(drop_rate)
    self.dropout2 = tf.keras.layers.Dropout(drop_rate)
    self.dropout3 = tf.keras.layers.Dropout(drop_rate)

  def call(self, y_prev: tf.Tensor, x: tf.Tensor):
    # y_prev: (batch, inp_seq, D), x: (batch, P, D)
    attn = self.mha1(y_prev, y_prev) # (batch, inp_seq, D)
    attn = self.dropout1(attn)
    out = self.layernorm1(attn + y_prev) # (batch, inp_seq, D)

    attn = self.mha2(query=out, value=x, key=x) # (batch, inp_seq, D)
    attn = self.dropout2(attn)
    out = self.layernorm2(attn + out) # (batch, inp_seq, D)

    ffn = self.ffn(out) # (batch, inp_seq, D)
    ffn = self.dropout3(ffn)
    out = self.layernorm3(ffn + out) # (batch, inp_seq, D)
    return out

class Tr_Decoder(tf.keras.Model):
  def __init__(self, Q: int, D: int, L: int, K: int, d: int) -> None:
    super().__init__()
    self.Q = Q
    self.D = D
    self.pos_encoding = positional_encoding(Q, D)
    self.blocks = [Tr_DecoderBlock(K, d, D) for _ in range(L)]
    self.dense = FC(units=[1], activations=[None], input_shape=(D, 1, Q), drop_rate=0.1, use_bias=True)
  
  def call(self, x: tf.Tensor): # (batch, P, D) -> (batch, Q, D)
    batch_sz = tf.shape(x)[0]
    y = tf.zeros(shape=(batch_sz, 1, self.D)) # shape: (batch, 1, D)
    # shape -> (batch, Q+1, D)
    for q in range(self.Q):
      yi = self.blocks[0](y + self.pos_encoding[:, :q+1, :], x) # shape: (batch, q+1, D)
      for block in self.blocks[1:]:
        yi: tf.Tensor = block(yi, x) # shape: (batch, q+1, D)
      yi = tf.transpose(yi, perm=(0, 2, 1)) # shape -> (batch, D, q + 1)
      yi = tf.expand_dims(yi, axis=2) # shape -> (batch, D, 1, q + 1)
      yi = tf.concat([yi, tf.zeros([batch_sz, self.D, 1, self.Q-(q+1)])], axis=3) # shape -> (batch, D, 1, Q)
      yi = self.dense(yi) # shape -> (batch, D, 1, 1)
      yi = tf.squeeze(yi) # shape -> (batch, D)
      yi = tf.expand_dims(yi, axis=1) # shape -> (batch, 1, D)
      y = tf.concat([y, yi], axis=1)
    return y[:, 1:, :]

class Transformer(tf.keras.Model):
  def __init__(self, Q: int, P_max: int, D: int, L: int, K: int, d: int) -> None:
    super().__init__()
    assert Q > 0 and P_max > 0 and D > 0 and L > 0 and K > 0 and d > 0
    # Q: Output sequence length
    # P: Input sequence length
    # D: Latent space dimension
    # L: Attention blocks in encoder/decoder
    # K：number of attention heads
    # d：dimension of each attention head outputs
    self.encoder_pos_encoding = positional_encoding(P_max, D)
    self.encoder = tf.keras.Sequential([Tr_EncoderBlock(K, d, D) for _ in range(L)])
    self.decoder = Tr_Decoder(Q, D, L, K, d)
  
  def call(self, x: tf.Tensor):
    P = tf.shape(x)[1]
    x += self.encoder_pos_encoding # shape: (batch, P_max, D)
    x = self.encoder(x) # shape -> shape
    x = self.decoder(x) # shape -> (batch, Q, D)
    return x

class TrVAE(tf.keras.Model):
  def __init__(self, Q: int, D: int, L: int, K: int, d: int, F: int, P_max: int) -> None:
    # Q: Output sequence length
    # D: Latent space dimension
    # L: Attention blocks in encoder/decoder
    # K：number of attention heads
    # d：dimension of each attention head outputs
    # F: Input space dimension
    # P_max: Maximum input sequence length
    super().__init__()
    DH = K*d
    self.max_P = P_max
    self.encoder = VAE_Encoder(D, F, DH)
    self.transformer = Transformer(Q, P_max, D, L, K, d)
    self.decoder = VAE_Decoder(D, F, DH)
  
  def call(self, x: tf.Tensor):
    eP_mean, eP_logvar, eP = self.encoder.encode(x) # shape: (batch, P, F) -> (batch, P, D), (batch, P, D), (batch, P, D)
    eQ = self.transformer(eP) # shape -> (batch, Q, D)
    ePQ = tf.concat([eP, eQ], axis=1) # shape: (batch, P+Q, D)
    ePQ = self.decoder(ePQ) # shape -> (batch, P+Q, F)
    #return eP_mean, eP_logvar, ePQ
    return ePQ
  
  def save_weights(self, path: str, overwrite: bool):
    path = path.replace('.model', '')
    self.encoder.save_weights(path + '_encoder.model', overwrite=overwrite)
    self.transformer.save_weights(path + '_transformer.model', overwrite=overwrite)
    self.decoder.save_weights(path + '_decoder.model', overwrite=overwrite)

  def load_weights(self, path: str):
    path = path.replace('.model', '')
    self.encoder.load_weights(path + '_encoder.model')
    self.transformer.load_weights(path + '_transformer.model')
    self.decoder.load_weights(path + '_decoder.model')