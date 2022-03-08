import tensorflow as tf

def FC(x, units, activations, bn: bool, bn_decay, is_training: bool, use_bias = True):
  if isinstance(units, int):
    units = [units]
    activations = [activations]
  elif isinstance(units, tuple):
    units = list(units)
    activations = list(activations)
  assert type(units) == list
  for num_unit, activation in zip(units, activations):
    x = tf.keras.layers.Conv2D(filters=num_unit, kernel_size=[1,1], strides=[1, 1], 
      padding='valid', use_bias=use_bias, activation=activation
    )(inputs=x, training=is_training)
    if bn:
      x = tf.keras.layers.BatchNormalization(
        momentum=bn_decay, epsilon=1e-3
      )(inputs=x, training=is_training)
  return x

def STEmbedding(SE, TE, UEx, UEy, T, D, P, bn, bn_decay, is_training):
  '''
  spatio-temporal embedding
  SE:     [1, N, SE_D]
  TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
  UEx:    [batch_size, N, UE_D]
  UEy:    [batch_size, N, UE_D]
  T:      num of time steps in one day
  D:      output dims
  retrun: [batch_size, P + Q, N, D]
  '''
  # spatial embedding
  SE = tf.expand_dims(SE, axis = 0)
  SE = FC( # -> [1, 1, N, D]
    SE, units = [D, D], activations = [tf.nn.relu, None],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  print(f'SE: {SE.shape}')
  # temporal embedding
  dayofweek = tf.one_hot(TE[..., 0], depth = 7)
  timeofday = tf.one_hot(TE[..., 1], depth = T)
  TE = tf.concat((dayofweek, timeofday), axis = -1)
  print(f'TE: {TE.shape}')
  TE = tf.expand_dims(TE, axis = 2)
  print(f'TE: {TE.shape}')
  TE = FC( # -> [batch_size, P+Q, 1, D]
    TE, units = [D, D], activations = [tf.nn.relu, None],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  print(f'TE: {TE.shape}')
  STE = tf.add(SE, TE) # [batch_size, P + Q, N, D]
  print(f'STE: {STE.shape}')
  ###############################################################################################################################
  UEx = tf.expand_dims(UEx, axis = 1) # [batch_size, N, UE_D] -> [batch_size, 1, N, UE_D]
  UEx = FC( # -> [batch_size, 1, N, D]
    UEx, units = [D, D], activations = [tf.nn.relu, tf.nn.relu],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  print(f'UEx: {UEx.shape}')
  UEy = tf.expand_dims(UEy, axis = 1) # [batch_size, N, UE_D] -> [batch_size, 1, N, UE_D]
  UEy = FC( # -> [batch_size, 1, N, D]
    UEy, units = [D, D], activations = [tf.nn.relu, tf.nn.relu],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  print(f'UEy: {UEy.shape}')
  STE_P = STE[:, :P, :, :] # [batch_size, P, N, D]
  print(f'STE_P: {STE_P.shape}')
  STE_P = tf.add(STE_P, UEx)
  print(f'STE_P: {STE_P.shape}')
  STE_Q = STE[:, P:, :, :] # [batch_size, Q, N, D]
  print(f'STE_Q: {STE_Q.shape}')
  STE_Q = tf.add(STE_Q, UEy)
  print(f'STE_Q: {STE_Q.shape}')
  STE = tf.concat([STE_P, STE_Q], axis=1) # [batch_size, P + Q, N, D]
  print(f'STE: {STE.shape}')
  ###############################################################################################################################
  return STE

def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
  '''
  spatial attention mechanism
  X:      [batch_size, num_step, N, D]
  STE:    [batch_size, num_step, N, D]
  K:      number of attention heads
  d:      dimension of each attention outputs
  return: [batch_size, num_step, N, D]
  '''
  D = K * d
  X = tf.concat((X, STE), axis = -1)
  # [batch_size, num_step, N, K * d]
  query = FC(
    X, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  key = FC(
    X, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  value = FC(
    X, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  # [K * batch_size, num_step, N, d]
  query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
  key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
  value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
  # [K * batch_size, num_step, N, N]
  attention = tf.matmul(query, key, transpose_b = True)
  attention /= (d ** 0.5)
  attention = tf.nn.softmax(attention, axis = -1)
  # [batch_size, num_step, N, D]
  X = tf.matmul(attention, value)
  X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
  X = FC(
    X, units = [D, D], activations = [tf.nn.relu, None],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  return X

def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask = True):
  '''
  temporal attention mechanism
  X:      [batch_size, num_step, N, D]
  STE:    [batch_size, num_step, N, D]
  K:      number of attention heads
  d:      dimension of each attention outputs
  return: [batch_size, num_step, N, D]
  '''
  D = K * d
  X = tf.concat((X, STE), axis = -1)
  # [batch_size, num_step, N, K * d]
  query = FC(
    X, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  key = FC(
    X, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  value = FC(
    X, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  # [K * batch_size, num_step, N, d]
  query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
  key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
  value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
  # query: [K * batch_size, N, num_step, d]
  # key:   [K * batch_size, N, d, num_step]
  # value: [K * batch_size, N, num_step, d]
  query = tf.transpose(query, perm = (0, 2, 1, 3))
  key = tf.transpose(key, perm = (0, 2, 3, 1))
  value = tf.transpose(value, perm = (0, 2, 1, 3))
  # [K * batch_size, N, num_step, num_step]
  attention = tf.matmul(query, key)
  attention /= (d ** 0.5)
  # mask attention score
  if mask:
    batch_size = tf.shape(X)[0]
    num_step = X.get_shape()[1].value
    N = X.get_shape()[2].value
    mask = tf.ones(shape = (num_step, num_step))
    mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
    mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
    mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
    mask = tf.cast(mask, dtype = tf.bool)
    attention = tf.where(condition=mask, x=attention, y=-2 ** 15 + 1)
  # softmax   
  attention = tf.nn.softmax(attention, axis = -1)
  # [batch_size, num_step, N, D]
  X = tf.matmul(attention, value)
  X = tf.transpose(X, perm = (0, 2, 1, 3))
  X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
  X = FC(
    X, units = [D, D], activations = [tf.nn.relu, None],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  return X

def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
  '''
  gated fusion
  HS:     [batch_size, num_step, N, D]
  HT:     [batch_size, num_step, N, D]
  D:      output dims
  return: [batch_size, num_step, N, D]
  '''
  XS = FC(
    HS, units = D, activations = None,
    bn = bn, bn_decay = bn_decay,
    is_training = is_training, use_bias = False)
  XT = FC(
    HT, units = D, activations = None,
    bn = bn, bn_decay = bn_decay,
    is_training = is_training, use_bias = True)
  z = tf.nn.sigmoid(tf.add(XS, XT))
  H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
  H = FC(
    H, units = [D, D], activations = [tf.nn.relu, None],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  return H

def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask = False):
  HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
  HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask = mask)
  H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
  return tf.add(X, H)

def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
  '''
  transform attention mechanism
  X:      [batch_size, P, N, D]
  STE_P:  [batch_size, P, N, D]
  STE_Q:  [batch_size, Q, N, D]
  K:      number of attention heads
  d:      dimension of each attention outputs
  return: [batch_size, Q, N, D]
  '''
  D = K * d
  # query: [batch_size, Q, N, K * d]
  # key:   [batch_size, P, N, K * d]
  # value: [batch_size, P, N, K * d]
  query = FC(
    STE_Q, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  key = FC(
    STE_P, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  value = FC(
    X, units = D, activations = tf.nn.relu,
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  # query: [K * batch_size, Q, N, d]
  # key:   [K * batch_size, P, N, d]
  # value: [K * batch_size, P, N, d]
  query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
  key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
  value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
  # query: [K * batch_size, N, Q, d]
  # key:   [K * batch_size, N, d, P]
  # value: [K * batch_size, N, P, d]
  query = tf.transpose(query, perm = (0, 2, 1, 3))
  key = tf.transpose(key, perm = (0, 2, 3, 1))
  value = tf.transpose(value, perm = (0, 2, 1, 3))    
  # [K * batch_size, N, Q, P]
  attention = tf.matmul(query, key)
  attention /= (d ** 0.5)
  attention = tf.nn.softmax(attention, axis = -1)
  # [batch_size, Q, N, D]
  X = tf.matmul(attention, value)
  X = tf.transpose(X, perm = (0, 2, 1, 3))
  X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
  X = FC(
    X, units = [D, D], activations = [tf.nn.relu, None],
    bn = bn, bn_decay = bn_decay, is_training = is_training)
  return X

def GMAN(P: int, Q: int, N: int, F: int, SE_D: int, UE_D: int, T: int, L: int, K: int, d: int, bn: bool, bn_decay: float):
  '''
  P: input steps
  Q: output steps
  N: number of stations
  F: number of features for each station
  SE_D: spatial embedding dimension
  UE_D: urban embedding dimension
  T: one day is divided into T steps
  L: number of STAtt blocks in the encoder/decoder
  K: number of attention heads
  d: dimension of each attention head outputs
  bn: use batch normalization (true) or not (false)
  bn_decay: batch norm decay momentum
  '''
  # inputs
  X = tf.keras.Input(shape=(P,N,F), name="X", dtype=tf.float32)
  TE = tf.keras.Input(shape=(P+Q,2), name="TE", dtype=tf.int32) # (time-of-day, day-of-week)
  SE = tf.keras.Input(shape=(N,SE_D), batch_size=1, name="SE", dtype=tf.float32)
  UEx = tf.keras.Input(shape=(N,UE_D), name="UEx", dtype=tf.float32)
  UEy = tf.keras.Input(shape=(N,UE_D), name="UEy", dtype=tf.float32)
  training = tf.keras.Input(shape=(), batch_size=1, name="training", dtype=tf.bool)
  is_training = training[0]

  D = K*d
  # shape: [batch_size, P, N, F] -> [batch_size, P, N, D]
  Y = FC(X, units=[D,D], activations=[tf.nn.relu,None], bn=bn, bn_decay=bn_decay, is_training=is_training)
  # STE
  STE = STEmbedding(SE, TE, UEx, UEy, T, D, P, bn, bn_decay, is_training)
  assert STE.shape[1] == P+Q and STE.shape[2] == N and STE.shape[3] == D
  STE_P = STE[:, : P] # [batch_size, P, N, D]
  STE_Q = STE[:, P :] # [batch_size, Q, N, D]
  # encoder
  for _ in range(L):
    Y = STAttBlock(Y, STE_P, K, d, bn, bn_decay, is_training) # shape -> shape
  # transAtt
  Y = transformAttention(Y, STE_P, STE_Q, K, d, bn, bn_decay, is_training) # shape -> [batch_size, Q, N, D]
  # decoder
  for _ in range(L):
    Y = STAttBlock(Y, STE_Q, K, d, bn, bn_decay, is_training) # shape -> shape
  # output
  # shape -> [batch_size, Q, N, F]
  Y = FC(Y, units=[D,F], activations=[tf.nn.relu,None], bn=bn, bn_decay=bn_decay, is_training=is_training)
  return tf.keras.Model(
    inputs=[X, TE, SE, UEx, UEy, training],
    outputs=Y,
  )
