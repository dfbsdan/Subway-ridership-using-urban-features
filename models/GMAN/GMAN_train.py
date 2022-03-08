import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
from GMAN import GMAN

data_path = '/mnt/Storage/Documents/thesis/v2/data'
save_path = '/mnt/Storage/Documents/thesis/v2/models/GMAN/GMAN.model'
learning_rate = 0.001
decay_epoch = 5
bn_decay = 0.9
patience = 10
max_epoch = 1000
batch_size = 16
decay_rate = 0.7
no_SE = True # use graph information (False) or not (True)
L = 3 # number of STAtt blocks in the encoder/decoder
K = 8 # number of attention heads
d = 8 # dimension of each attention head outputs
random.seed(1)

class CustomDataset(Sequence):
  def __init__(self, x: np.ndarray, y: np.ndarray, batch_sz: int, SE: np.ndarray, TE: np.ndarray, training: bool) -> None:
    super().__init__()
    assert x.shape == y.shape
    self.x = x
    self.y = y
    self.SE = SE
    self.TE = TE
    self.batch_sz = batch_sz
    self.len = math.ceil(x.shape[0] / batch_sz)
    self.training = np.array([training])

  def __len__(self):
    return self.len

  def __getitem__(self, batch_idx: int):
    assert batch_idx < self.len
    start_idx = batch_idx * self.batch_sz
    X: np.ndarray = self.x[start_idx:start_idx + self.batch_sz]
    Y: np.ndarray = self.y[start_idx:start_idx + self.batch_sz]
    assert X.shape == Y.shape
    batch_sz = X.shape[0]
    return (
      {
        'X': X, # (batch, 7*H, N, 2)
        'TE': np.repeat(self.TE, batch_sz, axis=0), # (batch, 2*7*H, 2)
        'SE': self.SE, # (1, N, SE_D)
        'training': self.training,
      }, 
      Y # (batch, 7*H, N, 2)
    )

# saves and evaluates the model every time the validation loss improves
class BestValLossCallback(tf.keras.callbacks.Callback):
  def __init__(self, testX: np.ndarray, testY: np.ndarray, save_path: str, SE: np.ndarray, TE: np.ndarray):
    self.test = CustomDataset(testX, testY, batch_size, SE, TE, False)
    self.save_path = save_path
    self.best_val_loss = np.Inf
  
  def on_epoch_end(self, epoch, logs=None):
    cur_val_loss = logs.get("val_loss")
    if cur_val_loss < self.best_val_loss:
      self.best_val_loss = cur_val_loss
      self.model.save_weights(self.save_path, overwrite=True)
      if epoch >= 10:
        self.model.evaluate(
          self.test,
          verbose=1,
        )

class MSELoss(tf.keras.losses.Loss):
  def __init__(self, mean, std) -> None:
    super().__init__()
    self.mean = mean
    self.std = std
    self.loss = tf.losses.MeanSquaredError()

  def call(self, y_true, y_pred):
    return self.loss(y_true, y_pred * self.std + self.mean)

class MAEMetric(tf.keras.metrics.Metric):
  def __init__(self, mean, std, name='MAE', **kwargs):
    super(MAEMetric, self).__init__(name=name, **kwargs)
    self.main = tf.metrics.MeanAbsoluteError()
    self.mean = mean
    self.std = std
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.main.update_state(y_true, y_pred * self.std + self.mean, sample_weight)
  
  def result(self):
    return self.main.result()

class MSEMetric(tf.keras.metrics.Metric):
  def __init__(self, mean, std, name='MSE', **kwargs):
    super(MSEMetric, self).__init__(name=name, **kwargs)
    self.main = tf.metrics.MeanSquaredError()
    self.mean = mean
    self.std = std
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.main.update_state(y_true, y_pred * self.std + self.mean, sample_weight)
  
  def result(self):
    return self.main.result()

class RMSEMetric(tf.keras.metrics.Metric):
  def __init__(self, mean, std, name='RMSE', **kwargs):
    super(RMSEMetric, self).__init__(name=name, **kwargs)
    self.main = tf.metrics.RootMeanSquaredError()
    self.mean = mean
    self.std = std
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.main.update_state(y_true, y_pred * self.std + self.mean, sample_weight)
  
  def result(self):
    return self.main.result()

def train_and_test(model: tf.keras.Model, 
    trainX: np.ndarray, trainY: np.ndarray, 
    valX: np.ndarray, valY: np.ndarray, 
    testX: np.ndarray, testY: np.ndarray,
    SE: np.ndarray, TE: np.ndarray):
  mean = trainX.mean()
  std = trainX.std()
  trainX = (trainX - mean) / std
  valX = (valX - mean) / std
  testX = (testX - mean) / std
  model.compile(
    loss=MSELoss(mean, std),
    optimizer=tf.keras.optimizers.Adam(
      learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_epoch * trainX.shape[0] // batch_size,
        decay_rate=decay_rate, 
        staircase=True
      )
    ),
    metrics=[
      MAEMetric(mean, std),
      MSEMetric(mean, std),
      RMSEMetric(mean, std)
    ]
  )
  #tf.keras.utils.plot_model(model, save_path.replace('.model', '.png'), show_shapes=True)
  #exit(0)
  model.fit(
    CustomDataset(trainX, trainY, batch_size, SE, TE, True),
    epochs=max_epoch,
    verbose=1,
    validation_data=CustomDataset(valX, valY, batch_size, SE, TE, False),
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
      ),
      BestValLossCallback(testX, testY, save_path, SE, TE),
    ],
  )

# ridership
ridership: np.ndarray = np.load(data_path + '/ridership.npy') # (M, N, H, 7, 2)
M, N, H = ridership.shape[:3]
ridership = np.transpose(ridership, (0, 3, 2, 1, 4)) # (M, 7, H, N, 2)
ridership = np.reshape(ridership, (M, 7*H, N, 2)) # (M, 7*H, N, 2)
xy = []
for i in range(len(ridership)-1):
  xy.append(ridership[i:i+2])
xy = np.stack(xy) # (_, 2, 7*H, N, 2)
# temporal embeddings
TE = np.array([[day, hour] for day in range(7) for hour in range(H)]) # (7*H, 2)
TE = np.tile(TE, (2, 1)) # (2*7*H, 2)
TE = np.expand_dims(TE, axis=0) # (1, 2*7*H, 2)
# spatial embeddings
SE: np.ndarray = np.load(data_path + '/spatial_embeddings.npy') # (N, SE_D)
SE_D = SE.shape[1]
SE = np.zeros((1, N, SE_D)) if no_SE else np.expand_dims(SE, axis=0) # (1, N, SE_D)
assert N == SE.shape[1]

train_split = int(0.7 * len(xy))
val_split = int(0.1 * len(xy))  
for _ in range(5):
  random.shuffle(xy)
  train = xy[:train_split]
  val = xy[train_split:train_split + val_split]
  test = xy[train_split + val_split:]
  trainX = train[:, 0, :, :] # (train_len, 7*H, N, 2)
  trainY = train[:, 1, :, :] # (train_len, 7*H, N, 2)
  valX = val[:, 0, :, :] # (val_len, 7*H, N, 2)
  valY = val[:, 1, :, :] # (val_len, 7*H, N, 2)
  testX = test[:, 0, :, :] # (test_len, 7*H, N, 2)
  testY = test[:, 1, :, :] # (test_len, 7*H, N, 2)
  print(f'trainX: {trainX.shape}, trainY: {trainY.shape}, valX: {valX.shape}, valY: {valY.shape}, testX: {testX.shape}, testY: {testY.shape}')
  model = GMAN(P=7*H, Q=7*H, N=N, F=2, SE_D=SE_D, T=H, L=L, K=K, d=d, bn=True, bn_decay=bn_decay)
  train_and_test(model, trainX, trainY, valX, valY, testX, testY, SE, TE)