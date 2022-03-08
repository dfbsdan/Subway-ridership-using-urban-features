from re import A
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math

data_path = '/mnt/Storage/Documents/thesis/v2/data'
learning_rate = 0.001
offs_months = 7
decay_epoch = 5
patience = 10
max_epoch = 1000
save_path = '/mnt/Storage/Documents/thesis/v2/models/fc/fc.model'
batch_size = 16
decay_rate = 0.7
random.seed(1)

class CustomDataset(Sequence):
  def __init__(self, x: np.ndarray, y: np.ndarray, batch_sz: int) -> None:
    super().__init__()
    assert x.shape == y.shape
    self.x = x
    self.y = y
    self.batch_sz = batch_sz
    self.len = math.ceil(x.shape[0] / batch_sz)

  def __len__(self):
    return self.len

  def __getitem__(self, batch_idx: int):
    assert batch_idx < self.len
    start_idx = batch_idx * self.batch_sz
    return self.x[start_idx:start_idx + self.batch_sz], self.y[start_idx:start_idx + self.batch_sz]

# saves and evaluates the model every time the validation loss improves
class BestValLossCallback(tf.keras.callbacks.Callback):
  def __init__(self, testX: np.ndarray, testY: np.ndarray, save_path: str):
    self.test = CustomDataset(testX, testY, batch_size)
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

class FC(tf.keras.Model):
  def __init__(self, N: int, H: int):
    super().__init__()
    self.main = tf.keras.Sequential([
      tf.keras.layers.Dense(2048, activation='relu', input_shape=(N*H*7*2,)),
      tf.keras.layers.Dense(2048, activation='relu'),
      tf.keras.layers.Dense(2048, activation='relu'),
      tf.keras.layers.Dense(N*H*7*2, activation='relu')
    ])
    
  def call(self, x: tf.Tensor):
    return self.main(x) # (batch, N*H*7*2) -> (batch, N*H*7*2)

def train_and_test(model: tf.keras.Model, 
    trainX: np.ndarray, trainY: np.ndarray, 
    valX: np.ndarray, valY: np.ndarray, 
    testX: np.ndarray, testY: np.ndarray):
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
  model.fit(
    CustomDataset(trainX, trainY, batch_size),
    epochs=max_epoch,
    verbose=1,
    validation_data=CustomDataset(valX, valY, batch_size),
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
      ),
      BestValLossCallback(testX, testY, save_path),
    ],
  )

ridership: np.ndarray = np.load(data_path + '/ridership.npy') # (M, N, H, 7, 2)
M, N, H = ridership.shape[:3]
ridership = np.reshape(ridership, (M, N*H*7*2)) # (M, N*H*7*2)
xy = []
for i in range(len(ridership)-offs_months):
  x = ridership[i] # (N*H*7*2)
  y = ridership[i+offs_months]
  xy.append(np.stack([x, y]))
xy = np.stack(xy) # (M-offs_monts, 2, N*H*7*2)
train_split = int(0.7 * len(xy))
val_split = int(0.1 * len(xy))  
for _ in range(5):
  random.shuffle(xy)
  train = xy[:train_split]
  val = xy[train_split:train_split + val_split]
  test = xy[train_split + val_split:]
  trainX = train[:, 0, :] # (train_len, N*H*7*2)
  trainY = train[:, 1, :] # (train_len, N*H*7*2)
  valX = val[:, 0, :] # (val_len, N*H*7*2)
  valY = val[:, 1, :] # (val_len, N*H*7*2)
  testX = test[:, 0, :] # (test_len, N*H*7*2)
  testY = test[:, 1, :] # (test_len, N*H*7*2)
  print(f'trainX: {trainX.shape}, trainY: {trainY.shape}, valX: {valX.shape}, valY: {valY.shape}, testX: {testX.shape}, testY: {testY.shape}')
  model = FC(N, H)
  train_and_test(model, trainX, trainY, valX, valY, testX, testY)