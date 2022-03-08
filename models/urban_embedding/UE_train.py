import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
from UE import TrVAE

dir_path = '/mnt/Storage/Documents/thesis/v2'
data_path = dir_path + '/data'
reconstruction_loss = 'MAE' # 'MSE'
save_path = dir_path + f'/models/urban_embedding/UE_{reconstruction_loss}.model'
learning_rate = 0.001
decay_epoch = 5
bn_decay = 0.9
patience = 10
max_epoch = 1000
batch_size = 16
offs_months = 7
rec_loss_weight = 1000
decay_rate = 0.7
Q = 1 # output sequence length
D = 64 # latent space dim
L = 3 # number of STAtt blocks in the encoder/decoder
K = 8 # number of attention heads
d = 8 # dimension of each attention head outputs

class CustomDataset(Sequence):
  def __init__(self, data: np.ndarray, batch_sz: int, Q: int) -> None:
    super().__init__()
    N, M, _ = data.shape
    self.data = data
    self.N = N
    self.Q = Q
    self.P_max = M - offs_months - Q + 1
    assert self.P_max > 0
    self.batch_sz = batch_sz
    self.len = math.ceil((N * self.P_max) / batch_sz)

  def __len__(self):
    return self.len

  def __get_xy(self, idx: int):
    n = idx // self.P_max
    if n >= self.N:
      return None
    P = idx % self.P_max + 1
    x = self.data[n, :P, :] # (P, F)
    x = tf.pad(x, [[self.P_max-P, 0], [0,0]], 'CONSTANT') # (P_max, F)
    y_start = P + offs_months - 1
    y = self.data[n, y_start:y_start + self.Q, :] # (Q, F)
    y = np.concatenate((x, y), axis=0) # (P_max+Q, F)
    return x, y

  def __getitem__(self, batch_idx: int):
    assert batch_idx < self.len
    start_idx = batch_idx * self.batch_sz
    batch = [self.__get_xy(idx) for idx in range(start_idx, start_idx + self.batch_sz)]
    batch = [xy for xy in batch if xy != None]
    return tf.stack([x for x, _ in batch]), tf.stack([y for _, y in batch])

# saves and evaluates the model every time the validation loss improves
class BestValLossCallback(tf.keras.callbacks.Callback):
  def __init__(self, test: np.ndarray, save_path: str):
    self.test = CustomDataset(test, batch_size, Q)
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

class MeanELBOLoss(tf.keras.losses.Loss):
  def __init__(self, mean, std) -> None:
    super().__init__()
    self.mean = mean
    self.std = std
    self.kl = tf.keras.losses.KLDivergence()
    reconst_losses = {
      'MSE': tf.keras.losses.MeanSquaredError, 
      'MAE': tf.keras.losses.MeanAbsoluteError
    }
    self.reconst = reconst_losses[reconstruction_loss]()

  def call(self, y_true, y_pred):
    y_true = y_true * self.std + self.mean
    y_pred = y_pred * self.std + self.mean
    # (batch, P+Q, F)
    return self.kl(y_true, y_pred) + rec_loss_weight * self.reconst(y_true, y_pred)

class CustomMetric(tf.keras.metrics.Metric):
  def __init__(self, mean: float, std: float, name: str) -> None:
    super(CustomMetric, self).__init__(name=name)
    self.mean = mean
    self.std = std
    metrics = {
      'MAE': tf.metrics.MeanAbsoluteError, 
      'MSE': tf.metrics.MeanSquaredError,
      'RMSE': tf.metrics.RootMeanSquaredError,
      'KL': tf.metrics.KLDivergence,
    }
    self.main = metrics[name]()
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.main.update_state(y_true * self.std + self.mean, y_pred * self.std + self.mean, sample_weight)
  
  def result(self):
    return self.main.result()

def train_and_test(model: tf.keras.Model, 
    train: np.ndarray, val: np.ndarray, test: np.ndarray):
  N_train = train.shape[0]
  mean = train.mean()
  std = train.std()
  train = (train - mean) / std
  val = (val - mean) / std
  test = (test - mean) / std
  model.compile(
    loss=MeanELBOLoss(mean, std),
    optimizer=tf.keras.optimizers.Adam(
      learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_epoch * (N_train*(M-offs_months)) // batch_size,
        decay_rate=decay_rate, 
        staircase=True
      )
    ),
    metrics=[
      CustomMetric(mean, std, 'MAE'),
      CustomMetric(mean, std, 'MSE'),
      CustomMetric(mean, std, 'RMSE'),
      CustomMetric(mean, std, 'KL'),
    ]
  )
  model.fit(
    CustomDataset(train, batch_size, Q),
    epochs=max_epoch,
    verbose=1,
    shuffle=True,
    validation_data=CustomDataset(val, batch_size, Q),
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
      ),
      BestValLossCallback(test, save_path),
    ],
  )

# urban features
UF: np.ndarray = np.load(data_path + '/urban_features.npy') # (M, N, F)
M, N, F = UF.shape
P_max = M - offs_months - Q + 1
UF = np.transpose(UF, (1, 0, 2)) # (N, M, F)
train_split = int(0.7 * len(UF))
val_split = int(0.1 * len(UF))  
train = UF[:train_split] # (N_train, M, F)
val = UF[train_split:train_split + val_split] # (N_val, M, F)
test = UF[train_split + val_split:] # (N_test, M, F)
print(f'train: {train.shape}, val: {val.shape}, test: {test.shape}')
model = TrVAE(Q=Q, D=D, L=L, K=K, d=d, F=F, P_max=P_max)
train_and_test(model, train, val, test)