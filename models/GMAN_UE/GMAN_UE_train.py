import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
from GMAN_UE import GMAN

dir_path = '/mnt/Storage/Documents/thesis/v2'
data_path = dir_path + '/data'
UE_loss = 'MAE' # 'MSE' # loss used to train the UE model
use_SE = True # use graph information (True) or not (False)
use_UE = True # use urban information (True) or not (False)
save_path = dir_path + f"/models/GMAN_UE/GMAN_{'SE' if use_SE else 'noSE'}_{'UE' + UE_loss if use_UE else 'noUE'}.model"
learning_rate = 0.001
decay_epoch = 5
bn_decay = None#0.9
patience = 10
max_epoch = 1000
offs_months = 7
batch_size = 16
decay_rate = 0.7
L = 3 # number of STAtt blocks in the encoder/decoder
K = 8 # number of attention heads
d = 8 # dimension of each attention head outputs
random.seed(1)

class CustomDataset(Sequence):
  def __init__(self, x: np.ndarray, y: np.ndarray, UEx: np.ndarray, UEy: np.ndarray, batch_sz: int, SE: np.ndarray, TE: np.ndarray, training: bool) -> None:
    super().__init__()
    assert x.shape == y.shape
    self.x = x # (seq_len, 7*H, N, 2)
    self.y = y # (seq_len, 7*H, N, 2)
    self.UEx = UEx # (seq_len, N, UE_D)
    self.UEy = UEy # (seq_len, N, UE_D)
    assert x.shape[0] == y.shape[0] and x.shape[0] == UEx.shape[0] and x.shape[0] == UEy.shape[0]
    self.SE = SE # (1, N, SE_D)
    self.TE = TE # (1, 2*7*H, 2)
    self.batch_sz = batch_sz
    self.len = math.ceil(x.shape[0] / batch_sz)
    self.training = np.array([training])

  def __len__(self):
    return self.len

  def __getitem__(self, batch_idx: int):
    assert batch_idx < self.len
    start_idx = batch_idx * self.batch_sz
    b_split = slice(start_idx, start_idx + self.batch_sz)
    X: np.ndarray = self.x[b_split]
    UEx: np.ndarray = self.UEx[b_split]
    Y: np.ndarray = self.y[b_split]
    UEy: np.ndarray = self.UEy[b_split]
    assert X.shape == Y.shape and X.shape[0] == UEx.shape[0] and X.shape[0] == UEy.shape[0]
    batch_sz = X.shape[0]
    return (
      {
        'X': X, # (batch, 7*H, N, 2)
        'TE': np.repeat(self.TE, batch_sz, axis=0), # (batch, 2*7*H, 2)
        'SE': self.SE, # (1, N, SE_D)
        'UEx': UEx, # (batch, N, UE_D)
        'UEy': UEy, # (batch, N, UE_D)
        'training': self.training,
      }, 
      Y # (batch, 7*H, N, 2)
    )

# saves and evaluates the model every time the validation loss improves
class BestValLossCallback(tf.keras.callbacks.Callback):
  def __init__(self, testX: np.ndarray, testY: np.ndarray, UEtestX: np.ndarray, UEtestY: np.ndarray, save_path: str, SE: np.ndarray, TE: np.ndarray):
    self.test = CustomDataset(testX, testY, UEtestX, UEtestY, batch_size, SE, TE, False)
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

class CustomMetric(tf.keras.metrics.Metric):
  def __init__(self, mean: float, std: float, name: str) -> None:
    super(CustomMetric, self).__init__(name=name)
    self.mean = mean
    self.std = std
    metrics = {
      'MAE': tf.metrics.MeanAbsoluteError, 
      'MSE': tf.metrics.MeanSquaredError,
      'RMSE': tf.metrics.RootMeanSquaredError,
    }
    self.main = metrics[name]()
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.main.update_state(y_true, y_pred * self.std + self.mean, sample_weight)
  
  def result(self):
    return self.main.result()

def train_and_test(model: tf.keras.Model, 
    trainX: np.ndarray, trainY: np.ndarray, UEtrainX: np.ndarray, UEtrainY: np.ndarray,
    valX: np.ndarray, valY: np.ndarray, UEvalX: np.ndarray, UEvalY: np.ndarray, 
    testX: np.ndarray, testY: np.ndarray, UEtestX: np.ndarray, UEtestY: np.ndarray, 
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
      CustomMetric(mean, std, 'MAE'),
      CustomMetric(mean, std, 'MSE'),
      CustomMetric(mean, std, 'RMSE'),
    ]
  )
  #tf.keras.utils.plot_model(model, save_path.replace('.model', '.png'), show_shapes=True)
  #exit(0)
  model.fit(
    CustomDataset(trainX, trainY, UEtrainX, UEtrainY, batch_size, SE, TE, True),
    epochs=max_epoch,
    verbose=1,
    validation_data=CustomDataset(valX, valY, UEvalX, UEvalY, batch_size, SE, TE, False),
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
      ),
      BestValLossCallback(testX, testY, UEtestX, UEtestY, save_path, SE, TE),
    ],
  )

# ridership
ridership: np.ndarray = np.load(data_path + '/ridership.npy') # (M, N, H, 7, 2)
M, N, H = ridership.shape[:3]
ridership = np.transpose(ridership, (0, 3, 2, 1, 4)) # (M, 7, H, N, 2)
ridership = np.reshape(ridership, (M, 7*H, N, 2)) # (M, 7*H, N, 2)
xy = []
for i in range(len(ridership)-offs_months):
  x = ridership[i] # (7*H, N, 2)
  y = ridership[i+offs_months]
  xy.append(np.stack([x, y])) # (2, 7*H, N, 2)
xy = np.stack(xy) # (M-offs_months, 2, 7*H, N, 2)
# urban embeddings
UE: np.ndarray = np.load(data_path + f'/urban_embeddings_{UE_loss}.npy') # (M-offs_months, 2, N, UE_D)
UE_D = UE.shape[3]
UE = UE if use_UE else np.zeros((xy.shape[0], 2, N, UE_D))
assert UE.shape[0] == xy.shape[0] and xy.shape[0] == M-offs_months
xy_idxs = np.array([i for i in range(xy.shape[0])])
# temporal embeddings
TE = np.array([[day, hour] for day in range(7) for hour in range(H)]) # (7*H, 2)
TE = np.tile(TE, (2, 1)) # (2*7*H, 2)
TE = np.expand_dims(TE, axis=0) # (1, 2*7*H, 2)
# spatial embeddings
SE: np.ndarray = np.load(data_path + '/spatial_embeddings.npy') # (N, SE_D)
SE_D = SE.shape[1]
SE = np.expand_dims(SE, axis=0) if use_SE else np.zeros((1, N, SE_D))  # (1, N, SE_D)
assert N == SE.shape[1]

train_split = int(0.7 * len(xy))
val_split = int(0.1 * len(xy))  
for _ in range(5):
  random.shuffle(xy_idxs)
  xy_ = xy[xy_idxs]
  UE_ = UE[xy_idxs]
  train = xy_[:train_split]
  val = xy_[train_split:train_split + val_split]
  test = xy_[train_split + val_split:]
  UEtrain = UE_[:train_split]
  UEval = UE_[train_split:train_split + val_split]
  UEtest = UE_[train_split + val_split:]

  trainX = train[:, 0, :, :] # (train_len, 7*H, N, 2)
  trainY = train[:, 1, :, :] # (train_len, 7*H, N, 2)
  UEtrainX = UEtrain[:, 0, :, :] # (train_len, N, D)
  UEtrainY = UEtrain[:, 1, :, :] # (train_len, N, D)

  valX = val[:, 0, :, :] # (val_len, 7*H, N, 2)
  valY = val[:, 1, :, :] # (val_len, 7*H, N, 2)
  UEvalX = UEval[:, 0, :, :] # (val_len, N, UE_D)
  UEvalY = UEval[:, 1, :, :] # (val_len, N, UE_D)

  testX = test[:, 0, :, :] # (test_len, 7*H, N, 2)
  testY = test[:, 1, :, :] # (test_len, 7*H, N, 2)
  UEtestX = UEtest[:, 0, :, :] # (test_len, N, D)
  UEtestY = UEtest[:, 1, :, :] # (test_len, N, D)
  print(f'trainX: {trainX.shape}, trainY: {trainY.shape}, valX: {valX.shape}, valY: {valY.shape}, testX: {testX.shape}, testY: {testY.shape}')
  model = GMAN(P=7*H, Q=7*H, N=N, F=2, SE_D=SE_D, UE_D=UE_D, T=H, L=L, K=K, d=d, bn=False if bn_decay == None else True, bn_decay=bn_decay)
  train_and_test(model, 
    trainX, trainY, UEtrainX, UEtrainY, 
    valX, valY, UEvalX, UEvalY, 
    testX, testY, UEtestX, UEtestY, 
    SE, TE)