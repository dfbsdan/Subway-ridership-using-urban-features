import numpy as np
import tensorflow as tf
from UE import VAE_Encoder, Transformer

dir_path = '/mnt/Storage/Documents/thesis/v2'
data_path = dir_path + '/data'
reconstruction_loss = 'MAE' # 'MSE' # reconstruction loss used to train the model
load_path = dir_path + f'/models/urban_embedding/UE_{reconstruction_loss}.model'
offs_months = 7
Q = 1 # output sequence length
D = 64 # latent space dimension
L = 3 # number of STAtt blocks in the encoder/decoder
K = 8 # number of attention heads
d = 8 # dimension of each attention head outputs

# urban features
UF: np.ndarray = np.load(data_path + '/urban_features.npy') # (M, N, F)
M, N, F = UF.shape
P_max = M - offs_months - Q + 1
UF = np.transpose(UF, (1, 0, 2)) # (N, M, F)

encoder = VAE_Encoder(D, F, K*d)
encoder.load_weights(load_path.replace('.model', '_encoder.model'))
transformer = Transformer(1, P_max, D, L, K, d)
transformer.load_weights(load_path.replace('.model', '_transformer.model'))
embeddings = []
for n in range(N):
  batch = []
  for m in range(1, P_max+1):
    features = UF[n, :m, :] # (m, F)
    features = tf.pad(features, [[P_max - m, 0], [0,0]], 'CONSTANT') # (P_max, F)
    batch.append(features) 
  batch = np.stack(batch) # (P_max, P_max, F)
  assert batch.shape == (P_max, P_max, F)
  _, _, P_embeddings = encoder.encode(batch) # (P_max, P_max, D)
  assert P_embeddings.shape == (P_max, P_max, D)
  Q_embeddings = transformer(P_embeddings) # (P_max, 1, D)
  assert Q_embeddings.shape == (P_max, 1, D)
  P_embeddings = P_embeddings[:, -1, :] # (P_max, D)
  assert P_embeddings.shape == (P_max, D)
  P_embeddings = np.expand_dims(P_embeddings, 1) # (P_max, 1, D)
  assert P_embeddings.shape == (P_max, 1, D)
  PQ_embeddings = np.concatenate([P_embeddings, Q_embeddings], axis=1) # (P_max, 2, D)
  assert PQ_embeddings.shape == (P_max, 2, D)
  embeddings.append(PQ_embeddings)
embeddings = np.stack(embeddings) # (N, P_max, 2, D)
assert embeddings.shape == (N, P_max, 2, D)
embeddings = np.transpose(embeddings, (1, 2, 0, 3)) # (P_max, 2, N, D)
assert embeddings.shape == (P_max, 2, N, D)
assert not np.isnan(np.sum(embeddings))
np.save(data_path + f'/urban_embeddings_{reconstruction_loss}.npy', embeddings)