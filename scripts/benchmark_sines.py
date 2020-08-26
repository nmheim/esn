import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import esn.sparse_esn as se
from esn.input_map import InputMap
from esn.utils import split_train_label_pred

input_size = 1
Ntrans = 500
Ntrain = 2500
Npred  = 500
xs   = jnp.linspace(0,30*2*jnp.pi,Ntrain+Npred+1)
data = jnp.sin(xs)
data = data.reshape(-1, 1)


specs = [{"type":"random_weights", "input_size":input_size, "hidden_size":1500, "factor": 1.0}]
map_ih = InputMap(specs)
hidden_size = map_ih.output_size((input_size,))

t1 = time.time()
esn = se.esncell(map_ih, hidden_size, spectral_radius=1.5, density=0.05)
t2 = time.time()
print("Reservoir: ", t2-t1)

t1 = time.time()
esn = se.esncell(map_ih, hidden_size, spectral_radius=1.5, density=0.05)
t2 = time.time()
print("Reservoir: ", t2-t1)

t1 = time.time()
esn = se.esncell(map_ih, hidden_size, spectral_radius=1.5, density=0.05)
t2 = time.time()
print("Reservoir: ", t2-t1)



inputs, labels, pred_labels = split_train_label_pred(data,Ntrain,Npred)

t1 = time.time()
H = se.augmented_state_matrix(esn, inputs, Ntrans)
t2 = time.time()
print("State matrix: ", t2-t1)

t1 = time.time()
H = se.augmented_state_matrix(esn, inputs, Ntrans)
t2 = time.time()
print("State matrix: ", t2-t1)

t1 = time.time()
H = se.augmented_state_matrix(esn, inputs, Ntrans)
t2 = time.time()
print("State matrix: ", t2-t1)


t1 = time.time()
model = se.train(esn, H, labels[Ntrans:])
t2 = time.time()
print("Training: ", t2-t1)

t1 = time.time()
model = se.train(esn, H, labels[Ntrans:])
t2 = time.time()
print("Training: ", t2-t1)

t1 = time.time()
model = se.train(esn, H, labels[Ntrans:])
t2 = time.time()
print("Training: ", t2-t1)

y0 = labels[-1]
h0 = H[-1]
(y,h), (ys,hs) = se.predict(model, y0, h0, Npred)

# plt.plot(ys, label="Prediction")
# plt.plot(pred_labels.reshape(-1), label="Truth")
# plt.title("500 step prediction vs. truth")
# plt.legend()
# plt.show()

mse = jnp.mean((ys - pred_labels)**2)
print(mse)
