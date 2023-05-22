# %%
import sys
import gpflux
import gpflow
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tools import datasets
from tools import plotting
import abc
from typing import Optional, Callable
from conditional_variance import ConditionalVariance


from uci_datasets.uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)
sys.stdout.flush()
x_train.shape



num_inducing = 500
def get_model(x_train, M, Z1, Z2):
    # Layer 1

    kernel1 = gpflow.kernels.SquaredExponential()
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z1.copy())
    gpflow.set_trainable(inducing_variable1, False)
    gp_layer1 = gpflux.layers.GPLayer(
    kernel1, inducing_variable1, num_data=x_train.shape[0], num_latent_gps=x_train.shape[1]
    )

    # Layer 2
    kernel2 = gpflow.kernels.SquaredExponential()
    inducing_variable2 = gpflow.inducing_variables.InducingPoints(Z2.copy())
    gpflow.set_trainable(inducing_variable2, False)
    gp_layer2 = gpflux.layers.GPLayer(
    kernel2,
    inducing_variable2,
    num_data=x_train.shape[0],
    num_latent_gps=x_train.shape[1],
    mean_function=gpflow.mean_functions.Zero(),
    )

    # Initialise likelihood and build model
    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
    m = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)

    # Compile and fit
    model = m.as_training_model()
    model.compile(tf.optimizers.Adam(0.01))
    return m, model

M = num_inducing
kernel1 = gpflow.kernels.SquaredExponential()
Z1, _ = ConditionalVariance().compute_initialisation(x_train, M, kernel1)

kernel2 = gpflow.kernels.SquaredExponential()
Z2, _ = ConditionalVariance().compute_initialisation(x_train, M, kernel2)

m, model = get_model(x_train, M, Z1, Z2)
model.compile(tf.optimizers.Adam(0.01))

# %%
from gpflux import callbacks
tb = callbacks.TensorBoard(log_dir="logs/scipy/dgp", keywords_to_monitor=["loss", "elbo"])

# %%
history = model.fit({"inputs": x_train, "targets": y_train}, epochs=int(100), verbose=0, callbacks=tb)

import pickle
params1 = gpflow.utilities.parameter_dict(m.f_layers[0].kernel)
params2 = gpflow.utilities.parameter_dict(m.f_layers[1].kernel)
with open("kernel1.pickle", "wb") as handle:
    pickle.dump(params1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("kernel2.pickle", "wb") as handle:
    pickle.dump(params2, handle, protocol=pickle.HIGHEST_PROTOCOL)
fig = plt.figure()
plt.plot(history.history["loss"])
fig.savefig(f"/vol/bitbucket/rjb19/conditional_elevators_tightness.png")


