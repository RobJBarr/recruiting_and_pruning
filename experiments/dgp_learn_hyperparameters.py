from conditional_variance import ConditionalVariance
import gpflow
import gpflux
import sys
import tqdm
from scipy.cluster.vq import kmeans
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
from uci_datasets import Dataset

dataset_name = sys.argv[1]

data = Dataset(dataset_name)
x_train, y_train, x_test, y_test = data.get_split(split=5)

def get_model(x_train, Z1, Z2):
    # Layer 1

    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1.]*x_train.shape[1], variance=1.0)
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z1.copy())
    gpflow.set_trainable(inducing_variable1, False)
    gp_layer1 = gpflux.layers.GPLayer(
    kernel1, inducing_variable1, num_data=x_train.shape[0], num_latent_gps=x_train.shape[1]
    )

    # Layer 2
    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=[1.]*x_train.shape[1], variance=1.0)
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

def train_deep_gp(deep_gp, data, maxiter=int(8000), plotter=None, plotter_interval=10):
    optimizer = tf.optimizers.Adam(0.01)

    @tf.function(autograph=False)
    def objective_closure():
        return -deep_gp.elbo(data)

    @tf.function
    def step():
        optimizer.minimize(objective_closure, deep_gp.trainable_variables)

    tq = tqdm.tqdm(range(maxiter))
    for i in tq:
        step()
        if i % plotter_interval == 0:
            sys.stdout.flush()
            tq.set_postfix_str(f"objective: {objective_closure()}")
    return deep_gp

kernel1 = gpflow.kernels.SquaredExponential(lengthscales=[1.]*x_train.shape[1], variance=1.0)
Z1, _ = ConditionalVariance().compute_initialisation(x_train, 300, kernel1)

kernel2 = gpflow.kernels.SquaredExponential(lengthscales=[1.]*x_train.shape[1], variance=1.0)
Z2, _ = ConditionalVariance().compute_initialisation(x_train, 300, kernel2)

m, model = get_model(x_train, Z1, Z2)
m = train_deep_gp(m, (x_train, y_train))
lengthscale_0 = m.f_layers[0].kernel.lengthscales.numpy().tolist()
lengthscale_1 = m.f_layers[1].kernel.lengthscales.numpy().tolist()
var_0 = m.f_layers[0].kernel.variance.numpy().tolist()
var_1 = m.f_layers[1].kernel.variance.numpy().tolist()

import json
kernel_hyperparams = {0: {"lengthscales": lengthscale_0, "variance": var_0}, 1: {"lengthscales": lengthscale_1, "variance": var_1}}
with open(f"hyperparameters/dgp_{dataset_name}_hyperparams.json", "w") as f:
    json.dump(kernel_hyperparams, f)