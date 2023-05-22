from conditional_variance import ConditionalVariance
import gpflow
import gpflux
import sys
print("Beginning script")
sys.stdout.flush()
from scipy.cluster.vq import kmeans
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary
from tools import datasets
from tools import plotting
import abc
from typing import Optional, Callable
from scipy.cluster.vq import kmeans

from gpflux import callbacks
tb = callbacks.TensorBoard(log_dir="logs/scipy/dgp", keywords_to_monitor=["loss", "elbo"])
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)


from uci_datasets.uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)
sys.stdout.flush()


def get_model(x_train, M, Z1, Z2):
    # Layer 1

    kernel1 = gpflow.kernels.SquaredExponential(1.1020279925373941, lengthscales=1.0000039340043378)
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z1.copy())
    gpflow.set_trainable(inducing_variable1, False)
    gp_layer1 = gpflux.layers.GPLayer(
    kernel1, inducing_variable1, num_data=x_train.shape[0], num_latent_gps=x_train.shape[1]
    )

    # Layer 2
    kernel2 = gpflow.kernels.SquaredExponential(7.781688426546651e-12, lengthscales=0.9379017362894123)
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


from scipy.cluster.vq import kmeans
Ms = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
models = []
Zs = []
for M in Ms:
    
    print(M)
    sys.stdout.flush()
    Z = kmeans(x_train, M)[0]
    kernel1 = gpflow.kernels.SquaredExponential(1.1020279925373941, lengthscales=1.0000039340043378)
    Z1, _ = ConditionalVariance().compute_initialisation(x_train, M, kernel1)

    kernel2 = gpflow.kernels.SquaredExponential(7.781688426546651e-12, lengthscales=0.9379017362894123)
    Z2, _ = ConditionalVariance().compute_initialisation(x_train, M, kernel2)
    
    m, model = get_model(x_train, M, Z1, Z2)
    
    history = model.fit({"inputs": x_train, "targets": y_train}, epochs=int(1e2), verbose=0, callbacks=tb)
    

    models.append(m)
    sys.stdout.flush()
cond_var = ConditionalVariance()

new_models = []
final_Z2 = Z2.copy()
final_Z1 = Z1.copy()

i = 0
for M in Ms[:-1]:
    print(M)
    sys.stdout.flush()
    kernel1 = gpflow.kernels.SquaredExponential(1.1020279925373941, lengthscales=1.0000039340043378)
    Z1 = ConditionalVariance().remove_points_batch(x_train, kernel2, Ms[-1]-M, final_Z1.copy())
    
    kernel2 = gpflow.kernels.SquaredExponential(7.781688426546651e-12, lengthscales=0.9379017362894123)
    Z2 = ConditionalVariance().remove_points_batch(x_train, kernel2, Ms[-1]-M, final_Z2.copy())

    m, model = get_model(x_train, M, Z1, Z2)
    history = model.fit({"inputs": x_train, "targets": y_train}, epochs=int(1e2), verbose=0, callbacks=tb)

    new_models.append(m)
    sys.stdout.flush()

new_lbs = [m.elbo((x_train, y_train)) for m in new_models]


lbs = [m.elbo((x_train, y_train)) for m in models]


fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)

axs.plot(Ms, lbs, label='lower bound dgp')

axs.set_title("Tightness of bounds")
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")

axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

axs.plot(Ms, new_lbs + [lbs[-1]], label='lower bound removing')
axs.legend(loc='upper right')
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/dgp_removing_inducing_kmeans.png")


# saver.save('/vol/bitbucket/rjb19/elevators_conditional_model', m)
