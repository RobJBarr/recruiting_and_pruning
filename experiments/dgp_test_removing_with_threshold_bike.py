from conditional_variance import ConditionalVariance
from uci_datasets import Dataset
import gpflow
import gpflux
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


data = Dataset("bike")
x_train, y_train, x_test, y_test = data.get_split(split=5)
def get_model(x_train, Z1, Z2):
    # Layer 1
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=123.3552264398122, variance=56.945089537411704)
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z1.copy())
    gpflow.set_trainable(inducing_variable1, False)
    gp_layer1 = gpflux.layers.GPLayer(
    kernel1, inducing_variable1, num_data=x_train.shape[0], num_latent_gps=x_train.shape[1], whiten=False
    )
    # Layer 2
    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=54.057837771972025, variance=0.07618044828480358)
    inducing_variable2 = gpflow.inducing_variables.InducingPoints(Z2.copy())
    gpflow.set_trainable(inducing_variable2, False)
    gp_layer2 = gpflux.layers.GPLayer(
    kernel2,
    inducing_variable2,
    num_data=x_train.shape[0],
    num_latent_gps=x_train.shape[1],
    mean_function=gpflow.mean_functions.Zero(), whiten=False
    )

    # Initialise likelihood and build model
    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.1))
    m = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)

    # Compile and fit
    model = m.as_training_model()
    model.compile(tf.optimizers.Adam(0.01))
    return m, model



kernel1 = gpflow.kernels.SquaredExponential(lengthscales=123.3552264398122, variance=56.945089537411704)
Z1, _ = ConditionalVariance().compute_initialisation(x_train, 500, kernel1)

kernel2 = gpflow.kernels.SquaredExponential(lengthscales=54.057837771972025, variance=0.07618044828480358)
Z2, _ = ConditionalVariance().compute_initialisation(x_train, 500, kernel2)

m, model = get_model(x_train, Z1, Z2)

import tqdm
def train_deep_gp(deep_gp, data, maxiter=int(2000), plotter=None, plotter_interval=10):
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
m = train_deep_gp(m, (x_train, y_train))
m_params = gpflow.utilities.traversal.parameter_dict(m)
log_dir = "/vol/bitbucket/rjb19/recruiting_and_pruning/saved_models"
ckpt = tf.train.Checkpoint(full=m)
manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
()
Z1_new, _, deep_gp, elbos1, dis1 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 0, m, 0, True)


m2, model = get_model(x_train, Z1, Z2)
gpflow.utilities.multiple_assign(m2, m_params)
Z2_new, _, deep_gp, elbos2, dis2 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 1, m2, 0, True)

fig, axs = plt.subplots(1, 2, figsize=(20, 5), dpi=200)
plt.grid()
fig.suptitle('ELBOs of 1st and 2nd Layers as Points are Removed: Bike Dataset')
axs[0].grid(True)
axs[1].grid(True)
axs[0].plot(np.arange(500, 500-len(elbos1), step=-1), elbos1, label='lower bound')
axs[0].set_title(f"ELBO for First Layer, M={500}")
axs[0].set_ylabel("Marglik bound (nats)")
axs[0].set_xlabel("Number of inducing points $M$")
axs[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axs[0].invert_xaxis()
axs[0].legend(loc='upper right')
axs[1].plot(np.arange(500, 500-len(elbos2), step=-1), elbos2, label='lower bound')
axs[1].set_title(f"ELBO for Second Layer,  M={500}")
axs[1].set_ylabel("Marglik bound (nats)")
axs[1].set_xlabel("Number of inducing points $M$")
axs[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axs[1].invert_xaxis()
axs[1].legend(loc='upper right')
plt.tight_layout()

fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/dgp_with_threshold_bike5.png")