from conditional_variance import ConditionalVariance
import gpflow
import gpflux
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

def get_model(x_train, Z1, Z2):
    # Layer 1
    kernel1 = gpflow.kernels.SquaredExponential(variance=11.549294367028518, lengthscales=[6.37919952e+01, 5.66161808e+01, 1.86719253e+00, 5.84993698e+01,
       6.57044876e+00, 1.51769330e+01, 4.17545977e+01, 3.41052947e+01,
       4.42534446e+01, 2.43949624e-03, 3.57300223e-02, 3.51140477e-02,
       1.36634293e-02, 1.82622128e+01, 4.51932699e+00, 2.17802432e-03,
       3.59586000e-01, 3.35527153e-03])
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z1.copy())
    gpflow.set_trainable(inducing_variable1, False)
    gp_layer1 = gpflux.layers.GPLayer(
    kernel1, inducing_variable1, num_data=x_train.shape[0], num_latent_gps=x_train.shape[1], whiten=False
    )
    # Layer 2
    kernel2 = gpflow.kernels.SquaredExponential(variance=0.01556443561934226, lengthscales=[32.21276111, 27.21183329, 12.27497103, 28.01006433,  7.70357752,
        6.02872358, 20.64083685, 28.09876175, 13.59814583, 28.25271207,
       28.07835146, 28.15605443, 28.04529149, 28.04160157, 27.98790662,
       28.17802824, 28.02799948, 28.20739588])
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

from uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)

kernel1 = gpflow.kernels.SquaredExponential(variance=11.549294367028518, lengthscales=[6.37919952e+01, 5.66161808e+01, 1.86719253e+00, 5.84993698e+01,
       6.57044876e+00, 1.51769330e+01, 4.17545977e+01, 3.41052947e+01,
       4.42534446e+01, 2.43949624e-03, 3.57300223e-02, 3.51140477e-02,
       1.36634293e-02, 1.82622128e+01, 4.51932699e+00, 2.17802432e-03,
       3.59586000e-01, 3.35527153e-03])
Z1, _ = ConditionalVariance().compute_initialisation(x_train, 450, kernel1)

kernel2 = gpflow.kernels.SquaredExponential(variance=0.01556443561934226, lengthscales=[32.21276111, 27.21183329, 12.27497103, 28.01006433,  7.70357752,
        6.02872358, 20.64083685, 28.09876175, 13.59814583, 28.25271207,
       28.07835146, 28.15605443, 28.04529149, 28.04160157, 27.98790662,
       28.17802824, 28.02799948, 28.20739588])
Z2, _ = ConditionalVariance().compute_initialisation(x_train, 450, kernel2)

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
            tq.set_postfix_str(f"objective: {objective_closure()}")
    return deep_gp
m = train_deep_gp(m, (x_train, y_train))
m_params = gpflow.utilities.traversal.parameter_dict(m)
log_dir = "/vol/bitbucket/rjb19/recruiting_and_pruning/saved_models"
ckpt = tf.train.Checkpoint(full=m)
manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
()
Z1_new, _, deep_gp, elbos1 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 0, m, 0)


m2, model = get_model(x_train, Z1, Z2)
gpflow.utilities.multiple_assign(m2, m_params)

Z2_new, _, deep_gp, elbos2 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 1, m2, 0)

fig, axs = plt.subplots(1, 2, figsize=(20, 5), dpi=200)
axs[0].grid(True)
axs[1].grid(True)
fig.suptitle('ELBOs 1st and 2nd Layers as Points are Removed: Elevators Dataset')
axs[0].plot(np.arange(450, 450-len(elbos1), step=-1), elbos1, label='lower bound')
axs[0].set_title(f"ELBO for First Layer, M={450}")
axs[0].set_ylabel("Marglik bound (nats)")
axs[0].set_xlabel("Number of inducing points $M$")
axs[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axs[0].invert_xaxis()
axs[0].legend(loc='upper right')
axs[1].plot(np.arange(450, 450-len(elbos2), step=-1), elbos2, label='lower bound')
axs[1].set_title(f"ELBO for Second Layer,  M={450}")
axs[1].set_ylabel("Marglik bound (nats)")
axs[1].set_xlabel("Number of inducing points $M$")
axs[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axs[1].invert_xaxis()
axs[1].legend(loc='upper right')
plt.tight_layout()

fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/dgp_with_threshold_elevators6.png")