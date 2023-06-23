from conditional_variance import ConditionalVariance
import gpflow
import gpflux
import sys
import tqdm
import json
from scipy.cluster.vq import kmeans
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary
from scipy.cluster.vq import kmeans

from gpflux import callbacks
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

dataset_name = sys.argv[1]
from uci_datasets import Dataset
data = Dataset(dataset_name)
x_train, y_train, x_test, y_test = data.get_split(split=5)
sys.stdout.flush()
with open(f"hyperparameters/dgp_{dataset_name}_hyperparams.json", "r") as f:
    data = json.load(f)
lengthscales0 = np.array(data["0"]["lengthscales"])
variance0 = data["0"]["variance"]

lengthscales1 = np.array(data["1"]["lengthscales"])
variance1 = data["1"]["variance"]

def get_model(x_train, Z1, Z2):
    # Layer 1
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales0, variance=variance0)
    inducing_variable1 = gpflow.inducing_variables.InducingPoints(Z1.copy())
    gpflow.set_trainable(inducing_variable1, False)
    gp_layer1 = gpflux.layers.GPLayer(
    kernel1, inducing_variable1, num_data=x_train.shape[0], num_latent_gps=x_train.shape[1], whiten=False
    )
    # Layer 2
    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales1, variance=variance1)
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

def train_deep_gp(deep_gp, data, M, maxiter=int(4000), plotter=None, plotter_interval=10):
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
            tq.set_postfix_str(f"M: {M} objective: {objective_closure()}")
    return deep_gp

from scipy.cluster.vq import kmeans
Ms = [1, 2]#, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300]
models = []
Zs = []
for M in Ms:
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales0, variance=variance0)
    Z1, _ = ConditionalVariance().compute_initialisation(x_train, M, kernel1)

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=lengthscales1, variance=variance1)
    Z2, _ = ConditionalVariance().compute_initialisation(x_train, M, kernel2)

    m, model = get_model(x_train, Z1, Z2)
    m = train_deep_gp(m, (x_train, y_train), M)
    models.append(m)
    print(f" after {m.elbo((x_train, y_train))}")
    sys.stdout.flush()

with open(f"hyperparameters/sgpr_{dataset_name}_hyperparams.json", "r") as f:
    sgpr_data = json.load(f)
sgpr_lengthscales = sgpr_data["lengthscales"]
sgpr_variance = sgpr_data["variance"]
kern = gpflow.kernels.RBF(variance=sgpr_variance, lengthscales=sgpr_lengthscales)
Z, _ = ConditionalVariance().compute_initialisation(x_train, 3000, kern)
m = gpflow.models.SGPR((x_train, y_train),
kernel=kern,
inducing_variable=Z, likelihood=gpflow.likelihoods.Gaussian())
lml_task2 = ScalarToTensorBoard("logs/scipy/dgp/train", lambda: m.elbo(), f"sgp-elbo")
monitor2 = Monitor(
    MonitorTaskGroup([lml_task2], period=1)
)
gpflow.set_trainable(m.inducing_variable, False)

opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor2, options={"maxiter":2000})
sgp_elbo = m.elbo()

lbs = [m.elbo((x_train, y_train)) for m in models]


fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)
plt.grid()
axs.plot(Ms, lbs, label='DGP ELBO')
axs.set_title("ELBO of 2-layer DGP vs SGP: Elevators Dataset")
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")
axs.axhline(sgp_elbo, color='k', linestyle='--', label="ELBO of SVGP with 3000 inducing points", linewidth=2.0)
axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

axs.legend(loc='upper right')
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/dgp_vs_single_layer_gp_{dataset_name}3.png")


