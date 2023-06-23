
import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
import json
from gpflow.utilities import print_summary
from conditional_variance import ConditionalVariance
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

import json
with open(f"hyperparameters/sgpr_{dataset_name}_hyperparams.json", "r") as f:
    data = json.load(f)
lengthscales = data["lengthscales"]
variance = data["variance"]
lml_task = ScalarToTensorBoard("logs/scipy", lambda: model.log_marginal_likelihood(), f"gpr-{dataset_name}")
monitor = Monitor(
    MonitorTaskGroup([lml_task], period=1)
)

kern = gpflow.kernels.SquaredExponential(lengthscales=lengthscales, variance=variance)
model = gpflow.models.GPR((x_train, y_train), kern)
opt = gpflow.optimizers.Scipy()
_ = opt.minimize(model.training_loss, variables=model.trainable_variables, step_callback=monitor)

lml = model.log_marginal_likelihood()



from scipy.cluster.vq import kmeans
Ms = [1,5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000]
models = []
for M in Ms:
    kern = gpflow.kernels.SquaredExponential(lengthscales=lengthscales, variance=variance)
    Z, _ = ConditionalVariance().compute_initialisation(x_train, M, kern)
    m = gpflow.models.SGPR((x_train, y_train),
    kernel=kern,
    inducing_variable=Z, likelihood=gpflow.likelihoods.Gaussian())
    lml_task = ScalarToTensorBoard("logs/scipy", lambda: m.elbo(), f"elbo-{M}-{dataset_name}")
    monitor = Monitor(
        MonitorTaskGroup([lml_task], period=1)
    )
    gpflow.set_trainable(m.inducing_variable, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor)
    models.append(m)

    sys.stdout.flush()

lbs = [m.elbo() for m in models]
lengthscales = [m.kernel.lengthscales.numpy() for m in models]
noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in models]

fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)

axs.plot(Ms, lbs, label='lower bound')
axs.set_title(f"ELBO Convergence: {dataset_name.title()} Dataset")
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")
axs.axhline(lml, color='k', linestyle='--', label="marg. lik.", linewidth=2.0)
axs.legend(loc='upper right')
axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/conditional_{dataset_name}_tightness.png")