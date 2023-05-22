
import sys
print("Beginning script")
sys.stdout.flush()

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
from conditional_variance import ConditionalVariance
print("Here ")
sys.stdout.flush()

# %%
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

# %%
from uci_datasets.uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)
sys.stdout.flush()
# %%
log_dir_scipy = "logs/scipy/conditional"
lml_task = ScalarToTensorBoard(log_dir_scipy, lambda: full.training_loss(), "training_objective")
monitor = Monitor(
    MonitorTaskGroup([lml_task], period=1)
)
full = gpflow.models.GPR((x_test, y_test), gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()
_ = opt.minimize(full.training_loss, variables=full.trainable_variables, step_callback=monitor)

log_dir = "/vol/bitbucket/rjb19"
ckpt = tf.train.Checkpoint(full=full)
manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=5)
print("here 43" , log_dir)
print("asdfasdfasdfasdfasdfasdf")
sys.stdout.flush()
print(manager.save())
sys.stdout.flush()
# # %%
inducing = x_train[:50,:].copy()
gpflow.kernels.RBF(x_train.shape[0], lengthscales=float(x_train.shape[0])**0.5)
sparse = gpflow.models.SVGP(gpflow.kernels.SquaredExponential(), gpflow.likelihoods.Gaussian(), inducing,whiten=False)

# %%
elbo = tf.function(sparse.elbo)
tensor_data = tuple(map(tf.convert_to_tensor, (x_train,y_train)))
elbo(tensor_data)

# %%
log_dir_scipy = "logs/scipy/conditional"
sys.stdout.flush()
# %%
from scipy.cluster.vq import kmeans
Ms = [1,2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000]
models = []
Zs = []
for M in Ms:
    kern = gpflow.kernels.RBF(16453.294046883326, lengthscales=float(1058.5050695720367))
    Z, _ = ConditionalVariance().compute_initialisation(x_train, M, kern)
    m = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=kern,
    inducing_variable=Z)
    lml_task2 = ScalarToTensorBoard("logs/scipy/conditional", lambda: m.elbo(), f"elbo-{M}")
    monitor2 = Monitor(
        MonitorTaskGroup([lml_task2], period=1)
    )
    gpflow.set_trainable(m.inducing_variable, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor2, options={"maxiter":500})
    models.append(m)
    print(M)
    sys.stdout.flush()
cond_var = ConditionalVariance()
params = gpflow.utilities.parameter_dict(models[-1])
new_models = []
final_Z = Z.copy()
final_kern = gpflow.utilities.parameter_dict(kern)
for M in Ms[:-1]:
    kern = gpflow.kernels.RBF(16453.294046883326, lengthscales=float(1058.5050695720367))
    Z, _ = cond_var.remove_points_batch(x_train, kern, Ms[-1]-M, final_Z.copy())
    print("Num Z:", len(Z))
    sys.stdout.flush()
    m = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=kern,
    inducing_variable=Z)
    lml_task2 = ScalarToTensorBoard("logs/scipy/conditional", lambda: m.elbo(), f"elbo-{M}-removing")
    monitor2 = Monitor(
        MonitorTaskGroup([lml_task2], period=1)
    )
    gpflow.set_trainable(m.inducing_variable, False)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor2, options={"maxiter":500})
    new_models.append(m)
    print(M)
    sys.stdout.flush()
# %%
new_lbs = [m.elbo() for m in new_models]
new_ubs = [m.upper_bound() for m in new_models]
new_lengthscales = [m.kernel.lengthscales.numpy() for m in new_models]
new_noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in new_models]

lbs = [m.elbo() for m in models]
ubs = [m.upper_bound() for m in models]
lengthscales = [m.kernel.lengthscales.numpy() for m in models]
noise_stds = [m.likelihood.variance.numpy() ** 0.5 for m in models]

# %%
lml = full.log_marginal_likelihood()

# %%

fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)

axs.plot(Ms, lbs, label='lower bound')
axs.plot(Ms, ubs, label='upper bound')
#axs[0].axvline(Ms[i], color='C3')
axs.set_title("Tightness of bounds")
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")
# axs.axhline(lml, color='k', linestyle='--', label="marg. lik.", linewidth=2.0)
axs.legend(loc='upper right')
axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

axs.plot(Ms, new_lbs + [lbs[-1]], label='lower bound removing')
axs.plot(Ms, new_ubs + [ubs[-1]], label='upper bound removing')
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/conditional_remove_inducing3.png")
# saver.save('/vol/bitbucket/rjb19/elevators_conditional_model', m)
