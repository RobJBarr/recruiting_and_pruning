
import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import gpflow
from gpflow.utilities import print_summary
from conditional_variance import ConditionalVariance

from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)

from uci_datasets import Dataset
data = Dataset("elevators")
x_train, y_train, x_test, y_test = data.get_split(split=5)

M = 4000
kern = gpflow.kernels.RBF(variance=16467.93166587576, lengthscales=740.4639311514006)
Z, _ = ConditionalVariance().compute_initialisation(x_train, M, kern)
m = gpflow.models.SGPR((x_train, y_train),
kernel=kern,
inducing_variable=Z, likelihood=gpflow.likelihoods.Gaussian())
lml_task2 = ScalarToTensorBoard("logs/scipy/conditional", lambda: m.elbo(), f"remove-elbo-{M}-elevators")
monitor2 = Monitor(
    MonitorTaskGroup([lml_task2], period=1)
)
gpflow.set_trainable(m.inducing_variable, False)

opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, m.trainable_variables, step_callback=monitor2)
final_elbo = m.elbo()

_, _, _, lbs = ConditionalVariance().remove_points_sgpr((x_train, y_train), m, 0)

fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)
plt.grid()
axs.set_title('ELBO of SVGP as Points are Removed: Elevators Dataset')
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")
axs.legend(loc='upper right')
axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

axs.plot(np.arange(M, M-len(lbs), step=-1) , lbs, label='ELBO')
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/svgp_remove_inducing_elevators.png")
