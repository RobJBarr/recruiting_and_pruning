
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
from scipy.cluster.vq import kmeans
print("Here ")
sys.stdout.flush()

class ConditionalVariance():
    def __init__(self, sample: Optional[bool] = False, threshold: Optional[int] = 0.0, seed: Optional[int] = 0,
                 **kwargs):
        """
        :param sample: bool, if True, sample points into subset to use with weights based on variance, if False choose
        point with highest variance at each iteration
        :param threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx.
        has converged.
        """
        self.first_time = True
        self.randomized=True
        self.seed = seed if self.randomized else None
        self.sample = sample
        self.threshold = threshold

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        N = training_inputs.shape[0]
        perm = np.random.permutation(N)  # permute entries so tiebreaking is random
        training_inputs = training_inputs[perm]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = np.zeros(M, dtype=int) + N
        di = kernel(training_inputs, None, full_cov=False) + 1e-12  # jitter
        if self.sample:
            indices[0] = sample_discrete(di)
        else:
            indices[0] = np.argmax(di)  # select first point, add to index 0
        if M == 1:
            indices = indices.astype(int)
            Z = training_inputs[indices]
            indices = perm[indices]
            return Z, indices
        ci = np.zeros((M - 1, N))  # [M,N]
        for m in range(M - 1):
            j = int(indices[m])  # int
            new_Z = training_inputs[j:j + 1]  # [1,D]
            dj = np.sqrt(di[j])  # float
            cj = ci[:m, j]  # [m, 1]
            Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))
            L = np.round(np.squeeze(Lraw), 20)  # [N]
            L[j] += 1e-12  # jitter
            ei = (L - np.dot(cj, ci[:m])) / dj
            ci[m, :] = ei
            try:
                di -= ei ** 2
            except FloatingPointError:
                pass
            di = np.clip(di, 0, None)
            if self.sample:
                indices[m + 1] = sample_discrete(di)
            else:
                indices[m + 1] = np.argmax(di)
                a = di[np.where(di==di[indices[m+1]])]
                if len(a)>1:
                    print(len(a), m, (a==0.).all())
                    sys.stdout.flush()  # select first point, add to index 0
            # sum of di is tr(Kff-Qff), if this is small things are ok
            if np.sum(np.clip(di, 0, None)) < self.threshold:
                indices = indices[:m]
                warnings.warn("ConditionalVariance: Terminating selection of inducing points early.")
                break
        indices = indices.astype(int)
        Z = training_inputs[indices]
        indices = perm[indices]
        return Z, indices

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()
                            if
                            k not in ['_randomized'] and
                            not (k == "threshold" and self.threshold == 0.0)])
        return f"{type(self).__name__}({params})"

    def remove_points_increment(self, training_inputs: np.ndarray, kernel: gpflow.kernels.Kernel, num_to_remove: int, Z):
        N = training_inputs.shape[0]
        M = len(Z)-num_to_remove
        print(f"Removing {M}")
        sys.stdout.flush()
        # perm = np.random.permutation(N)  # permute entries so tiebreaking is random
        # training_inputs = training_inputs[perm]
        # note this will throw an out of bounds exception if we do not update each entry
        intersection = np.asarray([i for i,x in enumerate(training_inputs) for y in Z  if(np.array_equal(x, y))]).T
        while num_to_remove > 0:
            M = len(Z)
            indices = np.zeros(M, dtype=int) + N
            
            #TODO FIX cj/ci issues with correct indexing. Need to get real index as well as intersection index
            di = kernel(training_inputs, None, full_cov=False).numpy() + 1e-12  # jitter
            if self.sample:
                indices[0] = sample_discrete(di[intersection])
            else:
                indices[0] = np.argmax(di[intersection])
            ci = np.zeros((M - 1, N))  # [M,N]
            for m in range(M - 1):
                j = int(intersection[indices[m]])  # int
                new_Z = training_inputs[j:j + 1]  # [1,D]
                dj = np.sqrt(di[j])  # float
                cj = ci[:m, j]  # [m, 1]
                Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))
                L = np.round(np.squeeze(Lraw), 20)  # [N]
                L[j] += 1e-12  # jitter
                ei = (L - np.dot(cj, ci[:m])) / dj
                ci[m, :] = ei
                try:
                    di -= ei ** 2
                except FloatingPointError:
                    pass
                di = np.clip(di, 0, None)
                if self.sample:
                    indices[m + 1] = sample_discrete(di[intersection])
                else:
                    indices[m + 1] = np.argmax(di[intersection])# select first point, add to index 0
                # sum of di is tr(Kff-Qff), if this is small things are ok
            Z = np.delete(Z, indices[-1], 0)
            intersection = np.delete(intersection, indices[-1], 0)
            num_to_remove -= 1
        indices = indices.astype(int)
        print("Z shape:",Z.shape)
        sys.stdout.flush()
        return Z.copy()

    def remove_points_batch(self, training_inputs: np.ndarray, kernel: gpflow.kernels.Kernel, num_to_remove: int, Z):
        N = training_inputs.shape[0]
        M = len(Z)-num_to_remove
        print(f"Removing {M}")
        sys.stdout.flush()
        # perm = np.random.permutation(N)  # permute entries so tiebreaking is random
        # training_inputs = training_inputs[perm]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = np.zeros(M, dtype=int) + N
        intersection = np.asarray([i for i,x in enumerate(training_inputs) for y in Z  if(np.array_equal(x, y))]).T
        #TODO FIX cj/ci issues with correct indexing. Need to get real index as well as intersection index
        di = kernel(training_inputs, None, full_cov=False).numpy() + 1e-12  # jitter
        if self.sample:
            indices[0] = sample_discrete(di[intersection])
        else:
            indices[0] = np.argmax(di[intersection])
        ci = np.zeros((M - 1, N))  # [M,N]
        for m in range(M - 1):
            j = int(intersection[indices[m]])  # int
            new_Z = training_inputs[j:j + 1]  # [1,D]
            dj = np.sqrt(di[j])  # float
            cj = ci[:m, j]  # [m, 1]
            Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))
            L = np.round(np.squeeze(Lraw), 20)  # [N]
            L[j] += 1e-12  # jitter
            ei = (L - np.dot(cj, ci[:m])) / dj
            ci[m, :] = ei
            try:
                di -= ei ** 2
            except FloatingPointError:
                pass
            di = np.clip(di, 0, None)
            if self.sample:
                indices[m + 1] = sample_discrete(di[intersection])
            else:
                indices[m + 1] = np.argmax(di[intersection])  # select first point, add to index 0
            # sum of di is tr(Kff-Qff), if this is small things are ok 
        indices = indices.astype(int)
        Z = Z[indices].copy()
        print("Z shape:",Z.shape)
        sys.stdout.flush()
        return Z
# %%
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




Ms = [1,2, 3, 4, 5, 6, 7, 8, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
models = []
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
i = 0
for M in Ms:
    kern = gpflow.kernels.RBF(16453.294046883326, lengthscales=float(1058.5050695720367))
    Z = kmeans(x_train, M)[0]
    sys.stdout.flush()
    m = gpflow.models.SGPR(
    (x_train, y_train),
    kernel=kern,
    inducing_variable=Z)
    lml_task2 = ScalarToTensorBoard("logs/scipy/conditional", lambda: m.elbo(), f"elbo-{M}-kmeans")
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

# %%

fig, axs = plt.subplots(1, 1, figsize=(14, 3), dpi=200)

axs.plot(Ms, lbs, label='lower bound conditional')
axs.plot(Ms, ubs, label='upper bound conditional')
#axs[0].axvline(Ms[i], color='C3')
axs.set_title("Tightness of bounds")
axs.set_ylabel("Marglik bound (nats)")
axs.set_xlabel("Number of inducing points $M$")
# axs.axhline(lml, color='k', linestyle='--', label="marg. lik.", linewidth=2.0)

axs.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

axs.plot(Ms, new_lbs, label='lower bound removing kmeans')
axs.plot(Ms, new_ubs, label='upper bound removing kmeans')
axs.legend(loc='upper right')
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/conditional_vs_kmeans.png")
# saver.save('/vol/bitbucket/rjb19/elevators_conditional_model', m)
