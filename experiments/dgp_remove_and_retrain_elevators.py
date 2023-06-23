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
Z1, _ = ConditionalVariance().compute_initialisation(x_train, 250, kernel1)

kernel2 = gpflow.kernels.SquaredExponential(variance=0.01556443561934226, lengthscales=[32.21276111, 27.21183329, 12.27497103, 28.01006433,  7.70357752,
        6.02872358, 20.64083685, 28.09876175, 13.59814583, 28.25271207,
       28.07835146, 28.15605443, 28.04529149, 28.04160157, 27.98790662,
       28.17802824, 28.02799948, 28.20739588])
Z2, _ = ConditionalVariance().compute_initialisation(x_train, 250, kernel2)

m, _ = get_model(x_train, Z1, Z2)

import tqdm
def train_deep_gp(deep_gp, data, maxiter=int(6000), plotter=None, plotter_interval=10):
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
final_elbo = m.elbo((x_train, y_train))
percentages = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
elbos_before_retrain1 = []
elbos_after_retrain1 = []
elbos_before_retrain2 = []
elbos_after_retrain2 = []
for p in percentages:
    print(p)
    new_model1, _ = get_model(x_train, Z1, Z2)
    gpflow.utilities.multiple_assign(new_model1, m_params)
    Z1_new, _, new_model1, elbos1 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 0, new_model1, p)
    elbos_before_retrain1.append(elbos1[-1])
    deep_gp1 = train_deep_gp(new_model1, (x_train, y_train), maxiter = 50)
    elbos_after_retrain1.append(new_model1.elbo((x_train, y_train)))

    new_model2, _ = get_model(x_train, Z1, Z2)
    gpflow.utilities.multiple_assign(new_model2, m_params)
    Z2_new, _, new_model2, elbos2 = ConditionalVariance().remove_points_deep_gp((x_train, y_train), 1, new_model1, p)
    elbos_before_retrain2.append(elbos2[-1])
    deep_gp2 = train_deep_gp(new_model2, (x_train, y_train), maxiter = 50)
    elbos_after_retrain2.append(new_model2.elbo((x_train, y_train)))

fig, axs = plt.subplots(1, 2, figsize=(20, 5), dpi=200)
plt.grid()
axs[0].grid(True)
axs[1].grid(True)
fig.suptitle("ELBO of DGP Before and After Retraining: Elevators Dataset, 250 Inducing Points in Each Layer")
axs[0].set_title('ELBO of DGP Before and After Retraining First Layer')
axs[0].set_ylabel("ELBO")
axs[0].set_xlabel("Fraction of Original ELBO Before Retraining")
axs[1].set_title('ELBO of DGP Before and After Retraining Second Layer')
axs[1].set_ylabel("ELBO")
axs[1].set_xlabel("Fraction of Original ELBO Before Retraining")

axs[0].plot( [1] + percentages, [final_elbo] + elbos_before_retrain1, label='ELBO Before Retraining')
axs[0].plot( [1] + percentages, [final_elbo] + elbos_after_retrain1, label='ELBO After Retraining')
axs[0].legend(loc='upper left')
axs[1].plot( [1] + percentages, [final_elbo] + elbos_before_retrain2, label='ELBO Before Retraining')
axs[1].plot( [1] + percentages, [final_elbo] + elbos_after_retrain2, label='ELBO After Retraining')
axs[1].legend(loc='upper left')
axs[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
axs[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
plt.tight_layout()
fig.savefig(f"/vol/bitbucket/rjb19/recruiting_and_pruning/figs/dgp_remove_and_retrain_elevators4.png")

