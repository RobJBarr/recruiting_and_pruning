from typing import Optional, Callable
import numpy as np
import gpflow
import gpflux
import sys
import tensorflow as tf
class ConditionalVariance():
    def __init__(self, sample: Optional[bool] = False, threshold: Optional[int] = 0.0, seed: Optional[int] = 0,
                 **kwargs):
        """
        :param sample: bool, if True, sample points into subset to use with weights based on variance, if False choose
        point with highest variance at each iteration
        :param threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx.
        has converged.
        """
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
            # indices = perm[indices]
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
                indices[m + 1] = np.argmax(di)  # select first point, add to index 0
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
        intersection = np.asarray([i for y in Z for i,x in enumerate(training_inputs) if(np.array_equal(x, y))]).T
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

        return Z.copy(), indices

    def remove_points_batch(self, training_inputs: np.ndarray, kernel: gpflow.kernels.Kernel, num_to_remove: int, Z):
        N = training_inputs.shape[0]
        M = len(Z)-num_to_remove
        perm = np.random.permutation(N)  # permute entries so tiebreaking is random
        training_inputs = training_inputs[perm]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = np.zeros(M, dtype=int) + N
        intersection = np.asarray([i for y in Z for i,x in enumerate(training_inputs) if(np.array_equal(x, y))]).T
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
        return Z, intersection[indices]

    def remove_points_sgpr(self, training_inputs, gp, threshold, return_di=False):
        kernel = gp.kernel
        x_train = training_inputs[0]
        Z = gp.inducing_variable.Z.numpy()
        N = x_train.shape[0]
        intersection = np.asarray([i for y in Z for i,x in enumerate(x_train) if(np.array_equal(x, y))]).T
        test_Z = Z.copy()
        test_intersection = intersection.copy()
        elbo_base = gp.elbo()
        elbo_curr = gp.elbo()
        elbos = []
        dis = []
        i = 0
        while (np.abs(elbo_base - elbo_curr)/np.abs(elbo_base) <= 1 - threshold or elbo_curr >= elbo_base) and len(test_Z) > 1:
            elbos.append(elbo_curr)
            Z = test_Z.copy()
            gp.inducing_variable = gpflow.inducing_variables.InducingPoints(Z)

            intersection = test_intersection.copy()
            M = len(Z)
            remaining_points = np.array(range(len(intersection)))
            indices = np.zeros(M, dtype=int) + N
            di = kernel(x_train, None, full_cov=False).numpy() + 1e-12  # jitter
            if self.sample:
                indices[0] = sample_discrete(di[intersection][remaining_points])
                remaining_points = np.delete(remaining_points,(indices[0]), 0)
            else:
                indices[0] = np.argmax(di[intersection][remaining_points])
                remaining_points = np.delete(remaining_points,(indices[0]), 0)
            ci = np.zeros((M - 2, N))  # [M,N]
            for m in range(M - 2):
                j = int(intersection[indices[m]])  # int
                new_Z = x_train[j:j + 1]  # [1,D]
                dj = np.sqrt(di[j])  # float
                cj = ci[:m, j]  # [m, 1]
                Lraw = np.array(kernel(x_train, new_Z, full_cov=True))
                L = np.round(np.squeeze(Lraw), 20)  # [N]
                L[j] += 1e-12  # jitter
                dj += 1e-12
                ei = (L - np.dot(cj, ci[:m])) / dj
                ci[m, :] = ei
                try:
                    di -= ei ** 2
                except FloatingPointError:
                    pass
                di = np.clip(di, 0, None)
                if self.sample:
                    indices[m + 1] = sample_discrete(di[intersection][remaining_points])
                    remaining_points = np.delete(remaining_points,(indices[m+1]), 0)
                else:
                    indices[m + 1] = np.argmax(di[intersection][remaining_points])
                    remaining_points = np.delete(remaining_points,(indices[m+1]), 0)
            print(i, elbo_base, elbo_curr)
            sys.stdout.flush()
            i+=1
            idx_to_remove = next(iter(remaining_points))
            dis.append(di[intersection][idx_to_remove])
            test_Z = np.delete(Z, idx_to_remove, 0)
            test_intersection = np.delete(intersection, idx_to_remove, 0)


            gp.inducing_variable = gpflow.inducing_variables.InducingPoints(test_Z)
            elbo_curr = gp.elbo()

            gp.inducing_variable = gpflow.inducing_variables.InducingPoints(Z)

        if np.abs(elbo_base - elbo_curr)/np.abs(elbo_base) <= 1 - threshold and len(test_Z) >= 1:
            Z = test_Z.copy()
            intersection = test_intersection.copy()
            elbos.append(elbo_curr)
        if return_di:
            return Z.copy(), indices, gp, elbos, dis
        return Z.copy(), indices, gp, elbos

    def remove_points_svgp(self, training_inputs, gp, threshold, return_di=False):
        kernel = gp.kernel
        x_train = training_inputs[0]
        Z = gp.inducing_variable.Z.numpy()
        N = x_train.shape[0]
        intersection = np.asarray([i for y in Z for i,x in enumerate(x_train) if(np.array_equal(x, y))]).T
        test_Z = Z.copy()
        test_intersection = intersection.copy()
        elbo_base = gp.elbo(training_inputs)
        elbo_curr = gp.elbo(training_inputs)
        elbos = []
        new_q_mu = gp.q_mu
        new_q_sqrt = gp.q_sqrt
        dis = []
        while (np.abs(elbo_base - elbo_curr)/np.abs(elbo_base) <= 1 - threshold or elbo_curr >= elbo_base) and len(test_Z) > 1:
            gp.q_mu = new_q_mu
            gp.q_sqrt = new_q_sqrt
            elbos.append(elbo_curr)
            Z = test_Z.copy()
            gp.inducing_variable = gpflow.inducing_variables.InducingPoints(Z)

            intersection = test_intersection.copy()
            M = len(Z)
            remaining_points = np.array(range(len(intersection)))
            indices = np.zeros(M, dtype=int) + N
            di = kernel(x_train, None, full_cov=False).numpy() + 1e-12  # jitter
            if self.sample:
                indices[0] = sample_discrete(di[intersection][remaining_points])
                remaining_points = np.delete(remaining_points,(indices[0]), 0)
            else:
                indices[0] = np.argmax(di[intersection][remaining_points])
                remaining_points = np.delete(remaining_points,(indices[0]), 0)
            ci = np.zeros((M - 2, N))  # [M,N]
            for m in range(M - 2):
                j = int(intersection[indices[m]])  # int
                new_Z = x_train[j:j + 1]  # [1,D]
                dj = np.sqrt(di[j])  # float
                cj = ci[:m, j]  # [m, 1]
                Lraw = np.array(kernel(x_train, new_Z, full_cov=True))
                L = np.round(np.squeeze(Lraw), 20)  # [N]
                L[j] += 1e-12  # jitter
                print(dj)
                # dj += 1e-12
                
                ei = (L - np.dot(cj, ci[:m])) / dj
                ci[m, :] = ei
                try:
                    di -= ei ** 2
                except FloatingPointError:
                    pass
                di = np.clip(di, 0, None)
                if self.sample:
                    indices[m + 1] = sample_discrete(di[intersection][remaining_points])
                    remaining_points = np.delete(remaining_points,(indices[m+1]), 0)
                else:
                    indices[m + 1] = np.argmax(di[intersection][remaining_points])
                    remaining_points = np.delete(remaining_points,(indices[m+1]), 0)
            idx_to_remove = next(iter(remaining_points))
            dis.append(di[intersection][idx_to_remove])
            test_Z = np.delete(Z, idx_to_remove, 0)
            test_intersection = np.delete(intersection, idx_to_remove, 0)
            old_q_mu = gp.q_mu
            old_q_sqrt = gp.q_sqrt

            new_q_mu = gpflow.Parameter(np.delete(old_q_mu, idx_to_remove, 0), trainable=True, name=f"{layer.name}_q_mu" if layer.name else "q_mu")
            new_q_sqrt  = gpflow.Parameter(update_cholesky(old_q_sqrt.numpy().copy(), idx_to_remove), trainable=True, name=f"{layer.name}_q_sqrt" if layer.name else "q_sqrt")

            gp.q_mu = new_q_mu
            gp.q_sqrt = new_q_sqrt
            gp.inducing_variable = gpflow.inducing_variables.InducingPoints(test_Z)
            elbo_curr = gp.elbo(training_inputs)
            
            gp.q_mu = old_q_mu
            gp.q_sqrt = old_q_sqrt
            gp.inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
            
            print(elbo_base, elbo_curr)
            sys.stdout.flush()
        if np.abs(elbo_base - elbo_curr)/np.abs(elbo_base) <= 1 - threshold and len(test_Z) >= 1:
            Z = test_Z.copy()
            intersection = test_intersection.copy()
            elbos.append(elbo_curr)
        gp = gpflow.models.SVGP(kernel=gp.kernel, likelihood=gp.likelihood, inducing_variable=gp.inducing_variable, mean_function=gp.mean_function, q_mu = gp.q_mu, q_sqrt =gp.q_sqrt, whiten=False)
        if return_di:
            return Z.copy(), indices, gp, elbos, dis
        return Z.copy(), indices, gp, elbos

    def remove_points_deep_gp(self, training_inputs, layer_idx, deep_gp, threshold:float, return_di=False):
        i= 0
        assert (threshold <= 1 and threshold >= 0)
        
        kernel = deep_gp.f_layers[layer_idx].kernel
        elbo_base = deep_gp.elbo(training_inputs)
        elbo_curr = deep_gp.elbo(training_inputs)
        x_train = training_inputs[0]
        Z = deep_gp.f_layers[layer_idx].inducing_variable.Z.numpy()
        N = x_train.shape[0]
        intersection = np.asarray([i for y in Z for i,x in enumerate(x_train) if(np.array_equal(x, y))]).T
        test_Z = Z.copy()
        test_intersection = intersection.copy()
        layer = deep_gp.f_layers[layer_idx]
        new_q_mu = deep_gp.f_layers[layer_idx].q_mu
        new_q_sqrt = deep_gp.f_layers[layer_idx].q_sqrt
        elbos = []
        dis = []
        while (np.abs(elbo_base - elbo_curr)/np.abs(elbo_base) <= 1 - threshold or elbo_curr >= elbo_base)  and len(test_Z) > 1:
            print(i)
            sys.stdout.flush()
            deep_gp.f_layers[layer_idx].q_sqrt = new_q_sqrt
            deep_gp.f_layers[layer_idx].q_mu = new_q_mu
            elbos.append(elbo_curr)
            Z = test_Z.copy()
            intersection = test_intersection.copy()
            M = len(Z)
            remaining_points = np.array(range(len(intersection)))
            indices = np.zeros(M, dtype=int) + N
            di = kernel(x_train, None, full_cov=False).numpy() + 1e-12  # jitter
            if self.sample:
                indices[0] = sample_discrete(di[intersection][remaining_points])
                remaining_points = np.delete(remaining_points,(indices[0]), 0)
            else:
                indices[0] = np.argmax(di[intersection][remaining_points])
                remaining_points = np.delete(remaining_points,(indices[0]), 0)
            
            ci = np.zeros((M - 2, N))  # [M,N]
            for m in range(M - 2):
                j = int(intersection[indices[m]])  # int
                new_Z = x_train[j:j + 1]  # [1,D]
                dj = np.sqrt(di[j])  # float
                cj = ci[:m, j]  # [m, 1]
                Lraw = np.array(kernel(x_train, new_Z, full_cov=True))
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
                    indices[m + 1] = sample_discrete(di[intersection][remaining_points])
                    remaining_points = np.delete(remaining_points,(indices[m+1]), 0)
                else:
                    indices[m + 1] = np.argmax(di[intersection][remaining_points])
                    remaining_points = np.delete(remaining_points,(indices[m+1]), 0)
            idx_to_remove = next(iter(remaining_points))

            i+=1 
            dis.append(di[intersection][idx_to_remove])
            test_Z = np.delete(Z, idx_to_remove, 0)
            test_intersection = np.delete(intersection, idx_to_remove, 0)
            old_layer = deep_gp.f_layers[layer_idx]
            old_q_mu = old_layer.q_mu
            old_q_sqrt = old_layer.q_sqrt

            new_q_mu = gpflow.Parameter(np.delete(old_q_mu, idx_to_remove, 0), trainable=True, name=f"{layer.name}_q_mu" if layer.name else "q_mu")
            new_q_sqrt = gpflow.Parameter(update_cholesky(old_q_sqrt.numpy().copy(), idx_to_remove), trainable=True, name=f"{layer.name}_q_sqrt" if layer.name else "q_sqrt")

            deep_gp.f_layers[layer_idx].inducing_variable.Z = test_Z
            deep_gp.f_layers[layer_idx].q_mu = new_q_mu
            deep_gp.f_layers[layer_idx].q_sqrt = new_q_sqrt

            elbo_curr = deep_gp.elbo(training_inputs)
            deep_gp.f_layers[layer_idx].inducing_variable.Z = Z
            deep_gp.f_layers[layer_idx].q_mu = old_q_mu
            deep_gp.f_layers[layer_idx].q_sqrt = old_q_sqrt

        if np.abs(elbo_base - elbo_curr)/np.abs(elbo_base) <= 1 - threshold and len(test_Z) >= 1:
            deep_gp.f_layers[layer_idx].inducing_variable = gpflow.inducing_variables.InducingPoints(test_Z)
            deep_gp.f_layers[layer_idx].q_mu = new_q_mu
            deep_gp.f_layers[layer_idx].q_sqrt = new_q_sqrt
            Z = test_Z.copy()
            intersection = test_intersection.copy()
            elbos.append(elbo_curr)
        if return_di:
            return Z.copy(), indices, deep_gp, elbos, dis
        return Z.copy(), indices, deep_gp, elbos

def update_cholesky(L: np.ndarray, index):
    if index == L.shape[1]-1:
        return np.delete(np.delete(L, index, axis=1), index, axis=2)
    L_33 = L[:, index+1:,index+1:]
    L_32 = L[:, index+1:, index].T
    new_L = np.delete(np.delete(L, index, axis=1), index, axis=2)
    new_L[:,index:, index:] = np.linalg.cholesky(L_33 @ L_33.transpose((0,2,1)) + L_32@L_32.T)
    return new_L