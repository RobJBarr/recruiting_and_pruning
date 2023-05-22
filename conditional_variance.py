from typing import Optional, Callable
import numpy as np
import gpflow
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
            print(di[intersection][indices[0]])
            print(di[intersection[indices[0]]])
            print((di[intersection][indices[0]] == di[intersection[indices[0]]]).all())
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