import numpy as np
import sys

sys.path.append('')
np.set_printoptions(threshold=np.inf)

class FSVD:
    def __init__(self, num_parties=2, random_seed=None):
        self.num_parties = num_parties
        self.seed = np.random.RandomState(random_seed)

    def load_data(self, X_shared: np.ndarray, Xs):
        self.X_shared = X_shared
        self.Xs = Xs
        # check each partition
        # print(self.Xs)

    def learning(self):
        m, n = self.X_shared.shape
        # Generate a random orthogonal matrix P
        P, _ = np.linalg.qr(np.random.randn(m, m))
        # print(P.shape)

        X_mask_partitions = []
        for i in range(self.num_parties):
            X_mask_partitions.append(P @ self.Xs[i])

        # Check the shape of each partition
        # for arr in X_mask_partitions:
        #     print(arr.shape)

        X_mask = np.concatenate(X_mask_partitions, axis=1)

        # Perform SVD on the masked data. To speed up, we only use the first n columns of U_mask if m >= n
        U_mask, Sigma, VT_mask = np.linalg.svd(X_mask, full_matrices=False)

        # # make sure Sigma = S
        # U, S, V = np.linalg.svd(self.X_shared, full_matrices=False)
        # print('Sigma', Sigma)
        # print('S', S)

        # Convert the 1D array Sigma into a diagonal matrix
        Sigma = np.diag(Sigma)

        # Compute the embedding
        E_U = P.T @ U_mask
        E_US = P.T @ U_mask @ Sigma

        E_U = np.array(E_U)
        E_US = np.array(E_US)
        self.Emb_U = E_U
        self.Emb_US = E_US

    def extract_embedding(self):
        return np.array(self.Emb_U), np.array(self.Emb_US)
