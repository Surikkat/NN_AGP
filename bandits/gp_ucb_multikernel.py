import numpy as np
from scipy.linalg import cho_solve, cho_factor

class GP_UCB_KernelFlexible:
    def __init__(self, x_dim, theta_dim, sigma_noise=0.01, beta=2.0,
                 lengthscale_x=1.0, lengthscale_theta=1.0,
                 kernel_type='multiplicative', adaptive_weights=None):
        self.x_dim = x_dim
        self.theta_dim = theta_dim
        self.sigma_noise = sigma_noise
        self.beta = beta
        self.l_x = lengthscale_x
        self.l_theta = lengthscale_theta
        self.kernel_type = kernel_type

        if adaptive_weights is None:
            self.adaptive_weights = {'theta': 0.5, 'x': 0.5}
        else:
            self.adaptive_weights = adaptive_weights

        self.X = []
        self.y = []

    def rbf_kernel(self, X1, X2, lengthscale):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / lengthscale**2)

    def kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        theta1, x1 = X1[:, :self.theta_dim], X1[:, self.theta_dim:]
        theta2, x2 = X2[:, :self.theta_dim], X2[:, self.theta_dim:]

        k_theta = self.rbf_kernel(theta1, theta2, self.l_theta)
        k_x = self.rbf_kernel(x1, x2, self.l_x)

        if self.kernel_type == 'multiplicative':
            return k_theta * k_x
        elif self.kernel_type == 'adaptive':
            w_theta = self.adaptive_weights.get('theta', 0.5)
            w_x = self.adaptive_weights.get('x', 0.5)
            return w_theta * k_theta + w_x * k_x

    def select_action(self, X_pool, theta_t):
        inputs = np.hstack([np.tile(theta_t, (X_pool.shape[0], 1)), X_pool])

        if len(self.X) == 0:
            idx = np.random.randint(X_pool.shape[0])
            return X_pool[idx], None, None

        X_train = np.array(self.X)
        y_train = np.array(self.y)

        K = self.kernel(X_train, X_train) + self.sigma_noise**2 * np.eye(len(self.X))
        K_s = self.kernel(X_train, inputs)
        K_ss = self.kernel(inputs, inputs)

        L = cho_factor(K, lower=True)
        alpha = cho_solve(L, y_train)

        mu = K_s.T @ alpha
        v = cho_solve(L, K_s)
        sigma = np.sqrt(np.maximum(np.diag(K_ss - K_s.T @ v), 1e-10))

        ucb = mu + np.sqrt(self.beta) * sigma
        idx = np.argmax(ucb)
        return X_pool[idx], mu[idx], sigma[idx]

    def update(self, theta_t, x_t, r_t):
        new_input = np.hstack([theta_t, x_t])
        self.X.append(new_input)
        self.y.append(r_t)

