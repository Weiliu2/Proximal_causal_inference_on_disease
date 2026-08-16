import numpy as np
import torch
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

class ATE_estimator():
    """The dual kernel embedded ATE estimator with proximal doubly robust standardization formula.
    EY^1 and EY^0 are estimated firstly. Then ATE is estimated from the difference between the two CERF estimators.

    Parameter
    ---------
    Z: np.ndarray
        The treatment confounding proxy, whose shape is (n,d_z).
    X: np.ndarray
        The covariate, whose shape is (n,d_x).
    W: np.ndarray
        The outcome confounding proxy, whose shape is (n,d_w).
    A: np.ndarray
        The binary treatment, whose shape is (n,1).
    Y: np.ndarray
        The outcome, whose shape is (n,1).
    cut_ratio: float=0.5
        The cut ratio of the original dataset, determining the data sizes for estimating bridge functions and ATE respectively.
    stabilizer: float=1
        The stabilizer maintaining the invertibility of the inner matrix: K+n * stabilizer * I.
    beta: float=1
        The coefficient determining the convergence of the regularized estimator.
    gamma: float=1e-3
        The band width of the Gaussian kernel function.
    lambda_val: float=1e-1
        The regularization coefficient of the final Tikhonov regularized solution.
    estimator: str="PDR"
        The choice of the standardization formula, chosen among "POR", "PIPW" and "PDR".
    """

    def __init__(self, Z: np.ndarray, X: np.ndarray, W: np.ndarray, A: np.ndarray, Y: np.ndarray, cut_ratio: float=0.5,
                 stabilizer: float=1, beta: float=1, gamma: float=1e-3, lambda_val: float=1e-1, estimator: str="PDR") -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # assign the device on which the code runs (cpu or gpu)
        self.dtype = torch.float64 # set the numerical precision
        self.estimator = estimator
        self.cut_ratio = cut_ratio
        self.stabilizer = torch.tensor(stabilizer, dtype=self.dtype, device=self.device) # impose those parameters to the datatype of tensor
        self.beta = torch.tensor(beta, dtype=self.dtype, device=self.device)
        self.lambda_val = torch.tensor(lambda_val, dtype=self.dtype, device=self.device)
        self.gamma = torch.tensor(gamma, dtype=self.dtype, device=self.device)

        # split the dataset into a bridge dataset for estimating bridge functions and a ATE dataset for estimating ATE together with the estimated bridge functions
        Z_bridge, X_bridge, W_bridge, A_bridge, Y_bridge, Z_ATE, X_ATE, W_ATE, A_ATE, Y_ATE = self.split_samples(Z, X, W, A, Y)

        # further split the dataset for estimating bridge functions. One is for EY^1 and the other is for EY^0.
        self.Z1_bridge, self.Z0_bridge, self.W1_bridge, self.W0_bridge, self.X1_bridge, self.X0_bridge, self.Y1_bridge, self.Y0_bridge = self.classify_samples(Z_bridge, X_bridge, W_bridge, A_bridge, Y_bridge)
        self.n1_bridge = self.Z1_bridge.shape[0] # sample size for bridge function estimators of EY^1
        self.n0_bridge = self.Z0_bridge.shape[0] # sample size for bridge function estimators of EY^0

        # Initialize vector of ones and identity matrices for later use
        self.ones1_bridge = torch.ones(self.n1_bridge, 1, dtype=self.dtype, device=self.device) 
        self.ones0_bridge = torch.ones(self.n0_bridge, 1, dtype=self.dtype, device=self.device) 
        self.I1_bridge = torch.eye(self.n1_bridge, dtype=self.dtype, device=self.device)
        self.I0_bridge = torch.eye(self.n0_bridge, dtype=self.dtype, device=self.device)
        
        # compute gram matrices for estimating bridge functions of both EY^1 and EY^0
        self.Gram_wx1_bridge, self.Gram_zx1_bridge, self.Gram_wx0_bridge, self.Gram_zx0_bridge = self.compute_grams_bridge(self.Z1_bridge, self.Z0_bridge, self.W1_bridge, self.W0_bridge, self.X1_bridge, self.X0_bridge)

        # compute the coefficients of the RKHS representatives of the bridge functions
        self.q_Lambda1, self.q_Lambda0 = self.compute_coefficients_q()
        self.h_Lambda1, self.h_Lambda0 = self.compute_coefficients_h()

        # further split the dataset for estimating EY^1 and EY^0
        self.Z1_ATE, self.Z0_ATE, self.W1_ATE, self.W0_ATE, self.X1_ATE, self.X0_ATE, self.Y1_ATE, self.Y0_ATE = self.classify_samples(Z_ATE, X_ATE, W_ATE, A_ATE, Y_ATE)
        self.n1_ATE = self.Z1_ATE.shape[0]
        self.n0_ATE = self.Z0_ATE.shape[0]
        self.n_ATE = self.n0_ATE + self.n1_ATE
        self.ones1_ATE = torch.ones(self.n1_ATE, 1, dtype=self.dtype, device=self.device)
        self.ones0_ATE = torch.ones(self.n0_ATE, 1, dtype=self.dtype, device=self.device)
        self.I1_ATE = torch.eye(self.n1_ATE, dtype=self.dtype, device=self.device)
        self.I0_ATE = torch.eye(self.n0_ATE, dtype=self.dtype, device=self.device)

        self.Gram_wx1_ATE, self.Gram_zx1_ATE, self.Gram_wx0_ATE, self.Gram_zx0_ATE = self.compute_grams_ATE(self.Z1_bridge, self.Z0_bridge, self.W1_bridge, self.W0_bridge, self.X1_bridge, self.X0_bridge, self.Z1_ATE, self.Z0_ATE, self.W1_ATE, self.W0_ATE, self.X1_ATE, self.X0_ATE)

        self.ATE = self.compute_ATE()
                
    def split_samples(self, Z: np.ndarray, X: np.ndarray, W: np.ndarray, A: np.ndarray, Y: np.ndarray):
        """Split samples into one dataset for building bridge functions and another dataset for computing ATE."""
        cut_point = int(round(Z.shape[0] * self.cut_ratio))
        Z_bridge, Z_ATE = Z[:cut_point], Z[cut_point:]
        X_bridge, X_ATE = X[:cut_point], X[cut_point:]
        W_bridge, W_ATE = W[:cut_point], W[cut_point:]
        A_bridge, A_ATE = A[:cut_point], A[cut_point:]
        Y_bridge, Y_ATE = Y[:cut_point], Y[cut_point:]
        return Z_bridge, X_bridge, W_bridge, A_bridge, Y_bridge, Z_ATE, X_ATE, W_ATE, A_ATE, Y_ATE
    
    def classify_samples(self, Z, X, W, A, Y):
        Z1 = torch.tensor(Z[A[:,0]==1], dtype=self.dtype, device=self.device)
        Z0 = torch.tensor(Z[A[:,0]==0], dtype=self.dtype, device=self.device)

        W1 = torch.tensor(W[A[:,0]==1], dtype=self.dtype, device=self.device)
        W0 = torch.tensor(W[A[:,0]==0], dtype=self.dtype, device=self.device)

        X1 = torch.tensor(X[A[:,0]==1], dtype=self.dtype, device=self.device)
        X0 = torch.tensor(X[A[:,0]==0], dtype=self.dtype, device=self.device)

        Y1 = torch.tensor(Y[A[:,0]==1], dtype=self.dtype, device=self.device)
        Y0 = torch.tensor(Y[A[:,0]==0], dtype=self.dtype, device=self.device)
        return Z1, Z0, W1, W0, X1, X0, Y1, Y0

    def gaussian_RBF(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """v1: (n1,d), v2: (n2,d). Return the Gram matrix of the two tensors using Gaussian RBF kernel."""
        diff = v1.unsqueeze(1) - v2.unsqueeze(0) # broadcast -> (n1, n2, d)
        Gram = torch.exp(-torch.sum(diff**2, dim=-1) * self.gamma)
        return Gram

    def compute_grams_bridge(self, Z1, Z0, W1, W0, X1, X0):
        wx1 = torch.cat((W1, X1), dim=1)
        zx1 = torch.cat((Z1, X1), dim=1)
        wx0 = torch.cat((W0, X0), dim=1)
        zx0 = torch.cat((Z0, X0), dim=1)
        Gram_wx1 = self.gaussian_RBF(wx1, wx1)
        Gram_zx1 = self.gaussian_RBF(zx1, zx1)
        Gram_wx0 = self.gaussian_RBF(wx0, wx0)
        Gram_zx0 = self.gaussian_RBF(zx0, zx0)
        return Gram_wx1, Gram_zx1, Gram_wx0, Gram_zx0
    
    def compute_grams_ATE(self, Z1_bridge, Z0_bridge, W1_bridge, W0_bridge, X1_bridge, X0_bridge, Z1_ATE, Z0_ATE, W1_ATE, W0_ATE, X1_ATE, X0_ATE):
        wx1_bridge = torch.cat((W1_bridge, X1_bridge), dim=1)
        wx1_ATE = torch.cat((W1_ATE, X1_ATE), dim=1)
        zx1_bridge = torch.cat((Z1_bridge, X1_bridge), dim=1)
        zx1_ATE = torch.cat((Z1_ATE, X1_ATE), dim=1)
        wx0_bridge = torch.cat((W0_bridge, X0_bridge), dim=1)
        wx0_ATE = torch.cat((W0_ATE, X0_ATE), dim=1)
        zx0_bridge = torch.cat((Z0_bridge, X0_bridge), dim=1)
        zx0_ATE = torch.cat((Z0_ATE, X0_ATE), dim=1)
        Gram_wx1 = self.gaussian_RBF(wx1_bridge, wx1_ATE)
        Gram_zx1 = self.gaussian_RBF(zx1_bridge, zx1_ATE)
        Gram_wx0 = self.gaussian_RBF(wx0_bridge, wx0_ATE)
        Gram_zx0 = self.gaussian_RBF(zx0_bridge, zx0_ATE)
        return Gram_wx1, Gram_zx1, Gram_wx0, Gram_zx0

    def compute_coefficients_q(self) -> tuple[torch.Tensor, torch.Tensor]:
        # (L(K+n*stabilizer*I)^{-1}KL+n**(1-beta)*K)^{-1}L(K+n*stabilizer*I)^{-1}K@1, K:(w,x), L:(Z,A,X)
        inv1 = torch.linalg.solve(self.Gram_wx1_bridge + self.n1_bridge * self.stabilizer * self.I1_bridge, self.I1_bridge) # inv=(K+n*stabilizer*I)^{-1}
        b1 = self.Gram_zx1_bridge @ inv1 @ self.Gram_wx1_bridge @ self.ones1_bridge # b=L@inv@K@1
        C1 = self.Gram_zx1_bridge @ inv1 @ self.Gram_wx1_bridge @ self.Gram_zx1_bridge + self.n1_bridge**(1 - self.beta) * self.Gram_zx1_bridge # C=L@inv@K@L+n**(1-beta)*K
        CTClambdainv1 = torch.linalg.solve(C1.transpose(-2,-1) @ C1 + self.lambda_val * self.I1_bridge, self.I1_bridge)
        Lambda1 = CTClambdainv1 @ C1.transpose(-2,-1) @ b1
        # Lambda1 = torch.linalg.pinv(C1) @ b1

        inv0 = torch.linalg.solve(self.Gram_wx0_bridge + self.n0_bridge * self.stabilizer * self.I0_bridge, self.I0_bridge)
        b0 = self.Gram_zx0_bridge @ inv0 @ self.Gram_wx0_bridge @ self.ones0_bridge
        C0 = self.Gram_zx0_bridge @ inv0 @ self.Gram_wx0_bridge @ self.Gram_zx0_bridge + self.n0_bridge**(1 - self.beta) * self.Gram_zx0_bridge
        CTClambdainv0 = torch.linalg.solve(C0.transpose(-2,-1) @ C0 + self.lambda_val * self.I0_bridge, self.I0_bridge)
        Lambda0 = CTClambdainv0 @ C0.transpose(-2,-1) @ b0
        # Lambda0 = torch.linalg.pinv(C0) @ b0
        return Lambda1, Lambda0 # (n,1)

    def compute_coefficients_h(self) -> tuple[torch.Tensor, torch.Tensor]:
        # (L(K+n*stabilizer*I)^{-1}KL+n**(1-beta)*K)^{-1}L(K+n*stabilizer*I)^{-1}K@1_{A=a}*Y, K:(Z,X), L:(W,A,X)
        inv1 = torch.linalg.solve(self.Gram_zx1_bridge + self.n1_bridge * self.stabilizer * self.I1_bridge, self.I1_bridge) # inv=(K+n*stabilizer*I)^{-1}
        b1 = self.Gram_wx1_bridge @ inv1 @ self.Gram_zx1_bridge @ self.Y1_bridge # b=L@inv@K@1_{A=a}*Y
        C1 = self.Gram_wx1_bridge @ inv1 @ self.Gram_zx1_bridge @ self.Gram_wx1_bridge + (self.n1_bridge**(1 - self.beta)) * self.Gram_wx1_bridge # C=L@inv@K@L+n**(1-beta)*K
        CTClambdainv1 = torch.linalg.solve(C1.transpose(-2,-1) @ C1 + self.lambda_val * self.I1_bridge, self.I1_bridge)
        Lambda1 = CTClambdainv1 @ C1.transpose(-2,-1) @ b1
        # Lambda1 = torch.linalg.pinv(C1) @ b1

        inv0 = torch.linalg.solve(self.Gram_zx0_bridge + self.n0_bridge * self.stabilizer * self.I0_bridge, self.I0_bridge)
        b0 = self.Gram_wx0_bridge @ inv0 @ self.Gram_zx0_bridge @ self.Y0_bridge
        C0 = self.Gram_wx0_bridge @ inv0 @ self.Gram_zx0_bridge @ self.Gram_wx0_bridge + (self.n0_bridge**(1 - self.beta)) * self.Gram_wx0_bridge
        CTClambdainv0 = torch.linalg.solve(C0.transpose(-2,-1) @ C0 + self.lambda_val * self.I0_bridge, self.I0_bridge)
        Lambda0 = CTClambdainv0 @ C0.transpose(-2,-1) @ b0
        # Lambda0 = torch.linalg.pinv(C0) @ b0
        return Lambda1, Lambda0 # (n,1)
    
    def compute_dual_q(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv1 = torch.linalg.solve(self.Gram_wx1_bridge + self.n1_bridge * self.stabilizer * self.I1_bridge, self.I1_bridge)
        Gamma1 = inv1 @ (self.Gram_zx1_bridge @ self.q_Lambda1 - self.ones1_bridge)
        u1 = Gamma1.T @ self.Gram_wx1_ATE

        inv0 = torch.linalg.solve(self.Gram_wx0_bridge + self.n0_bridge * self.stabilizer * self.I0_bridge, self.I0_bridge)
        Gamma0 = inv0 @ (self.Gram_zx0_bridge @ self.q_Lambda0 - self.ones0_bridge)
        u0 = Gamma0.T @ self.Gram_wx0_ATE
        return u0, u1

    def compute_dual_h(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv1 = torch.linalg.solve(self.Gram_zx1_bridge + self.n1_bridge * self.stabilizer * self.I1_bridge, self.I1_bridge)
        Gamma1 = inv1 @ (self.Gram_wx1_bridge @ self.h_Lambda1 - self.Y1_bridge)
        u1 = Gamma1.T @ self.Gram_zx1_ATE

        inv0 = torch.linalg.solve(self.Gram_zx0_bridge + self.n0_bridge * self.stabilizer * self.I0_bridge, self.I0_bridge)
        Gamma0 = inv0 @ (self.Gram_wx0_bridge @ self.h_Lambda0 - self.Y0_bridge)
        u0 = Gamma0.T @ self.Gram_zx0_ATE
        return u0, u1
    
    def compute_q_bridge(self) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q_Lambda1.T @ self.Gram_zx1_ATE
        q0 = self.q_Lambda0.T @ self.Gram_zx0_ATE
        return q1, q0 # (1,n)
    
    def compute_h_bridge(self) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.h_Lambda1.T @ self.Gram_wx1_ATE
        h0 = self.h_Lambda0.T @ self.Gram_wx0_ATE
        return h1, h0 # (1,n)
    
    def compute_ATE(self):
        "Compute the estimators of average treatment effect. Three methods are all included: POR/PIPW/PDR."
        q1, q0 = self.compute_q_bridge()
        h1, h0 = self.compute_h_bridge()
        if self.estimator == 'PDR':
            ATE = torch.mean(q1.T*(self.Y1_ATE-h1.T)+h1.T) - torch.mean(q0.T*(self.Y0_ATE-h0.T)+h0.T)
        elif self.estimator == 'PIPW':
            ATE = torch.mean(self.Y1_ATE*q1.T) - torch.mean(self.Y0_ATE*q0.T)
        elif self.estimator == 'POR':
            ATE = torch.mean(h1.T) - torch.mean(h0.T)
        else: 
            ATE = torch.mean(q1.T*(self.Y1_ATE-h1.T)+h1.T) - torch.mean(q0.T*(self.Y0_ATE-h0.T)+h0.T)
        return ATE.item()
    
    def compute_CERF(self):
        "Compute the estimators of causal exposure response function EY^1 and EY^0. Three methods are all included: POR/PIPW/PDR."
        q1, q0 = self.compute_q_bridge()
        h1, h0 = self.compute_h_bridge()
        CERF_1_POR = torch.mean(h1.T)
        CERF_0_POR = torch.mean(h0.T)
        CERF_1_PIPW = torch.mean(self.Y1_ATE*q1.T)
        CERF_0_PIPW = torch.mean(self.Y0_ATE*q0.T)
        CERF_1_PDR = torch.mean(q1.T*(self.Y1_ATE-h1.T)+h1.T)
        CERF_0_PDR = torch.mean(q0.T*(self.Y0_ATE-h0.T)+h0.T)
        return CERF_1_POR.item(), CERF_0_POR.item(), CERF_1_PIPW.item(), CERF_0_PIPW.item(), CERF_1_PDR.item(), CERF_0_PDR.item()

    def compute_p_value(self, alpha=0.05, CI=False, verbose=False):
        """Compute p-value of test H_0: ATE=0 <-> H_1: ATE neq 0 using PDR estimator.
        
        Parameters
        ----------
        alpha: float=0.05
            The significance level for the hypothesis test.
        CI: bool=False
            Whether to compute confidence interval.
        verbose: bool=False
            Whether to show the detail of the Wald ratio statistic including the sample variances and the value of the Wald ratio statistic.

        Return
        -------
        If CI is True
            Return a tuple (confidence_lower_bound, confidence_upper_bound, p_value)
        If CI is False
            Return only the p value
        
        Warning
        -------
        The p value is usually 0 due to the restricted precision of numpy.
        It is suggested to set verbose to True and record the value of the Wald ratio statistic and use R to compute the probability Pr(|Z|>Z_stat).
        """
        q1, q0 = self.compute_q_bridge()
        h1, h0 = self.compute_h_bridge()
        EY_1 = q1.T*(self.Y1_ATE-h1.T)+h1.T
        EY_0 = q0.T*(self.Y0_ATE-h0.T)+h0.T
        # get sample variances
        sigma_1 = EY_1.var()
        sigma_0 = EY_0.var()
        # compute statistic z
        S_sqrt = torch.sqrt(sigma_1/self.n1_ATE+sigma_0/self.n0_ATE)
        Z_stat = self.ATE / S_sqrt
        if verbose == True:
            print(f"sigma_1={sigma_1.item()}, sigma_0={sigma_0.item()}")
            print(f"S={S_sqrt.item()}, Z={Z_stat.item()}")
        # compute p value
        normal_dist = Normal(0, 1)
        p_value = 2 * (1 - normal_dist.cdf(torch.abs(Z_stat)))
        if CI == True:
            # compute confidence interval with significance level 0.05
            Z_alpha = normal_dist.icdf(torch.tensor(1-alpha/2))
            clb = self.ATE - S_sqrt * Z_alpha
            cub = self.ATE + S_sqrt * Z_alpha
            return clb.item(), cub.item(), p_value.item()
        else: 
            return p_value.item()


def hyperparameter_tuner(Z, X, W, A, Y, cut_ratio=0.5, shuffles=3, lambda_val=1, stabilizer_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], 
                         beta_range = [1e-2, 1e-1, 2e-1, 5e-1, 1],
                         verbose = False):
    score_best = torch.inf
    best_stabilizer = 0
    best_beta = 0
    shuffled_indices = []
    n_obs = Z.shape[0]
    for _ in range(shuffles):
        # shuffle the original dataset by "shuffles" times to create bootstrap dataset
        shuffled_indices.append(np.random.permutation(n_obs))
    for stabilizer in stabilizer_range:
        for beta in beta_range:
            score_temp = 0
            for i in range(shuffles):
                if verbose == True:
                    print(f"Testing stabilizer: {stabilizer}, beta: {beta} at {i+1}-th shuffle.")
                Z_ = Z[shuffled_indices[i]]
                X_ = X[shuffled_indices[i]]
                W_ = W[shuffled_indices[i]]
                A_ = A[shuffled_indices[i]]
                Y_ = Y[shuffled_indices[i]]
                model_ = ATE_estimator(Z_, X_, W_, A_, Y_, cut_ratio, stabilizer, beta, lambda_val=lambda_val)
                u0_q_, u1_q_ = model_.compute_dual_q()
                u0_h_, u1_h_ = model_.compute_dual_h()
                score_temp += torch.mean(u0_q_**2) + torch.mean(u1_q_**2) + torch.mean(u0_h_**2) + torch.mean(u1_h_**2)    
            if (score_temp/shuffles) <= score_best:
                score_best = score_temp
                best_stabilizer = stabilizer
                best_beta = beta
    return score_best, best_stabilizer, best_beta

def bootstrap_ATE(Z, X, W, A, Y, n_bootstrap=1000, alpha=0.05, **kwargs):
    """
    Use Bootstrap method to calculate confidence interval of ATE.
    
    Parameters
    ----------
    Z, X, W, A, Y : np.ndarray
        Original data.
    n_bootstrap : int
        Times of Bootstrap sampling.
    alpha : float
        Significance level: 1-alpha.
    **kwargs : dict
        Hyperparameters: stabilizer, beta, lambda_val, gamma, cut_ratio, estimator.
    
    Return
    -------
    "P" : float
        P-value of centered bootstrap percentile test
    "clb-alpha", "cub-alpha" : float
        Lower bound and upper bound of confidence interval.
    "ate_boot" : list
        ATE estimations for B Bootstrap attempts.
    "ate_obs" : ATE estimation of original dataset
    """
    n = Z.shape[0]
    ate_boot = []
    
    # # extract hyperparameters (use default if not set)
    # cut_ratio = kwargs.get('cut_ratio', 0.5)
    # stabilizer = kwargs.get('stabilizer', 1e-1)
    # beta = kwargs.get('beta', 0.125)
    # lambda_val = kwargs.get('lambda_val', 1e-3)
    # gamma = kwargs.get('gamma', 1e-3)
    # estimator = kwargs.get('estimator', 'PDR')

    # Estimate ATE of original dataset
    model_obs = ATE_estimator(Z, X, W, A, Y, **kwargs)
    ate_obs = model_obs.ATE
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Indices of resampling with returns
        idx = np.random.choice(n, n, replace=True)
        Z_boot, X_boot, W_boot, A_boot, Y_boot = Z[idx], X[idx], W[idx], A[idx], Y[idx]

        # Estimate ATE of bootstrap dataset
        model = ATE_estimator(Z_boot, X_boot, W_boot, A_boot, Y_boot, **kwargs)
        ate = model.ATE
        ate_boot.append(ate)

    ate_boot = np.array(ate_boot)
    
    # centered Bootstrap percentile test, H_0:ATE estimation=0 <-> H_1:ATE estimation neq 0
    centered_ate_boot = ate_boot - np.mean(ate_boot)
    extreme_count_1 = np.sum(centered_ate_boot >= np.abs(ate_obs))
    p_value = (1+extreme_count_1) / (1+n_bootstrap)

    # confidence interval with significance level: alpha
    lower = np.percentile(ate_boot, 100 * alpha / 2)
    upper = np.percentile(ate_boot, 100 * (1 - alpha / 2))
    
    return {"P": p_value, 
            "clb-alpha": lower, 
            "cub-alpha": upper, 
            "ate_boot": ate_boot, 
            "ate_obs": ate_obs}

def plot_bootstrap_ate(ate_boot, bins=30, title='Bootstrap ATE Distribution', save_to_file='svg', show_fig=True):
    """
    Plot histogram of Bootstrap ATE estimations.
    
    Parameters
    ----------
    ate_boot : array-like
        Array of Bootstrap ATE estimations.
    bins : int
        Number of pillars in the histogram.
    title : str
        Main title of the figure.
    """
    ate_boot = np.asarray(ate_boot)
    
    # descriptive statistics
    mean_ate = np.mean(ate_boot)
    median_ate = np.median(ate_boot)
    std_ate = np.std(ate_boot, ddof=1)
    skewness = skew(ate_boot)
    kurt = kurtosis(ate_boot)  # Extended Kurtosis (0 for normal distributions)
    
    # figure style
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    
    # histogram (uniformed density)
    plt.hist(ate_boot, bins=bins, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Bootstrap ATE')
    
    # kernel density curve
    sns.kdeplot(ate_boot, color='red', linewidth=2, label='Kernel Density')
    
    # locate mean and median
    plt.axvline(mean_ate, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean = {mean_ate:.4f}')
    plt.axvline(median_ate, color='purple', linestyle=':', linewidth=2, label=f'Median = {median_ate:.4f}')
    
    # add texts for descriptive statistics
    textstr = f'Std Dev = {std_ate:.4f}\nSkewness = {skewness:.4f}\nExcess Kurtosis = {kurt:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('ATE')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.{save_to_file}", dpi=300)
    if show_fig == True:
        plt.show()


"""---------------------------------------Example usage---------------------------------------"""
if __name__ == "__main__":
    def data_generation(n=10000):

        U = np.random.normal(0, 1, (n, 10))
        X = 0.05 * U.sum(axis=1, keepdims=True) + np.random.normal(0, 1, (n, 5))
        Z = 0.03 * U.sum(axis=1, keepdims=True) + 0.04 * X.sum(axis=1, keepdims=True) + np.random.normal(0, 1, (n, 6))
        W = 0.06 * U.sum(axis=1, keepdims=True) + 0.03 * X.sum(axis=1, keepdims=True) + np.random.normal(0, 1, (n, 3))

        A_logits = 0.03 * U.sum(axis=1, keepdims=True) + 0.05 * X.sum(axis=1, keepdims=True) + 0.07 * Z.sum(axis=1, keepdims=True) + np.random.normal(0, 1, (n, 1))
        A = (A_logits > 0).astype(int)
        Y1 = 2 + 0.07 * U.sum(axis=1, keepdims=True) + 0.02 * X.sum(axis=1, keepdims=True) + 0.08 * W.sum(axis=1, keepdims=True) + np.random.normal(0, 1, (n, 1))
        Y0 = -1 + 0.03 * U.sum(axis=1, keepdims=True) + 0.04 * X.sum(axis=1, keepdims=True) + 0.03 * W.sum(axis=1, keepdims=True) + np.random.normal(0, 1, (n, 1))
        Y = Y1 * A + Y0 * (1 - A)
        return Z, X, W, A, Y

    def p_value_test_distribution(n=1000, stabilizer=1, beta=1, lambda_val=1e-1, gamma=1e-3):
        Z, X, W, A, Y= data_generation(n)
        model = ATE_estimator(Z, X, W, A, Y, stabilizer=stabilizer, beta=beta, lambda_val=lambda_val, gamma=gamma, estimator='PDR')
        ATE = model.ATE
        CERF_1_POR, CERF_0_POR, CERF_1_PIPW, CERF_0_PIPW, CERF_1_PDR, CERF_0_PDR = model.compute_CERF()
        clb, cub, p_value = model.compute_p_value(CI=True, verbose=True)
        print(f"EY1, EY0 estimated as: {CERF_1_PDR}, {CERF_0_PDR}")
        print(f"ATE estimated as: {ATE}")
        print(f"P value is {p_value}")
        print(f"Confidence interval with alpha=0.05: [{clb}, {cub}]")

    def p_value_test_bootstrap(save=False, n=10000, n_bootstrap=2, **kwarg):
        Z, X, W, A, Y = data_generation(n)
        results = bootstrap_ATE(Z, X, W, A, Y, n_bootstrap=n_bootstrap, **kwarg)
        print(f"P-value for original hypothesis \"ATE=0\" is {results['P']:.2e}.")
        print(f"Confidence interval with significance level 0.05 is [{results['clb-alpha']:.2f},{results['cub-alpha']:.2f}].")
        print(f"Calculated ATE estimator is {results['ate_obs']:.2f}.")
        if save == True:
            np.save("ate_boot.npy", results['ate_boot'])
        plot_bootstrap_ate(results['ate_boot'])

    # """Start testing"""
    # p_value_test_distribution()
    # p_value_test_bootstrap()

    import pandas as pd
    # X: grp
    # grp = pd.read_csv("grp.csv")
    # # print(grp.columns) # ['HBP_Status_p131286', 'BMI', 'HF_Status_p131354', 'grp', 'crp'] # A, Z, Y, X, W
    # dt_grp = grp.to_numpy()
    # A = dt_grp[:,0].reshape(-1,1)
    # Z = dt_grp[:,1].reshape(-1,1)
    # Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    # Y = dt_grp[:,2].reshape(-1,1)
    # X = dt_grp[:,3].reshape(-1,1)
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    # W = dt_grp[:,4].reshape(-1,1)
    # W = (W - W.mean(axis=0)) / W.std(axis=0)
    # print(f"sample size:{A.shape[0]}")
    # model = ATE_estimator(Z=Z, X=X, W=W, A=A, Y=Y, stabilizer=50, beta=100, lambda_val=10)
    # CERF_1_POR, CERF_0_POR, CERF_1_PIPW, CERF_0_PIPW, CERF_1_PDR, CERF_0_PDR = model.compute_CERF()
    # ATE = model.ATE
    # clb, cub, p = model.compute_p_value(CI=True, verbose=True)
    # print(f"EY1:{CERF_1_PDR}, EY0:{CERF_0_PDR}, ATE:{ATE}, CI:[{clb}, {cub}], p-value:{p}")
    # # score_best_grp, best_stabilizer_grp, best_beta_grp = hyperparameter_tuner(Z=Z, X=X, W=W, A=A, Y=Y, shuffles=5, stabilizer_range=[1, 10, 20, 50], beta_range=[1, 10, 50, 75, 100], verbose=True)
    # # result_grp = bootstrap_ATE(Z=Z, X=X, W=W, A=A, Y=Y, stabilizer=best_stabilizer_grp, beta=best_beta_grp)
    # # plot_bootstrap_ate(ate_boot=result_grp["ate_boot"], title="Bootstrap result (grp)", show_fig=False)

    # X: dars1
    # dars1 = pd.read_csv("dars1.csv")
    # dt_dars1 = dars1.to_numpy()
    # # print(dars1.columns) # ['HBP_Status_p131286', 'BMI', 'HF_Status_p131354', 'dars1', 'crp'] # A, Z, Y, X, W
    # A = dt_dars1[:,0].reshape(-1,1)
    # Z = dt_dars1[:,1].reshape(-1,1)
    # Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    # Y = dt_dars1[:,2].reshape(-1,1)
    # X = dt_dars1[:,3].reshape(-1,1)
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    # W = dt_dars1[:,4].reshape(-1,1)
    # W = (W - W.mean(axis=0)) / W.std(axis=0)
    # print(f"sample size:{A.shape[0]}")
    # # score_best_dars1, best_stabilizer_dars1, best_beta_dars1= hyperparameter_tuner(Z=Z, X=X, W=W, A=A, Y=Y, shuffles=5, stabilizer_range=[1e-1, 1, 10, 20, 50], beta_range=[1, 10, 50, 75, 100], verbose=True)
    # # result_dars1 = bootstrap_ATE(Z=Z, X=X, W=W, A=A, Y=Y, stabilizer=best_stabilizer_dars1, beta=best_beta_dars1)
    # # plot_bootstrap_ate(ate_boot=result_dars1["ate_boot"], title="Bootstrap result (dars1)", show_fig=False)
    # model = ATE_estimator(Z=Z, X=X, W=W, A=A, Y=Y, stabilizer=50, beta=100, lambda_val=10)
    # CERF_1_POR, CERF_0_POR, CERF_1_PIPW, CERF_0_PIPW, CERF_1_PDR, CERF_0_PDR = model.compute_CERF()
    # ATE = model.ATE
    # clb, cub, p = model.compute_p_value(CI=True, verbose=True)
    # print(f"EY1:{CERF_1_PDR}, EY0:{CERF_0_PDR}, ATE:{ATE}, CI:[{clb}, {cub}], p-value:{p}")

    # ASCVD
    dt = pd.read_csv("raw_UKB_data_multiple_diseases/ASCVD.csv")
    colnames = dt.columns.tolist()
    # print(colnames)
    Z = dt[colnames[0]].to_numpy().reshape(-1,1)
    W = dt[colnames[1]].to_numpy().reshape(-1,1)
    A = dt[colnames[2]].to_numpy().reshape(-1,1)
    Y = dt[colnames[3]].to_numpy().reshape(-1,1)
    X = dt[colnames[4:]].to_numpy()
    model = ATE_estimator(Z=Z, X=X, W=W, A=A, Y=Y, stabilizer=50, beta=100, lambda_val=10)
    CERF_1_POR, CERF_0_POR, CERF_1_PIPW, CERF_0_PIPW, CERF_1_PDR, CERF_0_PDR = model.compute_CERF()
    ATE = model.ATE
    clb, cub, p = model.compute_p_value(CI=True, verbose=True)
    print(f"EY1:{CERF_1_PDR}, EY0:{CERF_0_PDR}, ATE:{ATE}, CI:[{clb}, {cub}], p-value:{p}")