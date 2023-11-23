from typing import Optional

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import erfinv

# Basemath's Test

# initialise instance by calling basemaths_test(p_A, p_B, alpha, beta)

# you can get the running time (in samples for one variation) with instance_name.req_samples

# the class has only one public method: evaluate_experiment(self, successes_A, successes_B, batch_samples)
#     takes successes of A and B for the last batch and number of samples for one variation(!)
#     returns 0 if result is insignificant, 1 for positively significant, and -1 for negatively significant


class BaseMathsTest:
    last_succ_diff = 0
    samples = 0

    stop = 0

    # sub method to calculate the sample size and the threshold
    @staticmethod
    def _calculate_sample_size(var_0, mean_1, var_1, alpha, beta):

        # base functions
        def D(T):
            if T < 0:
                return 0
            return -np.sqrt(2 * T * var_1) * erfinv(1 - beta)

        def term_1(x, T):
            return D(T) / np.sqrt(2 * np.pi * var_0 * np.power(x, 3))

        def term_2(x, T):
            return np.exp(-np.power(D(T) + mean_1 * x, 2) / (2 * var_0 * x))

        def integrand(x, T):
            return term_1(x, T) * term_2(x, T)

        def integral(T):
            if T <= 0:
                return 0
            return integrate.quad(integrand, 0, T, args=(T))[0]

        def fun(T):
            return -integral(T) - 1 + alpha

        sample_size = optimize.root(fun, x0=100, jac=False).x[0]
        bound = D(sample_size)

        return (int(np.ceil(sample_size)), bound)

    # sub method to calculate the probability that the experiment was
    # significantly positive or negative since the last check
    # Karatzas & Shreve, page 265
    @staticmethod
    def _probability_of_crossing(D_low, mean_1, var_1, T_0, e_0, T_diff, e_diff):

        # negative case
        if D_low + (T_0 + T_diff) * mean_1 >= e_0 + e_diff:
            return 1.0

        a_low = -e_diff + mean_1 * T_diff
        beta_low = -D_low - mean_1 * T_0 + e_0

        prob_neg = np.exp(-2 * beta_low * (beta_low - a_low) / (T_diff * var_1))

        return prob_neg

    def evaluate_experiment(self, val_A, val_B, batch_samples):

        if self.stop != 0:
            return self.stop

        T_0 = self.samples
        e_0 = self.last_succ_diff
        T_diff = batch_samples
        e_diff = val_B - val_A

        last_test = False

        if T_0 + T_diff > self.required_samples:
            T_diff = self.required_samples - T_0
            share = float(T_diff) / batch_samples
            e_diff = share * e_diff
            last_test = True
        prob_neg = self._probability_of_crossing(
            self.bound, self.mean_1, self.var_1, T_0, e_0, T_diff, e_diff
        )

        self.last_succ_diff += e_diff
        self.samples += T_diff

        stop = 0
        if stats.uniform.rvs(random_state=self.seed) < prob_neg:
            stop = -1
        self.seed = int(1_000_000_000 * stats.uniform.rvs(random_state=self.seed))

        if (last_test) & (stop == 0):
            stop = 1

        self.stop = stop

        return stop

    def __init__(
        self,
        mean_A,
        uplift: float,
        alpha: float = 0.05,
        beta: float = 0.2,
        var_A: Optional[float] = None,
        seed: Optional[object] = None,
    ):
        """
        :param mean_A: The (estimated) mean value of the success metric in the control variation
        :param uplift: The minimal expected percentage uplift we expect to see on the B side.
                       For example: An expected 1% uplift should be passed as 0.01.
        :param alpha: The alpha value, or type 1 error, to use for the test. Defaults to 5% (0.05)
        :param beta: The betq value, or type 2 error, to use for the test. Defaults to 20% (0.02)
        :param var_A: The (estimated) variance of the success metric in the control variation
                      # TODO: This explanation below could be clearer, I'm sure.
                      If the samples are Bernoulli distributed, then we can derive the variance
                      from the mean, and this parameter is not necessary. Otherwise, it should
                      be provided.
        :param seed: A seed used for the 'coin flip' compared against the probability
                     that the experiment may have "crossed the line" inbetween evaluations.
                     This should be set to something uniquely tied to the experiment, such as
                     a name or key, so that the experiment results stay consistent if the test
                     is performed multiple times.
        """
        self.mean_A = mean_A
        self.mean_B = mean_A * (1.0 + uplift)
        self.mean_1 = self.mean_B - self.mean_A

        if var_A is not None:
            self.var_A = var_A
            self.var_B = var_A
        else:
            self.var_A = self.mean_A * (1 - self.mean_A)
            self.var_B = self.mean_B * (1 - self.mean_B)

        self.var_0 = 2 * self.var_A
        self.var_1 = self.var_A + self.var_B

        self.alpha = alpha
        self.beta = beta
        self.uplift = uplift

        self.seed = None
        if seed is not None:
            self.seed = int(str(abs(hash(seed)))[:8])

        (self.required_samples, self.bound) = self._calculate_sample_size(
            self.var_0, self.mean_1, self.var_1, self.alpha, self.beta
        )
