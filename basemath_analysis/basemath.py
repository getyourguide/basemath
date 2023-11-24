import hashlib
from typing import Optional, Tuple

import numpy as np
from scipy import integrate, optimize, stats
from scipy.special import erfinv


class AnalysisException(Exception):
    pass


class BaseMathsTest:

    # sub method to calculate the sample size per variation and the intercept
    @staticmethod
    def _calculate_sample_size(
        var_H0: float, mean_H1: float, var_H1: float, alpha: float, beta: float
    ) -> Tuple[int, float]:
        def D(T):
            if T < 0:
                return 0
            return -np.sqrt(2 * T * var_H1) * erfinv(1 - beta)

        def term_1(x, T):
            return D(T) / np.sqrt(2 * np.pi * var_H0 * np.power(x, 3))

        def term_2(x, T):
            return np.exp(-np.power(D(T) + mean_H1 * x, 2) / (2 * var_H0 * x))

        def integrand(x, T):
            return term_1(x, T) * term_2(x, T)

        def integral(T) -> float:
            if T <= 0:
                return 0
            return integrate.quad(integrand, 0, T, args=(T))[0]

        def fun(T) -> float:
            return -integral(T) - 1 + alpha

        sample_size = optimize.root(fun, x0=100, jac=False).x[0]
        sample_size_int = int(np.ceil(sample_size))

        intercept = D(sample_size)

        if np.abs(fun(sample_size)) > 0.000001:
            raise AnalysisException(
                "The numerical solver was not able to find a root for the provided values."
                "This is an internal error that can happen with extreme values that result "
                "in a very low required number of samples."
            )

        return (sample_size_int, intercept)

    # calculates the probability that the experiment has hit the bound between the two check-ins.
    @staticmethod
    def _probability_of_crossing(
        intercept: float,
        mean_H1: float,
        var_H1: float,
        samples_0: int,
        successes_0: float,
        samples_increment: int,
        successes_change: float,
    ):

        if (
            intercept + (samples_0 + samples_increment) * mean_H1
            >= successes_0 + successes_change
        ):
            return 1.0

        term_1 = -successes_change + mean_H1 * samples_increment
        term_2 = -intercept - mean_H1 * samples_0 + successes_0

        crossing_probability = np.exp(
            -2 * term_2 * (term_2 - term_1) / (samples_increment * var_H1)
        )

        return crossing_probability

    def evaluate_experiment(
        self,
        previous_success_delta: float,
        success_change: float,
        previous_samples_number: int,
        samples_increment: int,
    ):
        """
        :param previous_success_delta: Difference between sum of successes of treatment and baseline at the last
            check-in.
        :param success_change: Difference between sum of successes of treatment and baseline in the current batch.
        :param previous_samples_number: Number of samples per variation at the last check-in.
        :param samples_increment: Number of samples per variation in the current batch.
        """
        if samples_increment < 0 or previous_samples_number < 0:
            raise AnalysisException("Number of samples cannot be less than 0")
        if (
            abs(success_change) > samples_increment
            or abs(previous_success_delta) > previous_samples_number
        ):
            raise AnalysisException(
                "Number of successes cannot be greater than number of samples"
            )
        if previous_samples_number > self.required_samples:
            raise AnalysisException(
                "Number of samples from previous check-in is greater than required samples. "
                "A conclusion (1 or -1) should already have been reached!"
            )

        scaled_samples_increment = samples_increment
        scaled_success_change = success_change

        is_last_evaluation = False

        if previous_samples_number + samples_increment > self.required_samples:
            scaled_samples_increment = self.required_samples - previous_samples_number
            samples_share = float(scaled_samples_increment) / samples_increment
            scaled_success_change = samples_share * success_change
            is_last_evaluation = True

        crossing_probability = self._probability_of_crossing(
            self.intercept,
            self.mean_H1,
            self.var_H1,
            previous_samples_number,
            previous_success_delta,
            scaled_samples_increment,
            scaled_success_change,
        )

        state = 0
        if stats.uniform.rvs(random_state=self.seed) < crossing_probability:
            state = -1
        self.seed: Optional[int] = int(
            1_000_000_000 * stats.uniform.rvs(random_state=self.seed)
        )

        if is_last_evaluation & (state == 0):
            state = 1

        return state

    def __init__(
        self,
        mean_A: float,
        mde: float,
        alpha: float,
        beta: float,
        var_A: Optional[float] = None,
        seed: Optional[str] = None,
    ):
        """
        :param mean_A: The (estimated) mean value of the success metric in the control variation.
        :param mde: The minimum detectable (relative) effect (MDE) we expect to see on the B side.
        :param alpha: The alpha value, or type 1 error, to use for the test.
        :param beta: The beta value, or type 2 error, to use for the test.
        :param var_A: The (estimated) variance of the success metric in the control variation
                      If this parameter is not given, we assume the success metric to be binary (0, 1).
                      In this case, the variance can be computed from the mean.
                      Otherwise, we require an explicit estimation of the variance.
        :param seed: A seed used for the 'coin flip' compared against the probability
                     that the experiment may have "crossed the line" in between evaluations.
                     This should be set to something uniquely tied to the experiment, such as
                     a name or key, so that the experiment results stay consistent if the test
                     is performed multiple times.
        """
        for value in (alpha, beta):
            if value <= 0 or value >= 1:
                raise ValueError(
                    f"Received invalid value of {value}. Passed values for alpha and beta should"
                    f"be within (0, 1)"
                )
        if mde <= 0:
            raise ValueError("The minimum detectable effect must be positive!")
        if mean_A <= 0:
            raise ValueError("mean_A must be positive!")
        if var_A is None and mean_A >= 1:
            raise ValueError(
                "When variance is not passed, we assume a binary metric -- in this case, "
                "the provided mean must be between 0 and 1 OR the variance must be provided."
            )
        if var_A is not None and var_A <= 0:
            raise ValueError("Variance must be positive if provided!")
        self.mean_A = mean_A
        self.mean_B = mean_A * (1.0 + mde)
        # This check only applies for the binary case, i.e. where we don't receive the variance
        if self.mean_B > 1 and var_A is None:
            raise AnalysisException(
                "Cannot possibly detect an effect that brings binary target metric over 100%"
            )
        self.mean_H1 = self.mean_B - self.mean_A

        if var_A is not None:
            self.var_A = var_A
            self.var_B = var_A
        else:
            self.var_A = self.mean_A * (1 - self.mean_A)
            self.var_B = self.mean_B * (1 - self.mean_B)

        self.var_H0 = 2 * self.var_A
        self.var_H1 = self.var_A + self.var_B

        self.alpha = alpha
        self.beta = beta

        self.seed = None
        if seed is not None:
            self.seed = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)

        (self.required_samples, self.intercept) = self._calculate_sample_size(
            self.var_H0, self.mean_H1, self.var_H1, self.alpha, self.beta
        )
