from unittest.mock import patch

import pytest

from basemath_analysis.basemath import AnalysisException, BaseMathsTest


def test_experiment_success():
    """
    Ensure that with typical usage and a successful experiment, we declare the
     experiment a success after collecting enough samples (and the experiment 'surviving'
     until that point)
    """
    basemath = BaseMathsTest(0.9, 0.01, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 15006
    assert basemath.evaluate_experiment(0, 10, 0, 1000) == 0
    assert basemath.evaluate_experiment(10, 30, 1000, 2000) == 0
    assert basemath.evaluate_experiment(40, 9000, 3000, 12500) == 1


def test_experiment_failure_samples_reached():
    """
    Ensure that with typical usage and an unsuccessful experiment, we declare
     the experiment non-positive (in a situation where we've collected enough samples)
    """
    basemath = BaseMathsTest(0.6, 0.5, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 30
    assert basemath.evaluate_experiment(0, 1, 0, 10) == 0
    assert basemath.evaluate_experiment(1, 2, 10, 12) == 0
    assert basemath.evaluate_experiment(3, 4, 22, 15) == -1


def test_experiment_failure_before_samples_reached():
    """
    Ensure that we can also declare the experiment non-positive before
    collecting enough samples
    """
    basemath = BaseMathsTest(0.6, 0.5, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 30
    assert basemath.evaluate_experiment(0, 1, 0, 10) == 0
    assert basemath.evaluate_experiment(1, -4, 10, 12) == -1


def test_experiment_meets_samples_exactly():
    """
    When we meet the required number of samples...we actually require a little
    more. That is, we require the required number of samples to be *exceeded*.
    Let's make sure this behaves as expected
    """
    basemath = BaseMathsTest(0.6, 0.5, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 30
    assert basemath.evaluate_experiment(0, 10, 0, 10) == 0
    assert basemath.evaluate_experiment(10, 12, 10, 12) == 0
    assert basemath.evaluate_experiment(22, 8, 22, 8) == 0
    assert basemath.evaluate_experiment(30, 1, 30, 1) == 1


def test_uplift_impossible():
    """
    For binary experiments, the uplift can't be such that it brings the target metric over 100%.
    We should proactively check for this case, as we don't error otherwise.
    """
    with pytest.raises(AnalysisException) as exception_context_manager:
        # 50% uplift on 0.9 would take us to 1.35, or 135%, which we cannot possibly reach
        BaseMathsTest(0.9, 0.5, 0.05, 0.2, seed="test-experiment")
    expected_exception_text = (
        "Cannot possibly detect an effect that brings binary target metric over 100%"
    )
    assert str(exception_context_manager.value) == expected_exception_text


def test_uplift_over_one_non_binary_case():
    """
    In the non-binary case, it's completely fine if the target metric goes over 100%
    """
    # We *don't* raise an exception here, so the test passes
    BaseMathsTest(0.9, 0.5, 0.05, 0.2, seed="test-experiment", var_A=0.5)


def test_evaluate_experiment_called_after_experiment_over():
    """
    If we have "total as of previous call" values that exceed the required samples,
    then our user is doing something wrong (continuing to evaluate an experiment
    after it has 'ended') AND our logic breaks (we can't coherently handle evaluating
    an interval that lies entirely beyond our required samples)

    So, we should raise an exception in this case
    """
    basemath = BaseMathsTest(0.6, 0.5, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 30
    assert basemath.evaluate_experiment(0, 30, 0, 50) == 1
    with pytest.raises(AnalysisException) as exception_context_manager:
        basemath.evaluate_experiment(30, 20, 50, 30)
    expected_exception_text = (
        "Number of samples from previous check-in is greater than required samples. "
        "A conclusion (1 or -1) should already have been reached!"
    )
    assert str(exception_context_manager.value) == expected_exception_text


@pytest.mark.parametrize("mde", [-0.5, 0])
def test_non_positive_mde(mde):
    """
    Negative (or zero) uplift is not handled by our approach -- the intercept will always
    be negative, and so a "successful" experiment in this regard would fail, which
    could be misleading. Not sure if anyone would actually *want* to run an experiment
    like this, but good to validate for it just in case.
    """
    with pytest.raises(ValueError) as exception_context_manager:
        BaseMathsTest(0.6, mde, 0.05, 0.2, seed="test-experiment")
    expected_exception_text = "The minimum detectable effect must be positive!"
    assert str(exception_context_manager.value) == expected_exception_text


def test_mde_greater_than_one():
    """
    Most of our parameters must be within (0, 1), but that's not true for mde.
    It could easily be greater than or equal to 1, like 2 for a 200% uplift. That's
    not a very realistic experiment, but it should still work with our approach.
    """
    basemath = BaseMathsTest(0.01, 3, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 261
    assert basemath.evaluate_experiment(0, 120, 0, 150) == 0
    assert basemath.evaluate_experiment(120, 50, 150, 80) == 0
    assert basemath.evaluate_experiment(170, 26, 230, 40) == 1


def test_first_call_concludes_experiment_failure():
    """
    It's possible that the very first call reaches enough samples to conclude the
    experiment -- we should validate that we can reach either success or failure
    in this special case.
    """
    basemath = BaseMathsTest(0.1, 0.01, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 1247728
    assert basemath.evaluate_experiment(0, -50, 0, 1500000) == -1


def test_first_call_concludes_experiment_success():
    """
    It's possible that the very first call reaches enough samples to conclude the
    experiment -- we should validate that we can reach either success or failure
    in this special case.
    """
    basemath = BaseMathsTest(0.1, 0.01, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 1247728
    assert basemath.evaluate_experiment(0, 10000, 0, 1500000) == 1


def test_providing_variance():
    """
    We optionally accept a parameter for variance on the control side in case of
    a nonbinary success metric. We should validate that we run without issue
    when this is provided.
    """
    basemath = BaseMathsTest(0.6, 0.5, 0.05, 0.2, seed="test-experiment", var_A=2)
    assert basemath.required_samples == 308
    assert basemath.evaluate_experiment(0, 68, 0, 150) == 0
    assert basemath.evaluate_experiment(68, 83, 150, 200) == 1


def test_seed_not_provided():
    """
    We provide a seed in most of our tests to guarantee consistency of the results,
    but we could also not provide one. We should run without issue in this case.
    """
    basemath = BaseMathsTest(0.3, 0.9, 0.05, 0.2)
    assert basemath.required_samples == 42
    # First evaluation to show that the experiment starts inconclusive. Using a number of samples that's greater than
    #  zero but less than the total requirement risks losing the 'coin flip' to determine whether we crossed the line
    #  (since we don't control the seed in this test), so we set it to zero to keep the test deterministic
    assert basemath.evaluate_experiment(0, 0, 0, 0) == 0
    # And now we evaluate with >=required_samples, again deterministic
    assert basemath.evaluate_experiment(0, -10, 0, 50) == -1


def test_all_zeros_in_experiment_evaluation():
    """
    We don't really expect to be called without *any* samples, but in the case
    that we are called like this, we should return 0 as you'd expect
    """
    basemath = BaseMathsTest(0.3, 0.9, 0.05, 0.2, seed="test-experiment")
    assert basemath.evaluate_experiment(0, 0, 0, 0) == 0


@pytest.mark.parametrize("samples", [100, -100])
def test_number_of_successes_exceeds_samples(samples):
    """
    If we have more successes than actual samples, that's not actually coherent
    input, but we currently accept it. Should we add validation for this case?
    """
    basemath = BaseMathsTest(0.3, 0.9, 0.05, 0.2, seed="test-experiment")
    # Negative change in samples
    with pytest.raises(AnalysisException) as exception_context_manager_current:
        basemath.evaluate_experiment(0, samples, 0, 10)
    # Negative total of samples (as of previous check-in)
    with pytest.raises(AnalysisException) as exception_context_manager_previous:
        basemath.evaluate_experiment(samples, 0, 10, 0)
    expected_exception = "Number of successes cannot be greater than number of samples"
    assert str(exception_context_manager_current.value) == expected_exception
    assert str(exception_context_manager_previous.value) == expected_exception


def test_negative_number_of_samples():
    """
    A negative number of samples is also not realistic, but we accept
    it anyway. Another question of whether we should validate for this.

    Currently, a negative value sets the probability of crossing to a value
    greater than 1, meaning the call always results in experiment failure (-1)
    """
    basemath = BaseMathsTest(0.3, 0.9, 0.05, 0.2, seed="test-experiment")
    # Negative change in samples
    with pytest.raises(AnalysisException) as exception_context_manager_current:
        basemath.evaluate_experiment(0, -1, 0, -1)
    # Negative total of samples (as of previous check-in)
    with pytest.raises(AnalysisException) as exception_context_manager_previous:
        basemath.evaluate_experiment(0, 5, -1, 10)
    expected_exception = "Number of samples cannot be less than 0"
    assert str(exception_context_manager_current.value) == expected_exception
    assert str(exception_context_manager_previous.value) == expected_exception


def test_guarantee_crossing_bound():
    """
    If the 'line' is below the bound, then we should always fail the experiment,
    no matter how high we roll on the coin toss
    """
    basemath = BaseMathsTest(0.3, 0.9, 0.05, 0.2, seed="test-experiment")
    assert basemath.required_samples == 42
    assert basemath.evaluate_experiment(0, -1, 0, 10) == 0
    with patch("basemath_analysis.basemath.stats.uniform.rvs") as mock_rvs:
        mock_rvs.return_value = 0.999999999
        assert basemath.evaluate_experiment(-1, -10, 10, 40) == -1


def test_atypical_alpha_beta_values():
    """
    The alpha and beta values are consistent across most of our other tests -- here,
    we confirm that other values also work without issue
    """
    basemath = BaseMathsTest(0.3, 0.9, 0.01, 0.31, seed="test-experiment")
    assert basemath.required_samples == 55
    assert basemath.evaluate_experiment(0, 30, 0, 40) == 0
    assert basemath.evaluate_experiment(30, 20, 40, 30) == 1


@pytest.mark.parametrize("value", [-0.05, 0, 1, 1.01])
def test_alpha_beta_out_of_bounds(value):
    """
    If we receive an alpha value outside of (0, 1), we should raise an exception
    """
    with pytest.raises(ValueError):
        BaseMathsTest(0.9, 0.01, value, 0.2)
    with pytest.raises(ValueError):
        BaseMathsTest(0.9, 0.01, 0.05, value)


@pytest.mark.parametrize("mean_A", [-10, 0])
def test_mean_non_positive(mean_A):
    """
    mean_A can be any positive number, meaning it cannot be 0 or negative. We
    should raise an exception in this case.
    """
    with pytest.raises(ValueError) as exception_context_manager:
        BaseMathsTest(mean_A, 0.01, 0.05, 0.2, seed="test-experiment")
    expected_exception_text = "mean_A must be positive!"
    assert str(exception_context_manager.value) == expected_exception_text


def test_mean_greater_than_one_and_variance_not_provided():
    """
    In the non-binary case, we can have a mean greater than 1.
    BUT, we need the variance if this happens. So, validate for this case.
    """
    with pytest.raises(ValueError) as exception_context_manager:
        BaseMathsTest(5, 0.01, 0.05, 0.2, seed="test-experiment")
    expected_exception_text = (
        "When variance is not passed, we assume a binary metric -- in this case, "
        "the provided mean must be between 0 and 1 OR the variance must be provided."
    )
    assert str(exception_context_manager.value) == expected_exception_text


def test_mean_greater_than_one_and_variance_provided():
    """
    We accept a mean value that is greater than 1 as long as variance is also provided.
    In this case, we should run without issue.
    """
    basemath = BaseMathsTest(5, 0.2, 0.05, 0.2, seed="test-experiment", var_A=2)
    assert basemath.required_samples == 28
    assert basemath.evaluate_experiment(0, 8, 0, 10) == 0
    assert basemath.evaluate_experiment(10, 9, 10, 12) == 0
    assert basemath.evaluate_experiment(19, 6, 22, 8) == 1


def test_mean_greater_than_one_and_variance_provided_():
    """
    In certain situations with high MDE, the solver hits an issue as the number
    of required samples decays. For now, we just raise an exception when this behavior
    is hit, as the low sample values aren't realistic for an actual experiment
    """
    with pytest.raises(AnalysisException) as exception_context_manager:
        BaseMathsTest(5, 0.5, 0.05, 0.2, seed="test-experiment", var_A=2)
    expected_exception_text = (
        "The numerical solver was not able to find a root for the provided values."
        "This is an internal error that can happen with extreme values that result "
        "in a very low required number of samples."
    )
    assert str(exception_context_manager.value) == expected_exception_text


@pytest.mark.parametrize("variance", [-10, 0])
def test_variance_non_positive(variance):
    """
    The provided variance must be greater than zero -- if it's not, we should
    raise an exception
    """
    with pytest.raises(ValueError) as exception_context_manager:
        BaseMathsTest(50, 0.01, 0.05, 0.2, var_A=variance, seed="test-experiment")
    expected_exception_text = "Variance must be positive if provided!"
    assert str(exception_context_manager.value) == expected_exception_text
