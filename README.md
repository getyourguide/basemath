# Basemath

Welcome to Basemath, an open-source implementation of the statistical test bearing the same name designed for analyzing
AB experiments.

Basemath employs a one-sided testing approach, where the null hypothesis posits that the treatment performs either
equally or worse than the control concerning the target metric. The test has a predetermined maximum runtime determined
by the input parameters. Additionally, it ensures that both type I and type II errors remain below specified error
thresholds, denoted as α and β, respectively.  Basemath assesses the experiment in batches and terminates prematurely
if it can reject the null hypothesis. The β-spending function employed is is O’Brien-Fleming-like.
Given that the majority of experiments yield either flat or negative results, stopping early in this scenario saves more
running time compared to stopping in the less common case of a significant uplift.

What sets Basemath apart is its avoidance of recurrent numerical integration resulting in a straightforward and fast
implementation. For a detailed exploration of Basemath's mathematical foundations, refer to our article
[Basemath’s Test: Group Sequential Testing Without Recurrent Numerical Integration](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4599695).

## Installation
Simply install the library using your package manger of choice.

For instance, with pip:

```pip install basemath-analysis```

## Usage
### The Binary Case

Assume you are conducting an experiment on your platform. You have modified the user experience (UX) and aim to
demonstrate that this change improves the conversion of visitors to customers. To achieve this, you present the
current UX (control group) to half of your visitors and the new UX (treatment group) to the other half. Data on the
number of people visiting your platform and the number of visitors converting to customers are processed on a daily
basis. Your expectation is that the UX change will lead to a relative increase in the conversion rate (number of
customers / number of visitors) by at least 1%.

You initialize Basemath using the following Python code:

```python
import basemath as bm
bm_test = bm.BaseMathsTest(cr_A, mde, alpha, beta, seed="experiment_name")
```
The parameters are as follows:

* __cr_A__: The estimated conversion rate of your control group.
* __mde__: The minimal relative uplift you are aiming for (1% in our example).
* __alpha__: The maximal type I error you are willing to tolerate (α is often set to 5%).
* __beta__: The maximal type II error you are willing to tolerate (β is often set to 20%).
* __seed__: As the algorithm contains a random element, set a seed to ensure consistent outcomes when running the
algorithm repeatedly on the same data. The seed is generated from a string such as the unique experiment name.
After initialization, you can check the maximum number of required samples for each variation to estimate the running
time:

```python
print(bm_test.required_samples)
```

Enter the most recent data daily into the instance to determine if there is a significant outcome:

```python
bm_test.evaluate_experiment(
    previous_customer_delta,
    customer_delta_since_yesterday,
    previous_visitor_number,
    visitors_since_yesterday
```

The parameters for this method are:

* __previous_customer_delta__: The difference between the overall number of customers in the treatment and control groups
as of the last check-in, i.e., as of yesterday in our example.
* __customer_delta_since_yesterday__: The difference between the overall number of customers in the treatment and control
groups since the last check-in, i.e., since yesterday in our example.
* __previous_visitor_number__: The number of visitors per variation as of the last check-in.
* __visitors_since_yesterday__: The number of visitors per variation since the last check-in.

The outcome of this method is either 0, 1, or -1. If Basemath hasn't reached a significant conclusion yet, it returns
a 0. If the alternative hypothesis that the treatment is significantly better than the control can be rejected, it
returns a -1. This can occur in each evaluation step. If Basemath finds a significant uplift for the treatment
group, it returns a 1. This can only happen once the maximum number of required samples has been reached.

Once the outcome is no longer 0, the test has concluded, and the experiment can be stopped. Further evaluation beyond
this point is futile.

### The Continuous Case
Some changes may not result in more visitors converting to customers. The number of customers might remain the same;
however, customers in the treatment group might spend more money on your platform. In this case, change your target
metric from conversion rate to revenue per visitor (sum of revenue / number of visitors).

For a continuous target variable, the initialization of Basemath only slightly changes:
```python
bm_test = bm.BaseMathsTest(rpv_A, mde, alpha, beta, var_A=var_A, seed="experiment_name")
```
Instead of the estimated conversion rate, enter the estimated average revenue per visitor (rpv_A) for the control
group. Additionally, estimate the variance of the revenue per visitor, considering visitors who do not convert (i.e.,
assuming a revenue of 0 for them). The other parameters remain the same as in the binary case.

Basemath only works with two variations and a 50/50 split in traffic.

More detailed examples are available [here](./examples).

## Contributing

To contribute, you'll need a working python 3.8+ installation. We also recommend setting up a virtual environment for the project. You'll also need to [install poetry](https://python-poetry.org/docs/) if it's not already present.

Once you have the dependencies installed (with `poetry install`), you can set up pre-commit with `pre-commit install`. We run pre-commit in our CI as well, but it's recommended to install it locally so that you get immediate feedback from our various linters.

Beyond that, there's not really anything else you need to know to start contributing! Please create a pull request with whatever changes you'd like to propose and increment the version and update the changelog if necessary.
