# Basemath

Welcome to Basemath, an open-source implementation of the statistical test bearing the same name designed for analyzing
AB experiments.

Basemath employs a one-sided testing approach, where the null hypothesis posits that the treatment performs either
equally or worse than the control concerning the target metric. The test has a predetermined maximum runtime determined
by the input parameters. Additionally, it ensures that both type I and type II errors remain below specified error
thresholds, denoted as α and β, respectively.  Basemath assesses the experiment in batches and terminates prematurely
if it can reject the null hypothesis. The β-spending function employed is is O’Brien-Fleming-like.

What sets Basemath apart is its avoidance of recurrent numerical integration resulting in a straightforward and fast
implementation. For a detailed exploration of Basemath's mathematical foundations, refer to our article
[Basemath’s Test: Group Sequential Testing Without Recurrent Numerical Integration](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4599695).

## Installation
Simply install the library using your package manger of choice.

For instance, with pip:

```pip install basemath-analysis```

## Usage
Demonstrating the usage through an example scenario that references more concrete concepts like RpV, e.g. "Let's suppose we're running an experiment, we choose X for our alpha, Y for beta, etc." and then this being reflected in the example below.

In this example scenario, we can broadly cover:
- What variables need to be provided for initialization
- Consideration for binary vs nonbinary metric
- What helpful information is available after initialization (e.g. number of required samples) and what we might want to do with it

```python
basemath = BaseMathsTest(..., X, Y)
```

We continue with the example scenario here, "We've been running our experiment for however long and collected N samples...", how that translates into the parameters and potential return values

```python
basemath.evaluate_experiment(...)
```


A more detailed example is available [here](path to notebook file in repository)

## Contributing

(How to get started as a contributor, e.g. environment setup, Olivia will write this)
