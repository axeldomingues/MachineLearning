# Machine Learning (Andrew Ng, Coursera) — My 2016 Lab Notebook

In **2016**, I went through **Andrew Ng’s Machine Learning** course on Coursera and kept all the programming exercises in one place.

This repository is intentionally **fundamentals-first**:

- everything is written in **Octave/Matlab** (the course default in 2016)
- I focus on **implementing core algorithms from scratch**
- I keep the small helper scripts and “submit” flow that Coursera used at the time

Alongside the code, I wrote a **portfolio blog series** where I explain the intuition, the implementation choices, and what I learned while debugging.

---

## Blog series (2016)

These posts live on **axeldomingues.dev** (I’ll keep the slugs stable once published):

1. [**Why I’m Learning Machine Learning**](https://axeldomingues.dev/blog/why-im-learning-machine-learning)
2. [**Exercise 1: Linear Regression From Scratch**](https://axeldomingues.dev/blog/linear-regression-from-scratch)
3. [**Linear Regression With Multiple Variables (and Why Vectorization Matters)**](https://axeldomingues.dev/blog/linear-regression-with-multiple-vars)
4. [**Normal Equation vs Gradient Descent (Choosing Tools Like an Engineer)**](https://axeldomingues.dev/blog/normal-equation-vs-gradient-descent)
5. [**Exercise 2: Logistic Regression for Classification (My First Real Classifier)**](https://axeldomingues.dev/blog/logistic-regression-for-classification)
6. [**Regularization: Overfitting in the Real World (and How to Fight It)**](https://axeldomingues.dev/blog/regularization-overfitting-in-the-real-world)
7. [**Exercise 3: One-vs-All + Intro to Neural Networks (Handwritten Digits!)**](https://axeldomingues.dev/blog/intro-to-neural-networks)
8. [**Exercise 4: Neural Networks Learning (Backpropagation Without Tears)**](https://axeldomingues.dev/blog/neural-networks-learning-backpropagation)
9. [**Exercise 5: Debugging ML (Bias/Variance, Learning Curves, and “What to Try Next”)**](https://axeldomingues.dev/blog/debugging-ml-bias-variance)
10. [**Exercise 6: Support Vector Machines (When a Different Model Just Wins)**](https://axeldomingues.dev/blog/support-vector-machines)
11. [**Exercise 7: Unsupervised Learning (K-means) + PCA (Compression & Visualization)**](https://axeldomingues.dev/blog/unsupervised-learning-and-compression)
12. [**Exercise 8 + Course Wrap: Anomaly Detection & Recommenders (and My Next Steps)**](https://axeldomingues.dev/blog/anomaly-detection-and-recommenders)

---

## Repo structure

Each Coursera exercise is kept in its own folder, following the course’s original layout:

```
machine-learning-ex1/
  ex1/         # Octave code + datasets + submit scripts (2016 Coursera style)
machine-learning-ex2/
  ex2/
...
machine-learning-ex8/
  ex8/
```

Inside each `exN/` directory you’ll find:

- `exN.m` (and sometimes `exN_*.m`) — the main script you run
- the functions you implement (`computeCost.m`, `nnCostFunction.m`, …)
- datasets (`.txt` / `.mat`)
- helper utilities (plotting, feature mapping, `fmincg.m`, etc.)
- `submit.m` and `token.mat` (how Coursera validated exercises in 2016)

> **Note on `token.mat`:** this file was used for the Coursera submission system back in 2016.  
> If you fork this repo, consider removing it so you don’t publish old course tokens.

---

## Exercises overview

### Exercise 1 — Linear Regression (1 variable + multiple variables)

Folder: `machine-learning-ex1/ex1`

What’s inside (high level):

- univariate linear regression:
  - `plotData.m`, `computeCost.m`, `gradientDescent.m`
- multivariate linear regression:
  - `featureNormalize.m`, `computeCostMulti.m`, `gradientDescentMulti.m`
- closed-form solution:
  - `normalEqn.m`
- entrypoints:
  - `ex1.m`, `ex1_multi.m`

Run it:

```bash
cd machine-learning-ex1/ex1
octave ex1.m
octave ex1_multi.m
```

---

### Exercise 2 — Logistic Regression (classification)

Folder: `machine-learning-ex2/ex2`

Highlights:

- `sigmoid.m`
- `costFunction.m` (logistic regression)
- `costFunctionReg.m` (regularized logistic regression)
- `mapFeature.m`, `plotDecisionBoundary.m`
- entrypoints: `ex2.m`, `ex2_reg.m`

Run it:

```bash
cd machine-learning-ex2/ex2
octave ex2.m
octave ex2_reg.m
```

---

### Exercise 3 — Multi-class classification + intro to neural networks

Folder: `machine-learning-ex3/ex3`

Highlights:

- `oneVsAll.m`, `lrCostFunction.m`
- `predictOneVsAll.m`
- simple neural network forward-pass: `predict.m`
- entrypoints: `ex3.m`, `ex3_nn.m`

Run it:

```bash
cd machine-learning-ex3/ex3
octave ex3.m
octave ex3_nn.m
```

---

### Exercise 4 — Neural network learning (backpropagation)

Folder: `machine-learning-ex4/ex4`

Highlights:

- `nnCostFunction.m` (forward + backprop + regularization)
- gradient checking utilities:
  - `checkNNGradients.m`, `computeNumericalGradient.m`
- `randInitializeWeights.m`, `sigmoidGradient.m`
- entrypoint: `ex4.m`

Run it:

```bash
cd machine-learning-ex4/ex4
octave ex4.m
```

---

### Exercise 5 — Debugging ML: bias/variance + learning curves

Folder: `machine-learning-ex5/ex5`

Highlights:

- regularized linear regression: `linearRegCostFunction.m`
- training: `trainLinearReg.m`
- diagnostics:
  - `learningCurve.m`
  - `validationCurve.m`
- feature engineering helpers:
  - `polyFeatures.m`, `featureNormalize.m`, `plotFit.m`
- entrypoint: `ex5.m`

Run it:

```bash
cd machine-learning-ex5/ex5
octave ex5.m
```

---

### Exercise 6 — Support Vector Machines (SVM)

Folder: `machine-learning-ex6/ex6`

Highlights:

- SVM training/prediction: `svmTrain.m`, `svmPredict.m`
- kernels: `linearKernel.m`, `gaussianKernel.m`
- parameter search: `dataset3Params.m`
- spam classifier pipeline:
  - `processEmail.m`, `emailFeatures.m`, `vocab.txt`
- entrypoints: `ex6.m`, `ex6_spam.m`

Run it:

```bash
cd machine-learning-ex6/ex6
octave ex6.m
octave ex6_spam.m
```

---

### Exercise 7 — Unsupervised learning: K-means + PCA

Folder: `machine-learning-ex7/ex7`

Highlights:

- K-means:
  - `findClosestCentroids.m`, `computeCentroids.m`, `runkMeans.m`
- PCA:
  - `pca.m`, `projectData.m`, `recoverData.m`
- entrypoints: `ex7.m`, `ex7_pca.m`

Run it:

```bash
cd machine-learning-ex7/ex7
octave ex7.m
octave ex7_pca.m
```

---

### Exercise 8 — Anomaly detection + recommender systems

Folder: `machine-learning-ex8/ex8`

Highlights:

- anomaly detection:
  - `estimateGaussian.m`, `selectThreshold.m`, `visualizeFit.m`
- collaborative filtering (recommenders):
  - `cofiCostFunc.m`, `normalizeRatings.m`, `loadMovieList.m`
- entrypoints: `ex8.m`, `ex8_cofi.m`

Run it:

```bash
cd machine-learning-ex8/ex8
octave ex8.m
octave ex8_cofi.m
```

---

## Getting started (2016)

### 1) Install Octave

In 2016 I used **GNU Octave** as a Matlab-compatible environment.

Verify:

```bash
octave --version
```

### 2) Run any exercise

Pick an exercise folder and run the entry script:

```bash
cd machine-learning-ex2/ex2
octave ex2.m
```

### 3) Plots

Most exercises open plots (decision boundaries, clustering progress, PCA projections).  
If plots don’t show, it’s usually a graphics backend issue in your Octave setup.

---

## Why this repo exists (portfolio intent)

Even though this is “just course code”, it demonstrates the skill I care about as a software engineer:

- reading a spec (the PDF instructions) and implementing missing functions
- debugging numerical code (shapes, vectorization, convergence)
- building intuition for when a model is underfitting vs overfitting
- treating ML like an engineering system (data checks, baselines, diagnostics)

This is the foundation I’m building on before going deeper into modern deep learning tooling.

---

## Credits

- **Andrew Ng — Machine Learning (Coursera)**  
  Course page: https://www.coursera.org/learn/machine-learning
- Datasets and starter code originate from the course assignments (2016 era).

---

## License / usage

No explicit license is included yet. If you want to reuse anything here, treat it as **study/reference code** and give proper attribution.
