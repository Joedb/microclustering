# Random partitions for microclustering

This demo requires Julia 0.6.0 with the following packages:
Distributions, PyPlot, ProgressMeter.

The file `demo.jl` contains the following function
```julia
function run_demo(xi = 1, sigma = 0.3, zeta = 1,
                  n_train = 300,
                  n_test = 400,
                  n_pred = 30,
                  num_particles = 1000,
                  n_sigma = 10,
                  n_smcruns = 10)
```
It samples a partition of size `n_train + n_test` from the non-exchangeable random partion model with parameters `(xi, sigma, zeta)` whose range is \{1,2,3\} x \[0,1\) x \(0,+inf\). These characterize the Generalized Gamma Process with mean measure

![equation](http://latex.codecogs.com/gif.latex?%5Cxi%20%5C%2C%5Ctheta%5E%7B%5Cxi-1%7D%20%5Cfrac%7B%5Comega%5E%7B-1-%5Csigma%7D%7D%7B%5CGamma%281-%5Csigma%29%7D%5C%2C%20e%5E%7B-%5Czeta%20%5Comega%7D%20%5C%2Cd%5Ctheta%5C%2Cd%5Comega)

The function produces two plots showing the clusters' size trajectories and the frequencies of clusters of given size in log-log scale. The following plots are produced with `run_demo()`. 

<img src="https://github.com/Joedb/microclustering/blob/master/data_clustersize.png" height="500" />
<img src="https://github.com/Joedb/microclustering/blob/master/freqs_data.png" height="500" />

Sequential Monte Carlo with `num_particles` is adopted to find the MLE of `sigma` and `xi` using the first `n_train` points of the simulated partition. The SMC algorithm runs `n_smcruns` times on each point of a grid of the parameters' space: \{1,2,3\} for `xi` and `n_sigma` equidistant points in \[0,0.9\] for `sigma`. A plot of the log-likelihood estimates is produced.

<img src="https://github.com/Joedb/microclustering/blob/master/loglikelihood_smc_estim.png" height="500" />

The prediction step generates `n_pred` partitions of size `n_train + n_test` from the predictive distribution and plots the 95% credible intervals for frequencies of clusters of given size.

<img src="https://github.com/Joedb/microclustering/blob/master/freqs_prediction.png" height="500" />
