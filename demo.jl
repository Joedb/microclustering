using Distributions, PyPlot, ProgressMeter
include("auxiliaryft.jl")
include("adapt_thin_GGP.jl")
include("functions.jl")
include("smc.jl")

println("

***************************************************** 
*                                                   *
*   Run the demo with run_demo function specifying  * 
*   the parameters of the non-exchangeable random   *
*   partition (xi, sigma, zeta)                     *
*                                                   *
*****************************************************

")

function run_demo(xi=1, sigma=.3, zeta=1,
                  n_train = 300,
                  n_test = 400,
                  n_pred = 30,
                  num_particles = 1000,
                  n_sigma = 10,
                  n_smcruns = 10)
    println("

    n_train               $(n_train)
    n_test                $(n_test)
    predictive samples    $(n_pred)
    num particles smc     $(num_particles)
    smc runs              $(n_smcruns)
    parameters            ($xi, $sigma, $zeta) 
    
    ")
    
    if !(xi in [1,2,3])
        error("xi must be in {1, 2, 3}")
    end
    n_tot = n_train + n_test

    srand(12345678)
    tic()
    # Sample partition
    data = nexch_rndpart(n_tot, xi, sigma, zeta)
    train_data = data[1:n_train]
    test_data = relabel(data[(n_train+1):n_tot])
    PyPlot.close()
    clustsize_asymp(data,"data",p=1)
    freqplots(data,0,name="data")
    println("Partition sampling done")
        
    # SMC inference
    grid_sigma = linspace(0,.9,n_sigma)
    grid_xi = [1,2,3]
    loglike = zeros(3,n_sigma,n_smcruns)
    means = zeros(3, n_sigma)
    sd = zeros(3, n_sigma)
    @showprogress 1 " SMC running..." for j in 1:n_sigma
        for i in 1:3
            for k in 1:n_smcruns
                loglike[i,j,k] = smc(train_data, num_particles,
                                     grid_xi[i], grid_sigma[j], zeta)[1]
            end
            means[i,j] = mean(collect(loglike[i,j,:]))
            sd[i,j] = std(collect(loglike[i,j,:]))
        end
    end
    I = findmax(means)[2]
    xi_max = ((I-1) % 3) + 1
    sigma_max_ind = Int64((I - xi_max)/3 + 1)
    sigma_max = grid_sigma[sigma_max_ind]
    println("\nMaximum log-likelihood:\t$(means[xi_max, sigma_max_ind])")
    println("MLE:\t xi = $(xi_max)\t sigma = $(sigma_max)\t zeta = $zeta")
    smc_plot([sigma_max,means[xi_max,sigma_max_ind]], means, sd, grid_sigma)    
    PyPlot.close()
        
    # Predictive
    println()
    pred_test = Array{Int64}(n_test, n_pred)
    @showprogress 1 " Prediction..." for i in 1:n_pred
        pred_test[:,i] = prediction(n_tot, train_data,num_particles,
                                   xi_max, sigma_max, zeta)
    end
    freqplots(test_data, pred_test, m=n_pred, name="prediction")
    PyPlot.close()
    println()
    toc()
end

