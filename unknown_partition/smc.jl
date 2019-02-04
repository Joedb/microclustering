# --------------------------------------------------------------------------------------------------
# Copyright (C) Giuseppe Di Benedetto, University of Oxford
# benedett@stats.ox.ac.uk
# October 2017
# --------------------------------------------------------------------------------------------------

# *********************************** #
#    SMC for likelihood estimation    #
# *********************************** #
using DistributedArrays

function smc(obs::Array{Int64,2},
             num_particles::Int64,
             c::Int64,
             sigma::Float64,
             ze::Float64;
             var_dir = 1.)
    # c is what is denoted by \xi in the paper
    # obs is a V*n array of Int64
    
    n                    = size(obs)[2]           # number of points in the data
    V                    = size(obs)[1]           # size of the vocabulary
    N                    = num_particles          # number of particles
    theta                = zeros(Float64,N)       # current value of theta for every particle
    old_theta            = zeros(Float64,N)       # previous value of theta for every particle
    tau                  = zeros(Float64,N)       # current value of tau for every particle
    old_tau              = zeros(Float64,N)       # previous value of theta for every particle
    thetastar            = zeros(Float64,n,N)     # unique thetas (in columns) for every particle 
    tmp_thetastar        = zeros(Float64,n,N)     # used to update thetastar_traj
    ancestor             = zeros(Int64,n,N)       # matrix of the ancestor indeces
    log_weights          = zeros(Float64,N)
    log_weights_proposal = zeros(Float64,N)
    logsum_weights       = 0.
    log_weight_proposal  = zeros(Float64,N)
    
    sumobs = sum(obs,1)
    w      = zeros(Float64,n)
    tmp    = zeros(Float64,n)

    clust_label     = zeros(Int64,n,N)         # cluster label for each point in each particle
    tmp_clust_label = zeros(Int64,n,N)
    nclust          = zeros(Int64,N)           # numeber of clusters for each particle
    tmp_nclust      = zeros(Int64,N)
    size_clust      = zeros(Int64,n,N)         # cluster sizes for each particle
    tmp_size_clust  = zeros(Int64,n,N)
    ss              = zeros(Int64,V)
    lg_clust        = zeros(Float64,n,N)       # lgamma(sum(s))-sum(lgamma.(s)) forall clust and prtcl
    tmp_lg_clust    = zeros(Float64,n,N)
    
    ESS                = ones(Float64,n)       # Eff. Sample Size
    normalized_weights = zeros(Float64,N)      # store the normalized weights of the current step
    loglike_estim      = 0.

    arrival_times = asymp_arrivaltime(n+1,c,sigma,ze)  # deterministic arrival times
    
    h                 = var_dir * ones(V)
    gamma_const_obsbe = zeros(Float64,n)
    gamma_const_obs   = zeros(Float64,n)
    gamma_const_o     = zeros(Float64,n)
    for i in 1:n
        gamma_const_o[i]     = lgamma(sum(h+obs[:,i])) - sum(lgamma.(h+obs[:,i])) 
        gamma_const_obs[i]   = lgamma(1+sum(obs[:,i])) - sum(lgamma.(1+obs[:,i])) 
        gamma_const_obsbe[i] = lgamma(1+sum(obs[:,i])) - sum(lgamma.(1+obs[:,i])) +
            lgamma(sum(h)) - sum(lgamma.(h)) +
            sum(lgamma.(obs[:,i]+h)) - lgamma(sum(obs[:,i]+h)) 
    end

    # STEP t = 1:
    for i in 1:N
        # Propagation 
        tau[i]           = sample_arrivaltime(0.,arrival_times,1)
        theta[i]         = rand() * tau[i]
        thetastar[1,i]   = theta[i]
        clust_label[1,i] = 1
        # Weights
        log_weights[i] = -laplace_int_gamma(tau[i],c,sigma,ze) +
            log_thetadens(theta[i], tau[i],c,sigma,ze) +
            -logpdf(SymTriangularDist(arrival_times[1],
                                  (arrival_times[2]-arrival_times[1])/2),tau[i])
        nclust[i]       = 1
        size_clust[1,i] = 1
        lg_clust[1,i]   = lgamma(sum(h+obs[:,1]))-sum(lgamma.(h+obs[:,1]))
    end
    M = maximum(log_weights)
    
    logsum_weights     = log(sum(exp.(log_weights)))
    loglike_estim     += logsum_weights
    normalized_weights = exp.(log_weights-M) / sum(exp.(log_weights-M))
    ESS[1]             = 1 / (sum(normalized_weights.^2))
    
    # STEP t = 2,..,n :
    for t in 2:n
        # Adaptive resampling 
        if ESS[t-1] < N/2            
            nclust_max = maximum(nclust)
            @inbounds tmp_thetastar[1:nclust_max,:] = thetastar[1:nclust_max,:]
            @inbounds tmp_clust_label[1:t,:]        = clust_label[1:t,:]
            @inbounds tmp_size_clust[1:t,:]         = size_clust[1:t,:]
            @inbounds tmp_nclust[:]                 = nclust[:]
            @inbounds tmp_lg_clust[1:t,:]           = lg_clust[1:t,:]
            
            @inbounds ancestor[t,:] = strat_res(normalized_weights)
            @inbounds @simd for i in 1:N
                thetastar[1:nclust[i],i] = tmp_thetastar[1:nclust[i],ancestor[t,i]]
                clust_label[1:t,i]       = tmp_clust_label[1:t,ancestor[t,i]]
                size_clust[1:t,i]        = tmp_size_clust[1:t,ancestor[t,i]]
                nclust[i]                = tmp_nclust[ancestor[t,i]]
                lg_clust[1:t,i]          = tmp_lg_clust[1:t,ancestor[t,i]]
                old_theta[i]             = theta[ancestor[t,i]]
                old_tau[i]               = tau[ancestor[t,i]]
            end
            @inbounds log_weights = fill(-log(N), N)
        else
            @inbounds ancestor[t,:] = 1:N
            @inbounds old_theta     = copy(theta)
            @inbounds old_tau       = copy(tau)
            @inbounds log_weights   = log.(normalized_weights) 
        end
        
        # Propagation step
        obs_value = obs[:,t]
        # propagate the particles
        @inbounds log_weight_proposal = propagate_and_weight(N,t,V,old_tau,tau,
                                                             arrival_times,
                                                             theta,thetastar,nclust,
                                                             clust_label,size_clust,
                                                             obs, sumobs[t], ss,
                                                             obs_value,lg_clust,
                                                             gamma_const_obs,
                                                             gamma_const_obsbe,
                                                             gamma_const_o,
                                                             c,sigma,ze,h,
                                                             w,tmp) 
        
        @inbounds loglike_estim     += log(sum(exp.(log_weight_proposal + log_weights)))
        @inbounds log_weights        = log_weights[collect(ancestor[t,:])] + log_weight_proposal
        @inbounds M                  = maximum(log_weights)
        @inbounds logsum_weights     = log(sum(exp.(log_weights)))
        @inbounds normalized_weights = exp.(log_weights-M) / sum(exp.(log_weights-M))
        @inbounds ESS[t]             = 1 / (sum(normalized_weights.^2))
    end

# Sample trajectory
index               = rand(Categorical(normalized_weights),1)
sampled_tstar       = thetastar[:,index]
sampled_tau         = tau[index]
sampled_nclust      = nclust[index]
sampled_clust_label = clust_label[:,index]

return Dict("loglike_estim" => loglike_estim,
            "n_clust"       => sampled_nclust,
            "part"          => sampled_clust_label,
            "tstar"         => sampled_tstar,
            "tau"           => sampled_tau,
            "ESS"           => ESS)
end
