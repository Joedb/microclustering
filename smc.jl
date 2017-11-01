# --------------------------------------------------------------------------------------------------
# Copyright (C) Giuseppe Di Benedetto, University of Oxford
# benedett@stats.ox.ac.uk
# October 2017
# --------------------------------------------------------------------------------------------------



# *********************************** #
#    SMC for likelihood estimation    #
# *********************************** #

function smc(obs, num_particles, c, σ, zeta)
    
    N = num_particles                      # number of particles
    n = length(obs)                        # number of points in the data
    theta_particle = zeros(Float64,N)      # current value of theta for every particle
    old_theta_particle = zeros(Float64,N)  # previous value of theta for every particle
    tau_particle = zeros(Float64,N)        # current value of tau for every particle
    old_tau_particle = zeros(Float64,N)    # previous value of theta for every particle
    thetastar_traj = zeros(Float64,n,N)    # unique thetas (in columns) for every particle 
    tmp_thetastar_traj = zeros(Float64,n,N)# used to update thetastar_traj
    ancestor = zeros(Int64,n,N)            # matrix of the ancestor indeces
    log_weights = zeros(Float64,N)
    log_weights_proposal = zeros(Float64,N)
    logsum_weights = 0.
    nclust_traj = 0
    sizeclust_traj = zeros(Int32,n)
    v = 0.5^(-1/3)                         # used to store the shrinking variance of the tau-proposal
    ESS = ones(Float64,n)                  # Eff. Sample Size
    normalized_weights = zeros(Float64,N)  # used to store the normalized weights of the current step
    loglike_estim = 0.

    
    # STEP t = 1:
    for i in 1:N
        # Propagation 
        tau_particle[i] = abs.(rand(Normal(0,v),1))[1] 
        theta_particle[i] = rand() * tau_particle[i]
        thetastar_traj[1,i] = theta_particle[i]
        # Weights
        log_weights[i] = -laplace_int_gamma(tau_particle[i],c,σ,zeta) +log_thetadens(theta_particle[i], tau_particle[i],c,σ,zeta) -log_two_norm_density(tau_particle[i],0,v) + log(tau_particle[i])
    end
    nclust_traj = 1
    sizeclust_traj[1] = 1
    M = maximum(log_weights)
    
    logsum_weights = log(sum(exp.(log_weights)))
    loglike_estim += logsum_weights
    normalized_weights = exp.(log_weights-M) / sum(exp.(log_weights-M))
    ESS[1] = 1 / (sum(normalized_weights.^2))
    
    # STEP t = 2,..,n :
    for t in 2:n
        # Adaptive resampling 
        if ESS[t-1] < N/2
            @inbounds tmp_thetastar_traj[1:nclust_traj,:] = thetastar_traj[1:nclust_traj,:]
            @inbounds ancestor[t,:] = strat_res(normalized_weights)
            @inbounds @simd for i in 1:N
                thetastar_traj[1:nclust_traj,i] =
                    tmp_thetastar_traj[1:nclust_traj,ancestor[t,i]]
                old_theta_particle[i] = theta_particle[ancestor[t,i]]
                old_tau_particle[i] = tau_particle[ancestor[t,i]]
            end
            @inbounds log_weights = fill(-log(N), N)
        else
            @inbounds ancestor[t,:] = 1:N
            @inbounds old_theta_particle = copy(theta_particle)
            @inbounds old_tau_particle = copy(tau_particle)
            @inbounds log_weights = log.(normalized_weights) #log_weights - logsum_weights
        end
        
        # Propagation step
        v = 0.5*t^(-1/3)
        obs_value = obs[t]
        @inbounds tau_particle = abs.(rand(Normal(0,v),N)) .+ old_tau_particle
        if obs_value <= nclust_traj
            @inbounds theta_particle = thetastar_traj[obs_value,:] 
            @inbounds log_weight_proposal = log_weight_oldclust_vec(N,
                                                                    nclust_traj,
                                                                    sizeclust_traj,
                                                                    obs_value,
                                                                    tau_particle,
                                                                    old_tau_particle,
                                                                    thetastar_traj,
                                                                    v, c, σ, zeta)
            @inbounds loglike_estim += log(sum(exp.(log_weight_proposal + log_weights)))
            @inbounds log_weights = log_weights[collect(ancestor[t,:])] + log_weight_proposal
            sizeclust_traj[obs_value] += 1
        else
            @inbounds theta_particle = tau_particle .* rand(N)
            @inbounds thetastar_traj[nclust_traj+1,:] = theta_particle
            @inbounds log_weight_proposal = log_weight_newclust_vec(N,
                                                                    nclust_traj,
                                                                    sizeclust_traj,
                                                                    tau_particle,
                                                                    old_tau_particle,
                                                                    thetastar_traj,
                                                                    v, c, σ, zeta)
            @inbounds loglike_estim += log(sum(exp.(log_weight_proposal + log_weights)))
            @inbounds log_weights = log_weights[collect(ancestor[t,:])] + log_weight_proposal
            nclust_traj += 1
            sizeclust_traj[nclust_traj] = 1
        end

        @inbounds M = maximum(log_weights)

        @inbounds logsum_weights = log(sum(exp.(log_weights)))
        @inbounds normalized_weights = exp.(log_weights-M) / sum(exp.(log_weights-M))
        @inbounds ESS[t] = 1 / (sum(normalized_weights.^2))
    end

# Sample trajectory
index = rand(Categorical(normalized_weights),1)
sampled_tstar = thetastar_traj[:,index]
sampled_tau = tau_particle[index]

return (loglike_estim,
        thetastar_traj, 
        tau_particle,
        ancestor,
        log_weights,
        nclust_traj,
        sizeclust_traj,
        ESS,
        sampled_tstar,
        sampled_tau)                
end
