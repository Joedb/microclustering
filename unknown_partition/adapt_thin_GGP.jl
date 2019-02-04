# Functions:
# --------------------------------------------------------------------------------------------------
# GGPrnd  samples weights from Generalized Gamma CRM in the domain [t_in, t_fin] using adaptive
#         thinning alg deveoped in [Favaro, Teh - Statistical Science, 2013] with truncation level T
#         sigma,tau: parameters of the L\'{e}vy measure of the GGP(sigma,tau):
#                     rho(dw) = \Gamma(1-sigma)^-1 * exp(-tau * w) * w^(-1-sigma) dw 
#         c: parameter of the base measure alpha(dx) = c x^(c-1) dx, with c \in {1,2,3}
#
# --------------------------------------------------------------------------------------------------
# nexch_rnd_part  samples the non-exchangeable random partition from the point process
#                 n: number of points in the partition 
#                 c, sigma, tau: parameters of the mean measure of the GGP 
#                 p: if p=0 ot plots the point process that generates the partition
#
# --------------------------------------------------------------------------------------------------
# posterior_CRM  samples partition from the predictive distribution of the point process given
#                a time-ordered sample from it
#                t: arrival time of the last obeservation
#                k: number of points in the sample from the predictive (including the observed ones)
#                obs_partition: time-ordered cluster labels
#                fixed_atoms: time_ordered unique theta values from the point process 
#                c, sigma, tau: parameters of the mean measure of the GGP 
#                stop_time: if stop_time=0 it will compute 
#
# --------------------------------------------------------------------------------------------------
# Copyright (C) Giuseppe Di Benedetto, University of Oxford
# benedett@stats.ox.ac.uk
# October 2017
# --------------------------------------------------------------------------------------------------


function GGPrnd(t_in, t_fin, c, sigma, tau, T; post = 0)
    maxiter = Int64(1e8)
    if post == 1
        tau =1
        if t_in != 0
            error("t_in must be equal to zero for the posterior")
        end
        if c == 1
            mass = 1/(1+sigma) * ((tau + t_fin)^(1+sigma) - tau^(1+sigma))
        elseif c ==2 
            mass = c * ((t_fin+1)^(sigma+2)-(sigma+2)*t_fin -1)/(sigma^2+3*sigma +2)
        elseif c==3
            mass = c * (2*t_fin*(t_fin^2+3*t_fin+3)*(t_fin+1)^sigma - t_fin*(sigma+3)*(2+t_fin*(sigma+2)) +2*((t_fin+1)^sigma-1))/((sigma+1)*(sigma+2)*(sigma+3))
        else
            error("xi must be 1 or 2 or 3")
        end
    else
        mass =  (t_fin^c - t_in^c)
    end
    completed = false
    t = T
    k = 0
    N = []
    log_cst = log(mass) - log(gamma(1-sigma)) - log(tau)
    sigmap1 = 1 + sigma
    tauinv = 1 / tau
    for i in 1:maxiter
        log_r = log(-log(rand())) # Sample exponential random variable e of unit rate
        log_G = log_cst - sigmap1 * log(t) - tau*t
        if log_r > log_G
            completed = true
            break
        end
        t_new = t-tauinv * log(1 - exp(log_r-log_G))
        if log(rand()) < sigmap1 * (log(t)-log(t_new))
            k = k + 1 
            push!(N,t_new)
        end
        t = t_new
    end
    if !completed
        T *= 10
        #warn("T too small: its value has been increased at $T")
        N = GGPrnd(t_in, t_fin, c, sigma, tau, T)
    end    
    return N
end

function relabel(v)
    n = length(v)
    count = 1
    rel_v = ones(Int64,n)
    for j in 2:n
        pos = findfirst(v[1:j], v[j])
        if pos == j
            count += 1
            rel_v[j] = count
        else
            rel_v[j] = rel_v[pos]
        end
    end
    return rel_v
end    

function nexch_rndpart(n, c, sigma, tau; t_fin=0, T=1e-8, p=1)
    if t_fin == 0
        extratime = 1.5
        if tau > .8
            extratime = 2
        end
        t_fin = (((c+1)*n * tau^(1-sigma)) ^ (1/(c+1)) ) * extratime
    end
    w = GGPrnd(0., t_fin, c, sigma, tau, T)
    k = length(w)
    theta = zeros(k)
    for j in 1:k
        flag = 0
        while flag == 0
            x = t_fin * rand()
            if rand() < (x/t_fin)^(c-1)
                theta[j] = x
                flag = 1
            end
        end
    end
    clusters = zeros(2,1)
    masses = []
    for j in 1:k
        scale = 1 / (w[j])
        u = rand(Exponential(scale))
        while u < t_fin # inside the big rectangle
            if u > theta[j] # inside the region
                clusters = hcat(clusters,[u,j])
                push!(masses,w[j])
            end
            u += rand(Exponential(scale))
        end
    end
    theta_clus = clusters[:,2:end]
    clusters = clusters[:,2:end]
    times = collect(clusters[1,:])
    sorted_index = sortperm(times)#
    l = length(clusters[2,:])
    partition = zeros(Int64,l)
    clusters_tmp = zeros(Float64,2,l)
    for i in 1:l
        clusters_tmp[:,i] = clusters[:,sorted_index[i]]
    end
    clusters = copy(clusters_tmp)
    for j in 1:l
        partition[j] = clusters[2,j]
    end
    partition = relabel(partition)
    return (partition[1:n])
end

function ttransf_base_measure(x, c, sigma, t, tau)
    return x^(c-1) * (tau+t-x)^sigma
end

function tmax_base(c, sigma, t, tau)
    if c == 1
        return (tau+t) ^ sigma
    else
        return ttransf_base_measure(min(t,(c-1)*(tau+t)/(c+sigma-1)) ,c, sigma, t, tau)
    end
end

function posterior_CRM(t, k,
                       obs_partition,
                       fixed_atoms,
                       c, sigma, ze; stop_time=0, T=1e-8)
    if stop_time == 0
        extratime = 1.3
        if sigma >= .8
            extratime = 1.6
        end
        stop_time =  (((c+1)*k * ze^(1-sigma)) ^ (1/(c+1))) * extratime
    elseif t > stop_time
        error("It must be stop_time > t")
    end
    stats_obs_partition = size_distr(obs_partition)
    n_fa = stats_obs_partition["n_clust"] 
    if fixed_atoms[n_fa + 1] != 0
        error("incorrect smc sample")
    end
    sizes = stats_obs_partition["clust_sizes"] 
    M = tmax_base(c, sigma, t, ze)
    u = GGPrnd(0., t, c, sigma, ze, T, post=1)
    n = length(u)
    N = n + n_fa
    omega = zeros(Float64, N)
    theta = zeros(Float64, N)
    newpoint = 0.
    flag = 0
    for i in 1:n_fa
        omega[i] = rand(Gamma(sizes[i]-sigma, 1/(ze+t-fixed_atoms[i])))
    end
    theta[1:n_fa] = fixed_atoms[1:n_fa]
    for i in (n_fa+1):N
        while flag == 0
            newpoint = rand() * t
            if rand() < ttransf_base_measure(newpoint, c, sigma, t, ze) / M 
                flag = 1
                theta[i] = newpoint
                omega[i] = u[i-n_fa] / (ze+t-theta[i])
            end
        end
        flag = 0
    end

    clusters = zeros(3,1) # each column: label, time, is(fixed_atom)

    # "Observed" points
    relabel_fixed_atoms = relabel(fixed_atoms)
    obs_fake_times = linspace(0.1, t, length(obs_partition))
    for i in 1:length(obs_partition)
        clusters = hcat(clusters, [obs_fake_times[i],theta[obs_partition[i]],1])
    end
    check_rel = collect(clusters[2,2:end])
        
    # Propagation of fixed atoms and remaining mass
    for i in 1:N
        scale = 1 / omega[i]
        u = t + rand(Exponential(scale))
        while u < stop_time  # inside the big rectangle            
            if i <= n_fa
                clusters = hcat(clusters,[u,theta[i],1])#i,1])
            else
                clusters = hcat(clusters,[u,theta[i],0])#i,0])
            end
            u += rand(Exponential(scale))
        end
    end

    # New points in the upper region
    new_omega = GGPrnd(t, stop_time , c, sigma, ze, T)
    n_new = length(new_omega)
    new_theta = zeros(n_new)
    for j in 1:n_new
        flag = 0
        while flag == 0
            x = t + (stop_time - t) * rand()
            if rand() < (x/stop_time)^(c-1)
                new_theta[j] = x
                flag = 1
            end
        end
    end    
    for j in 1:n_new
        scale = 1 / new_omega[j]
        u = t + rand(Exponential(scale))
        while u < stop_time # inside the big rectangle
            if u > new_theta[j] # inside the triangle
                clusters = hcat(clusters,[u,new_theta[j],0])#N+j,0])
            end
            u += rand(Exponential(scale))
        end
    end
    # Ordering the points and relabel
    clusters = clusters[:,2:end]
    times = collect(clusters[1,:])
    sorted_index = sortperm(times)
    l = length(clusters[2,:])
    partition = zeros(Int64,l)  
    clusters_tmp = zeros(Float64,3,l)
    for i in 1:l
        clusters_tmp[:,i] = clusters[:,sorted_index[i]]
    end
    clusters = copy(clusters_tmp)
    partition = relabel(collect(clusters[2,:]))
    if collect(check_rel[1:length(obs_partition)]) != collect(clusters[2,1:length(obs_partition)])
        println(sum(check_rel.==clusters[2,1:length(obs_partition)]))
        println(length(check_rel))
        println(length(clusters[2,1:length(obs_partition)]))
        error("incorret relabeling!")
    end

    return partition[1:k]
end

