# --------------------------------------------------------------------------------------------------
# Copyright (C) Giuseppe Di Benedetto, University of Oxford
# benedett@stats.ox.ac.uk
# October 2017
# --------------------------------------------------------------------------------------------------

# Integral of the Laplace exponent
function laplace_int_gamma(x, c, σ, zeta)
    if σ == 0
        if c == 1
            return  -x + (x+zeta)*log((x+zeta)/zeta)
        elseif c == 2
            return -1.5*x^2 - x*zeta + (x+zeta)^2 * log((x+zeta)/zeta)
        elseif c == 3
            return -(11*x^3 + 15*(x^2)*zeta +6*x*zeta^2)/6 +(x+zeta)^3 *log((x+zeta)/zeta)
        else throw("must be c ∈ {1,2,3}")
        end
    elseif 0 < σ < 1
        if c == 1
            return (-x*zeta^σ+((x+zeta)^(1+σ)-zeta^(1+σ))/(1+σ)) / σ
         elseif c == 2
            return (-x^2*zeta^σ + 2*((x+zeta)^(2+σ)-zeta^(1+σ) *((2+σ)*x+zeta))/((1+σ)*(2+σ))) / σ
        elseif c == 3
            return (-x^3*(zeta^σ) +3*(2*((x+zeta)^(3+σ))-zeta^(1+σ) *((2+σ)*(3+σ)*(x^2)+2*(3+σ)*x*zeta+2*(zeta^2))) /((1+σ)*(2+σ)*(3+σ))) / σ
        else throw("must be c ∈ {1,2,3}")
        end
    else print("must be σ ∈ (0,1)")
    end
end

function laplace_int_gamma_vec(x, c, σ, zeta)
    res = zeros(Float64,length(x))
    if σ == 0
        if c == 1
            res = -x .+ (x.+zeta).*log.((x.+zeta)./zeta)
            return res
        elseif c == 2
            res = -1.5.*x.^2 .- x.*zeta .+ (x.+zeta).^2 .* log.((x.+zeta)./zeta)
            return res
        elseif c == 3
            res =-(11.*(x.^3) + 15.*(x.^2).*zeta.+6.*x.*(zeta.^2))./6 .+((x.+zeta).^3) .*log.((x.+zeta)./zeta)
            return res
        else throw("must be c ∈ {1,2,3}")
        end
    elseif 0 < σ < 1
        if c == 1
            res = (-x.*zeta.^σ.+((x.+zeta).^(1+σ).-zeta.^(1+σ))./(1+σ)) ./ σ
            return res
        elseif c == 2
            res = (-x.^2.*zeta.^σ .+ 2.*((x.+zeta).^(2+σ).-zeta.^(1+σ) .*((2+σ).*x.+zeta))./((1+σ).*(2+σ))) ./ σ
            return res
        elseif c == 3
            res = (-(x.^3).*(zeta.^σ) .+3.*(2.*((x.+zeta).^(3+σ)).-(zeta.^(1+σ)) .*((2+σ).*(3+σ).*(x.^2).+2.*(3+σ).*x.*zeta.+2.*(zeta.^2))) ./((1+σ).*(2+σ).*(3+σ))) ./ σ
            return res
        else throw("must be c ∈ {1,2,3}")
        end
    else print("must be σ ∈ (0,1)")
    end
end


# Integral giving the probability of new cluster
function newclust_int(x, c, σ, zeta)
    value = 0.0
    if σ == 0
        if c == 1
            return log((x+zeta)/zeta)
        elseif c == 2
            return 2 * ((zeta+x)*log((x+zeta)/zeta) -x)
        elseif c == 3
            return 3 * ((zeta+x)^2 *log((x+zeta)/zeta) -0.5*x*(3*x+2*zeta))
        else throw("must be c ∈ {1,2,3}")
        end        
    elseif σ < 1
        if c == 1
            return ((zeta+x)^σ -zeta^σ)/σ 
        elseif c == 2
            return 2*((zeta+x)^(σ+1) -zeta^σ * (x+zeta+σ*x))/(σ*(σ+1))
        elseif c == 3
            return 3 * (2*(x+zeta)^(2+σ)-zeta^σ *((1+σ)*(2+σ)*x^2+(4+2*σ)*x*zeta+2*zeta^2))/
            (σ*(σ+1)*(σ+2))
        else throw("must be c ∈ {1,2,3}")
        end
    else throw("must be σ ∈ (0,1)")
    end
    return value
end

# Stratified resampler
function strat_res(p)
    n = length(p)
    vec = zeros(Int64,n)
    cumsum = p[1]
    j = 1
    @inbounds for i in 1:n
        x = (i-1)/n + rand()/n
        flag = 0
        while flag == 0
            if cumsum > x
                vec[i] = j
                flag = 1
            else
                j += 1
                cumsum += p[j]
            end
        end
    end
    return vec
end

function log_taudens(x,
                     taun1,
                     kn1,
                     m,
                     theta,
                     c, σ, zeta)
    if (kn1 > 0)
    	if x > taun1	
            summ = 0.
	    logprod = 0.
	    for i in 1:kn1
		summ += (m[i]-σ) / (x-theta[i]+zeta)  
               	logprod += (σ-m[i]) * log(x-theta[i]+zeta)		
	    end
	    return(logprod + 
		   log(newclust_int(x,c,σ,zeta) + summ) -
		   laplace_int_gamma(x,c,σ,zeta))
	else
            return -Inf
	end
    else
	if x > 0
	    return(log(newclust_int(x,c,σ,zeta)) - laplace_int_gamma(x,c,σ,zeta))
	else return -Inf
	end
    end
end

function log_thetadens(x, taun,
                     c, σ, zeta)
    return log(c) + (c-1)*log(x) - (1-σ)*log(zeta+taun-x)
end

# -------------------------------------------------- #

function proposal_par() #c,sigma,ze; h_c=, h_sigma=, h_ze=)
    new_c = rand([1,2,3])
    new_sigma = rand()
#    new_ze = rand(Exponential(10))    
    return Dict("c" => new_c,
                "sigma" => new_sigma)
#                "ze" => new_ze)
end

function asymp_arrivaltime(n,c,s,z)
    return [((c+1)*i*z^(s-1)) ^(1/(c+1)) for i in 1:n]
end

function sample_arrivaltime(old_tau,at,n)
    # at : sequence of deterministic arrival times
    v = (at[n+1]-at[n])/2.
    tau = rand(SymTriangularDist(at[n],(at[n+1]-at[n])/2))    
    return tau
end

function logmarg(ssph::Array{Float64,1},
                 sumssph::Float64,
                 obs::Array{Int64,1},
                 sumobs::Int64,
                 gamma_const_obs_ind::Float64,
                 V::Int64)
    s = gamma_const_obs_ind
    @inbounds for i in find(obs)
        @inbounds for j in 0:(obs[i]-1)               
            s += log(ssph[i] + j)
        end
    end
    @inbounds for i in 0:(sumobs-1)
        s -= log(sumssph +i) 
    end
    return s
end

function propagate_and_weight(N::Int64,
                               t::Int64,
                               V::Int64,
                               old_tau::Array{Float64,1},
                               tau::Array{Float64,1},
                               at::Array{Float64,1},
                               theta::Array{Float64,1},
                               thetastar::Array{Float64,2},
                               nclust::Array{Int64,1},
                               clust_label::Array{Int64,2},
                               size_clust::Array{Int64,2},
                               obs,sumobs_t,
                               ss, obs_t,
                               lg_clust::Array{Float64,2},
                               gamma_const_obs::Array{Float64,1},
                               gamma_const_obsbe::Array{Float64,1},
                               gamma_const_o::Array{Float64,1},
                               var_tau::Float64,
                               c::Int64,
                               sigma::Float64,
                               ze::Float64, h::Array{Float64,1},
                               w::Array{Float64,1},
                               tmp::Array{Float64,1}) 
    label = 0
    w = zeros(Float64,t+1)
    tmp = zeros(Float64,t)
    log_weights_p = zeros(Float64,N)
    a,b = 0.,0.
    cm1 = c-1
    sm1 = sigma -1
    sum_var_dir = sum(h)
        
    @inbounds for i in 1:N
        tau[i] = rand(SymTriangularDist(at[t],(at[t+1]-at[t])/2)) 
        n = nclust[i]
        for j in 1:n
            ss = sum(obs[:,find(clust_label[1:t-1,i].==j)],2)[:]
            w[j] = exp(logmarg(ss + h,
                               sum(ss)+sum_var_dir,obs_t,
                               sumobs_t,
                               gamma_const_obs[t],V)) *
              (size_clust[j,i]-sigma) / (tau[i]-thetastar[j,i]+ze)
        end
        w[n+1] = exp(gamma_const_obsbe[t]) *newclust_int(tau[i],c,sigma,ze)
        s = sum(w[1:n+1])
        if s == 0
            println("WARNING: sum of weights in cluster assignment is 0")
            label = rand(1:n+1)
        else
            label = rand(Categorical(w[1:n+1]/s)) 
        end
        clust_label[t,i] = label
        
        tmp[1:nclust[i]] = (size_clust[1:nclust[i],i].-sigma) .*
                           log.((ze.+old_tau[i].-thetastar[1:nclust[i],i]) ./
                           (ze.+tau[i].-thetastar[1:nclust[i],i]))              
        if label == n+1
            # Propagate particle
            theta[i] = tau[i] * rand()
            thetastar[n+1,i] = theta[i]
            log_weights_p[i] = sum(tmp) + log(s) + log(tau[i])+
              -logpdf(SymTriangularDist(at[t],(at[t+1]-at[t])/2),tau[i]) 
            nclust[i] += 1
            size_clust[n+1,i] = 1
        else
            # Propagate particle
            theta[i] = thetastar[label,i]
            # Compute log_weight of the particle
            log_weights_p[i] = sum(tmp) + log(s) +
              -logpdf(SymTriangularDist(at[t],(at[t+1]-at[t])/2),tau[i]) 
            size_clust[label,i] += 1
        end
    end
    log_weights_p += laplace_int_gamma_vec(old_tau,c,sigma,ze) - laplace_int_gamma_vec(tau,c,sigma,ze)
    w = 0
    tmp = 0
    return log_weights_p
end
    
#--------------------------------------------------#

function generate_partition(n,k,V,var_be,c,sigma,ze)
    # Requires JLD, Distributions, nexch_rndpart()
    # n: number of obs
    # k: number of words per doc
    # V: vocabulary size
    obs = zeros(Int64,V,n)
    part = nexch_rndpart(n,c,sigma,ze)
    nclust = maximum(part)
    be = zeros(Float64,V,nclust)
    for i in 1:nclust
        be[:,i] = rand(Dirichlet(var_be*ones(V)))
    end
    for i in 1:n
        obs[:,i] = rand(Multinomial(k,be[:,part[i]]))
    end
    save("synthdata_c$(c)s$(sigma)ze$(ze)var$(var_be)V$(V)k$(k).jld",
         "csz", [c,sigma,ze],
         "V", V,
         "n", n,
         "k", k,
         "var_be", var_be,
         "be", be,
         "obs", obs,
         "part", part,
         "nclust", nclust)
end

