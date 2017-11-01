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

function log_two_norm_density(x,μ,v)
    return log(2) - ((x-μ)^2 / (2*v^2)) - log(v*√(2*π)) 
end

function log_two_norm_density_vec(n_part,x,μ,v)
    res = zeros(Float64,n_part)
    @inbounds @simd for i in 1:n_part
        res[i] = log(2) - ((x[i]-μ[i])^2 / (2*v^2)) - log(v*√(2*π))
    end
    return res 
end

function log_thetadens(x, taun,
                     c, σ, zeta)
    return log(c) + (c-1)*log(x) - (1-σ)*log(zeta+taun-x)
end

function log_weight_oldclust_vec(n_part,
                                 kn1,
                                 m,
                                 pos,
                                 tau,
                                 old_tau,
                                 theta,
                                 v,
                                 c, σ, zeta)
    res = zeros(Float64,n_part)
    tmp = zeros(Float64,kn1)
    @inbounds for i in 1:n_part
        tmp = (m[1:kn1].-σ) .*
        log.((zeta.+old_tau[i].-theta[1:kn1,i])./(zeta.+tau[i].-theta[1:kn1,i]))
        res[i] = sum(tmp)
        res[i] += log((m[pos]-σ)/(zeta+tau[i]-theta[pos,i])) - log(2/(v*√(2*π))) +
            ((tau[i]-old_tau[i])^2 / (2*v^2))
    end
    res = res + laplace_int_gamma_vec(old_tau,c,σ,zeta) - laplace_int_gamma_vec(tau,c,σ,zeta)
    return res
end

function log_weight_newclust_vec(n_part,
                                 kn1,
                                 m,
                                 tau,
                                 old_tau,
                                 theta,
                                 v,
                                 c, σ, zeta)
    res = zeros(Float64,n_part)
    tmp = zeros(Float64,kn1)
    @inbounds for i in 1:n_part
        tmp = (m[1:kn1].-σ) .*
            log.((zeta.+old_tau[i].-theta[1:kn1,i])./(zeta.+tau[i].-theta[1:kn1,i]))
        res[i] = sum(tmp) 
        res[i] += (σ-1)*log(zeta+tau[i]-theta[kn1+1,i]) +
            log(c*(theta[kn1+1,i]^(c-1)))- log(2/(v*√(2*π))) +
            ((tau[i]-old_tau[i])^2/(2*v^2)) +log(tau[i]) 
    end
    res = res + laplace_int_gamma_vec(old_tau,c,σ,zeta) - laplace_int_gamma_vec(tau,c,σ,zeta)
    return res
end
