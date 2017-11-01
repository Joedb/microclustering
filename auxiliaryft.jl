# --------------------------------------------------------------------------------------------------
# Copyright (C) Giuseppe Di Benedetto, University of Oxford
# benedett@stats.ox.ac.uk
# October 2017
# --------------------------------------------------------------------------------------------------

function size_distr(part)
    n = length(part)
    n_clust = maximum(part)
    index = n_clust
    clust_sizes = zeros(Int64, n_clust)
    freqs = zeros(Float64, n)
    for i in 1:n_clust
        clust_sizes[i] = sum(part .== i)
    end
    max_clust_size = maximum(clust_sizes)
    for i in 1:n
        freqs[i] = sum(clust_sizes .== i) / n_clust
    end
    n_singl =  sum(clust_sizes .== 1)
    mean_size = mean(clust_sizes)
    for i in collect(n_clust:-1:1)
        if freqs[i] > 0
            index = i
            break
        end
    end
    return (n, n_clust, n_singl, mean_size, clust_sizes, freqs, max_clust_size)
end

function clustsize_asymp(part,
                         name;
                         p=0, minsize = 1, maxsize = 0)
    n = length(part)
    nclust = maximum(part)
    if maxsize == 0
        maxsize = n
    end
    growth = zeros(Int64, nclust, n)
    for i in 1:n
        if i > 1
            growth[:, i] = copy(growth[: ,i-1])
        end
        growth[part[i],i] += 1
    end
    if p ==1
        PyPlot.plot()
        #PyPlot.xticks(fontsize=15)
        #PyPlot.yticks(fontsize=15)
        PyPlot.xlabel("sample size")
        PyPlot.ylabel("size of the clusters",
                      rotation = 90)
        for i in 1:nclust
            if growth[i,end] >= minsize && growth[i,end] <= maxsize
                PyPlot.plot(1:n, collect(growth[i,:]))
            end
        end
        PyPlot.savefig("$(name)_clustersize.pdf")
        PyPlot.close()
    end
    return growth
end

function histg(x, edgepoints)
    k = length(edgepoints)
    n = length(x)
    h = Array{Int64}(k)
    for i in 1:(k-1)
        h[i] = sum(edgepoints[i].<=x.<edgepoints[i+1])
    end
    h[k] = sum(x.==edgepoints[end])
    return h
end

function freqplots(data, ne;
                   n=10000, m=100, name="")
    edgebins = 0:1:12
    edgebins = (2.^edgebins) #.-0.01
    sizebins = edgebins[2:end] - edgebins[1:end-1]
    sizebins = vcat(sizebins,1)
    centerbins = edgebins
    k = maximum(data)
    size = zeros(k)
    for j in 1:k
        size[j] = sum(data.==j)
    end
    counts = histg(size, edgebins)
    fr_data = counts ./ sizebins / k
    fr = zeros(m,13)
    fr_low = zeros(13)
    fr_up = zeros(13)
    PyPlot.plot()
    #PyPlot.xticks(fontsize=10)
    #PyPlot.yticks(fontsize=10)
    PyPlot.xscale("log")
    PyPlot.yscale("log")
    PyPlot.xlim(0.9,10^2.1)

    PyPlot.xlabel(L"$r$",
                  fontsize = 20,
                  labelpad = 7)
    PyPlot.ylabel(L"$\frac{K_{n,r}}{K_n}$",
                  rotation = 0,
                  fontsize = 20,
                  labelpad = 15)
    PyPlot.subplots_adjust(left=0.15)
    PyPlot.subplots_adjust(bottom=0.15)
    #PyPlot.subplots_adjust(top=0.93)
    #PyPlot.subplots_adjust(right=0.95)
    if ne == 0            
        PyPlot.plot(centerbins[1:end],fr_data, "o",
                    color = "red",
                    alpha = .8,
                    ms = 8,
                    label = "data")
        PyPlot.legend(fontsize=15,numpoints=1)    
        PyPlot.savefig("freqs_$(name).pdf")
    else            
        for i in 1:m
            k = maximum(ne[:,i])
            size = zeros(k)
            for j in 1:k
                size[j] = sum(ne[:,i].==j)
            end
            counts = histg(size, edgebins)
            fr[i,:] = counts ./ sizebins / k
        end
        for i in 1:12
            fr_low[i] = quantile(collect(fr[:,i]),0.025)
            fr_up[i] = quantile(collect(fr[:,i]),0.975)
        end
        stop_fr = findfirst(fr_low,0)
        PyPlot.fill_between(centerbins[1:(stop_fr-1)],
                            fr_low[1:(stop_fr-1)], fr_up[1:(stop_fr-1)],
                            alpha = 0.4,
                            color = "blue",
                            label = "95% CI")
        PyPlot.plot(centerbins[1:end],fr_data, "o",
                    color = "red",
                    alpha = .8,
                    ms = 8,
                    label = "data")
        PyPlot.legend(fontsize=15,numpoints=1)
        PyPlot.savefig("freqs_$(name).pdf")
    end
    PyPlot.close()
end

function smc_plot(MLE, mean_vals, sd_vals, sigma_vals)
    cols = ["blue","green","purple"]
    PyPlot.figure(figsize=(15,12))
    PyPlot.xticks(fontsize=20)
    PyPlot.yticks(fontsize=20)
    PyPlot.xlabel(L"$\sigma$",
                  fontsize = 35,
                  labelpad = 5)
    PyPlot.ylabel("Loglikelihood",
                  fontsize = 25,
                  labelpad = 15)
    PyPlot.xlim(0,1)
    for i in 1:3
        PyPlot.errorbar(sigma_vals,collect(mean_vals[i,:]),
                        yerr = collect(sd_vals[i,:]),
                        color = cols[i],
                        label=L"$\xi=$""$(i)",
                        linewidth = 2)
    end
    PyPlot.plot(MLE[1],MLE[2],"o",
                ms = 8,
                color = "red",
                label="ML")
    PyPlot.legend(loc=3,fontsize=25,numpoints=1)
    PyPlot.grid("on")
    PyPlot.savefig("loglikelihood_smc_estim.pdf")
    PyPlot.close()
end

function prediction(n_tot, train_data,num_particles,xi_max, sigma_max, zeta)
    res = smc(train_data, num_particles, xi_max, sigma_max, zeta)
    time = res[10][1]
    smc_samples_theta = collect(res[9])
    pred_part = posterior_CRM(time, n_tot, train_data, smc_samples_theta,
                              xi_max, sigma_max, zeta)
    return relabel(pred_part[(length(train_data)+1):n_tot])
end
