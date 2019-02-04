using Distributions, Devectorize, JLD, Plots, LaTeXStrings
include("adapt_thin_GGP.jl")
include("crp.jl")

#Function to plot the results for the 3 values of the parameter c
function plotfromfile(fname, ngrid, nrun, length_obs, num_particles;
                      grind_int=collect(1:ngrid), p=0, name="")
    loglike = zeros(Float64, 3, ngrid, nrun)
    m       = zeros(Float64, 3, ngrid)
    sd      = zeros(Float64, 3, ngrid)
    j       = zeros(Int64, 3, ngrid)
    f       = open(fname)
    s_grid  = collect(linspace(0,.99,ngrid))[grid_int]
    lgrid   = length(grid_int)
    tot     = lgrid * nrun * 3
    for i in 1:(tot)
        println(i)
        c = parse(Int64,readline(f)[3:(end)])
        s = parse(Float64,readline(f)[3:(end)])
        s_index = findfirst(s_grid,s)
        j[c, s_index] += 1
        loglike[c, s_index, j[c, s_index]] +=
            parse(Float64,readline(f)[3:(end)])
    end
    if sum(j) != tot
        println(j)
        throw("There are some data missing!")
    end

    close(f)
    ML = [0, 0]
    for i in 1:3
        for j in 1:ngrid
            m[i,j]  = mean(collect(loglike[i,j,:]))
            sd[i,j] = std(collect(loglike[i,j,:]))
        end
        if i == 1
            ML = [1, indmax(collect(m[1,1:end]))]
        else
            tmp = findmax(m[i,:])[1]
            if tmp > m[ML[1],ML[2]]
                ML = [i, indmax(collect(m[i,1:end]))]
            end
        end
    end
    println("ML: c=$(ML[1]), sigma=$(s_grid[ML[2]])")
    MLest = [s_grid[ML[2]], m[ML[1],ML[2]]]
    println(MLest[2])

    if p ==1
        Plots.plot(s_grid, collect(m[1,:]),
                   fmt       = :svg,
                   line      = 1.2,
                   xticks    = s_grid,
                   xlabel    = "sigma",
                   ylabel    = "loglikelihood",
                   xrotation = rad2deg(pi/3),
                   yerr      = (collect(sd[1,:]), collect(sd[1,:])), 
                   label     = "c = 1",
                   title     = "smc estimate on $name data")
        Plots.plot!(s_grid, collect(m[2,:]),
                    line  = 1.2,
                    yerr  = (collect(sd[2,:]), collect(sd[2,:])), 
                    label = "c = 2")
        Plots.plot!(s_grid, collect(m[3,:]),
                    line  = 1.2,
                    yerr  = (collect(sd[3,:]), collect(sd[3,:])), 
                    label = "c = 3")                    )
        Plots.scatter!([MLest[1]], [MLest[2]],
                       markersize        = 3,
                       markerstrokewidth = 1,
                       markerstrokecolor = :red,
                       label             = "ML")
        savefig("$(name)_$(length_obs)obs_$(num_particles)p_$(nrun)runs.svg")
    end
    return(MLest[2],ML[1],s_grid[ML[2]])
end

function clustsize_asymp(part,
                         name;
                         p=0, mass=0, minsize = 1, maxsize = 0)
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
        Plots.plot(fmt  = :pdf,
             label      = "",
             tickfont   = font(15),
             legendfont = font(15),
             guidefont  = font(15),
             xlabel     = "sample size",
             ylabel     = "size of the clusters")
        for i in 1:nclust
            if growth[i,end] >= minsize && growth[i,end] <= maxsize
                Plots.plot!(1:n, collect(growth[i,:]),
                      line = 1,
                      label = "")
            end
        end
        Plots.savefig("$(name)_clustersize.pdf")
        if mass != 0
            for i in 1:nclust
                if growth[i,n] > 3
                    Plots.plot(1:n, collect(growth[i,:]),
                         line = 1,
                         xlabel = "number of points",
                         ylabel = "cluster size",
                         fmt = :pdf,
                         label  = "cluster's size")
                    Plots.plot!(1:n, log(1:n) * mass[i],
                          line  = 2,
                          label = "theoretical rate")
                    Plots.savefig("p/c$(c)_s$(s)_g$(g)_clustersizerate$(i).pdf")
                end
            end
        end
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
    edgebins   = 0:1:12
    edgebins   = (2.^edgebins)
    sizebins   = edgebins[2:end] - edgebins[1:end-1]
    sizebins   = vcat(sizebins,1)
    centerbins = edgebins
    k          = maximum(data)
    size       = zeros(k)
    for j in 1:k
        size[j] = sum(data.==j)
    end
    counts = histg(size, edgebins)
    fr_data = counts ./ sizebins / k
    fr = zeros(m,13)
    fr_low = zeros(13)
    fr_up = zeros(13)
    PyPlot.plot()
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
    if ne == 0            
        PyPlot.plot(centerbins[1:end],fr_data, "o",
                    color = "red",
                    alpha = .8,
                    ms    = 8,
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
            fr_up[i]  = quantile(collect(fr[:,i]),0.975)
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
                    ms    = 8,
                    label = "data")
        PyPlot.legend(fontsize=15,numpoints=1)
        PyPlot.savefig("freqs_$(name).pdf")
    end
    PyPlot.close()
end


# --------------------------------------------------
#   Prediction
# --------------------------------------------------

function predict_cmp(true_p, pred, train_n)
    # true_p
    st       = 0.
    n        = length(true_p)
    n_sample = size(pred)[2]
    err      = zeros(Float64, n_sample)
    if n <= train_n
        throw("train_n > train_n + test_n")
    end
    for j in 1:n_sample
        for i in 1:train_n
            st = sum(true_p[train_n+1:n].==true_p[i])       
            err[j] += (st - sum(pred[train_n+1:n,j].==pred[i,j]))^2
        end
    end
    err = err ./ (n-train_n)
    return Dict("mean_err"  => mean(err),
                "90%CI_err" => [quantile(err,.05), quantile(err,.95)])
end


function ne_predict(data, tot_n, pars; add_t=50)
    # pars  : c,s,z
    # data  : Dictionary of partiotions and fixed atoms
    # tot_n : number of points in the partition: train_n + test_n
    fa       = data["fixed_atoms"]
    p        = data["part"]
    t        = data["t"]
    n_sample = length(p)
    p_pred   = Array{Int64}(tot_n,n_sample)
    for i in 1:n_sample
        println("$i of $(n_sample)")
        p_pred[:,i] = posterior_CRM(t[i],tot_n,p[i],fa[i],
                                    pars[1],pars[2],pars[3],
                                    stop_time=t[i]+add_t)
    end
    return p_pred
end

