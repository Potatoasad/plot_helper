import numpy as np
import pandas as pd

HAS_BEEN_EVALED = False

eval_string = """
using Pkg;
Pkg.add(["Distributions", "TruncatedGaussianMixtures", "DataFrames", "ProgressMeter", "StatsBase"])
using Distributions
using TruncatedGaussianMixtures
using DataFrames
using ProgressMeter
using StatsBase

make_trunc_norm(μ1, σ1, μ2, σ2, ρ; a=[0.0, 0.0], b=[1.0, 1.0]) = TruncatedMvNormal(MvNormal([μ1, μ2], [σ1^2 σ1*σ2*ρ; σ1*σ2*ρ σ2^2]), a, b);
function get_correlation(μ1, σ1, μ2, σ2, ρ; a=[0.0, 0.0], b=[1.0, 1.0])
    M1, M2 = TruncatedGaussianMixtures.moments(make_trunc_norm(μ1, σ1, μ2, σ2, ρ; a=a, b=b));
    Σ = M2 - M1 * M1';
    Σ[1,2]/sqrt(Σ[1,1]*Σ[2,2])
end

function get_correlation(μ1::AbstractVector, σ1::AbstractVector, μ2::AbstractVector, σ2::AbstractVector, ρ::AbstractVector; a=[0.0, 0.0], b=[1.0, 1.0])
    corr_true = zeros(length(μ1))
    @showprogress for i in eachindex(corr_true)
        corr_true[i] = get_correlation(μ1[i], σ1[i], μ2[i], σ2[i], ρ[i]; a=a, b=b)
    end
    return corr_true
end

function get_true_correlations(df; cols=[:mu_chi_1_at_0, :sigma_chi_1_at_0, :mu_chi_2_at_0, :sigma_chi_2_at_0, :rho_chi_1], a=[0.0, 0.0], b=[1.0, 1.0])
    df_prior = DataFrame();
    N = nrow(df);
    for col in cols
        if occursin("rho", string(col))
            df_prior[!, col] = 2 .* rand(N) .- 1
        else
            df_prior[!, col] = rand(N);
        end
    end

    corrs = zeros(N)
    corrs_prior = zeros(N)
    @showprogress for (i,x) in enumerate(eachrow(df))
        corrs[i] = get_correlation([df[i, col] for col in cols]...; a=a, b=b)
        corrs_prior[i] = get_correlation([df_prior[i, col] for col in cols]...; a=a,b=b)
    end
    df[!, :correlation_true] = corrs;
    df_prior[! ,:correlation_true] = corrs_prior;

    df, df_prior
end

(x, cols) -> get_true_correlations(DataFrame(x); cols=Symbol.(cols), a=[0.0, 0.0], b=[1.0, 1.0])
"""

def get_correlations(mu1, sig1, mu2, sig2, rho, a=[0.0, 0.0], b=[1.0, 1.0]):
    if not HAS_BEEN_EVALED:
        print("Importing truncatedgaussianmixtures, install it if you want to use this function")
        from truncatedgaussianmixtures import jl
        testfunc = jl.seval(eval_string)
        HAS_BEEN_EVALED = True


    a = juliacall.convert(juliacall.Main.Array, np.array(a));
    b = juliacall.convert(juliacall.Main.Array, np.array(b));
    corrs = juliacall.Main.get_correlation(juliacall.convert(juliacall.Main.Array, mu1), juliacall.convert(juliacall.Main.Array, sig1), 
                                           juliacall.convert(juliacall.Main.Array, mu2), juliacall.convert(juliacall.Main.Array, sig2), 
                                           juliacall.convert(juliacall.Main.Array, rho), a=a, b=b)
    return corrs