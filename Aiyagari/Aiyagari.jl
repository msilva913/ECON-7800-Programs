using PyPlot
using LaTeXStrings, KernelDensity
using Parameters, CSV, Random, QuantEcon
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim, LinearInterpolations
using BenchmarkTools
using DataFrames, KernelDensity
using Printf

include("Aiyagari_functions.jl")



para = Para(b=0.0, NS=7)
@unpack b, grid_max, NA, NS, σ, ρ, R, β, a, P = para

r, w, phi, asset_probs, C, K_supply, CV_C, CV_K, a_pol, c_pol, para =  general_equilibrium(para)

ind = a.<30.0
" Plot histogram "
fig, ax = subplots(ncols=1, figsize=(6, 6))
ax.plot(a[ind], asset_probs[ind], label="asset distribution")
ax.fill_between(a[ind], asset_probs[ind], 0.0, alpha=0.2)
ax.legend()
tight_layout()
display(fig)


# Policies
fig, ax = subplots(ncols=2, figsize=(12, 6))
ax[1].set_title("Consumption policy")
ax[2].set_title("Savings policy")
    for z_i in eachindex(para.y)
        ax[1].plot(a[ind], c_pol[ind, z_i], label="z=$z_i")
        ax[2].plot(a[ind], a_pol[ind, z_i], label="z=$z_i")
    end
ax[1].legend()
ax[2].legend()
tight_layout()
display(fig)

#r, w, phi, asset_probs, C, K, CV_C, CV_K = main(para)
#CV_c_vals, K_vals, r_vals = main_general()
rho_vals = [0.0, 0.3, 0.6, 0.9]
para = Para(b=0.0, NS=7)
CV_c_vals, K_vals, r_vals, t = generate_table(rho_vals, 0.4, para)
print(t)




