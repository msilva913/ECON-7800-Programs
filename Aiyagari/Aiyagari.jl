using PyPlot
using Parameters, CSV, Random, QuantEcon
using LinearAlgebra, LinearInterpolations
using BenchmarkTools
using DataFrames
using Printf
cd(@__DIR__)
include("Aiyagari_functions.jl")



para = Para(b=0.0, NS=7)
@unpack b, grid_max, NA, NS, σ, ρ, R, β, a, P = para

#Cluster points near constraint
@show a
fig, ax = plt.subplots()
ax.hist(a)
display(fig)

r, w, phi, asset_probs, C, K_supply, CV_C, CV_K, a_pol, c_pol, para =  general_equilibrium(para)

ind = a.<50.0
" Plot histogram "
fig, ax = subplots(ncols=1, figsize=(6, 6))
ax.plot(a[ind], asset_probs[ind], label="asset distribution")
ax.fill_between(a[ind], asset_probs[ind], 0.0, alpha=0.2)
ax.legend()
tight_layout()
plt.savefig("asset_distribution.pdf")
display(fig)

# Mass at constraint
@printf "%0.2f%% agents constrained  " sum(100*asset_probs[a.<=b])

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
plt.savefig("policies.pdf")
display(fig)

#r, w, phi, asset_probs, C, K, CV_C, CV_K = main(para)
#CV_c_vals, K_vals, r_vals = main_general()
rho_vals = [0.0, 0.3, 0.6, 0.9]
para = Para(b=0.0, NS=7)
CV_c_vals, K_vals, r_vals, t = generate_stats_table(rho_vals, 0.4, para)
print(t)




