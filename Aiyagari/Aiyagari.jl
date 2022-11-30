using PyPlot
<<<<<<< HEAD
using LaTeXStrings
using Parameters, CSV, Random, QuantEcon
using NLsolve, Dierckx
using LinearAlgebra, Roots, Optim, LinearInterpolations
=======
using Parameters, CSV, Random, QuantEcon
using LinearAlgebra, LinearInterpolations
>>>>>>> 6c74576fc64480f68e55b33b5f5d1b0dc951b5cd
using BenchmarkTools
using DataFrames
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
<<<<<<< HEAD
tight_layout()
plt.savefig("asset_distribution")
=======
plt.savefig("asset_distribution.pdf")
tight_layout()
>>>>>>> 6c74576fc64480f68e55b33b5f5d1b0dc951b5cd
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
<<<<<<< HEAD
tight_layout()
plt.savefig("policies.pdf")
=======
plt.savefig("policies.pdf")
tight_layout()
>>>>>>> 6c74576fc64480f68e55b33b5f5d1b0dc951b5cd
display(fig)

#r, w, phi, asset_probs, C, K, CV_C, CV_K = main(para)
#CV_c_vals, K_vals, r_vals = main_general()
rho_vals = [0.0, 0.3, 0.6, 0.9]
para = Para(b=0.0, NS=7)
CV_c_vals, K_vals, r_vals, t = generate_table(rho_vals, 0.4, para)
print(t)




