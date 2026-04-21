# ============================================================
# Aiyagari_v2.jl  —  Main script (Plots.jl backend)
# Depends on: Aiyagari_functions.jl
# ============================================================

using Plots
using Parameters, CSV, Random, QuantEcon
using LinearAlgebra, LinearInterpolations
using DataFrames
using Printf

gr()   # GR backend: fast, PDF/PNG capable, no Python dependency

cd(@__DIR__)
include("Aiyagari_functions.jl")

# ── Colour palette ────────────────────────────────────────────────────────────
C_MAIN = "#2c7bb6"
C_ACC  = "#d7191c"
C_MED  = "#1a9641"

# ============================================================
# EXERCISE 0  Baseline GE solution
# ============================================================
println("\n=== Solving baseline GE equilibrium ===")
para = Para(b=0.0, NS=7)
r, w, phi, asset_probs, C, K, CV_C, CV_K, a_pol, c_pol, para =
    general_equilibrium(para; use_egm=true)

@printf "\nBaseline equilibrium\n"
@printf "  r* = %.4f%%   K* = %.4f   C* = %.4f\n"  r*100 K C
@printf "  CV(c) = %.2f%%   CV(w) = %.2f%%\n"       CV_C CV_K
@printf "  Liquidity premium = %.4f%%\n"             ((1-para.β)/para.β - r)*100

# ============================================================
# EXERCISE 1  Asset distribution
# ============================================================
ind = para.a .< 40.0

p1 = Plots.plot(para.a[ind], asset_probs[ind],
          color=C_MAIN, lw=1.8, label="Asset distribution",
          xlabel="Assets", ylabel="Density",
          title="Stationary Asset Distribution",
          fillrange=0, fillalpha=0.18, fillcolor=C_MAIN)
vline!(p1, [para.b], color=C_ACC, ls=:dash, lw=1.2,
       label="Borrowing constraint")
Plots.savefig(p1, "fig1_asset_distribution.pdf")
display(p1)

pct_constrained = sum(asset_probs[para.a .<= para.b]) * 100
@printf "  %.2f%% of agents at borrowing constraint\n" pct_constrained

# ============================================================
# EXERCISE 2  Policy functions
# ============================================================
blues = cgrad(:Blues)
colors_z = [blues[x] for x in range(0.35, 0.95; length=para.NS)]

pc = Plots.plot(xlabel="Assets", ylabel="Consumption",
          title="Consumption policy c(a,z)", legend=:topleft)
ps = Plots.plot(xlabel="Assets", ylabel="Next-period assets",
          title="Savings policy a'(a,z)", legend=:topleft)

for z_i in 1:para.NS
    plot!(pc, para.a[ind], c_pol[ind, z_i],
          color=colors_z[z_i], lw=1.5, label="z=$z_i")
    plot!(ps, para.a[ind], a_pol[ind, z_i],
          color=colors_z[z_i], lw=1.5, label="z=$z_i")
end
plot!(ps, para.a[ind], para.a[ind],
      color=:black, ls=:dash, lw=0.8, alpha=0.5, label="45°")

fig2 = Plots.plot(pc, ps, layout=(1,2), size=(900,380))
Plots.savefig(fig2, "fig2_policy_functions.pdf")
Plots.display(fig2)

# ============================================================
# EXERCISE 3  MPC distribution
# ============================================================
println("\n=== Computing MPC distribution ===")
mpc     = compute_mpc(c_pol, para)
avg_mpc = sum(mpc .* phi)
@printf "  Aggregate (wealth-weighted) MPC = %.3f\n" avg_mpc

z_mid = ceil(Int, para.NS / 2)

pm1 = Plots.plot(xlabel="Assets", ylabel="MPC",
           title="MPC by asset level and income state",
           ylim=(-0.05, 1.1), legend=:topright)
Plots.plot!(pm1, para.a[ind], mpc[ind, 1],     color=C_ACC,  lw=1.8, label="z = lowest")
Plots.plot!(pm1, para.a[ind], mpc[ind, z_mid], color=C_MAIN, lw=1.8, label="z = median")
Plots.plot!(pm1, para.a[ind], mpc[ind, end],   color=C_MED,  lw=1.8, label="z = highest")
Plots.hline!(pm1, [avg_mpc], color=:black, ls=:dash, lw=1.0,
       label=@sprintf("Agg. MPC = %.2f", avg_mpc))

all_mpc  = vec(mpc)
all_wts  = vec(phi)
mpc_bins = range(0, 1; length=25)
mpc_hist = [sum(all_wts[(all_mpc .>= mpc_bins[i]) .& (all_mpc .< mpc_bins[i+1])])
            for i in 1:(length(mpc_bins)-1)]
bin_mids = collect(0.5 .* (mpc_bins[1:end-1] .+ mpc_bins[2:end]))
pm2 = Plots.bar(bin_mids, mpc_hist,
          bar_width=step(mpc_bins)*0.85, color=C_MAIN, alpha=0.7,
          xlabel="MPC", ylabel="Population share",
          title="Distribution of MPC across agents", legend=false)

fig3 = Plots.plot(pm1, pm2, layout=(1,2), size=(900,380))
Plots.savefig(fig3, "fig3_mpc_distribution.pdf")
Plots.display(fig3)

# ============================================================
# EXERCISE 4  Lorenz curves and Gini
# ============================================================
println("\n=== Wealth distribution statistics ===")

gini_w   = compute_gini(para.a, asset_probs)
y_flat   = vec(repeat(para.y', para.NA, 1))
phi_flat = vec(phi)
c_flat   = vec(c_pol)
gini_y   = compute_gini(y_flat,  phi_flat)
gini_c   = compute_gini(c_flat,  phi_flat)

@printf "  Gini wealth      = %.3f\n" gini_w
@printf "  Gini income      = %.3f\n" gini_y
@printf "  Gini consumption = %.3f\n" gini_c

s1, s10, s50 = compute_wealth_shares(para.a, asset_probs)
@printf "  Top 1%%  wealth share = %.1f%%\n" s1*100
@printf "  Top 10%% wealth share = %.1f%%\n" s10*100
@printf "  Bot 50%% wealth share = %.1f%%\n" s50*100

cw, lw_c = lorenz_curve(para.a, asset_probs)
cy, ly   = lorenz_curve(y_flat,  phi_flat)
cc, lc   = lorenz_curve(c_flat,  phi_flat)

fig4 = Plots.plot(cw, lw_c, color=C_ACC,  lw=2, label=@sprintf("Wealth (Gini=%.2f)", gini_w),
            xlabel="Cumulative population share",
            ylabel="Cumulative value share",
            title="Lorenz Curves: Wealth > Income > Consumption",
            size=(500,500))
Plots.plot!(fig4, cy, ly, color=C_MAIN, lw=2, label=@sprintf("Income (Gini=%.2f)", gini_y))
Plots.plot!(fig4, cc, lc, color=C_MED,  lw=2, label=@sprintf("Consumption (Gini=%.2f)", gini_c))
Plots.plot!(fig4, [0,1],[0,1], color=:black, ls=:dash, lw=0.8, label="Perfect equality")
Plots.savefig(fig4, "fig4_lorenz.pdf")
Plots.display(fig4)

# ============================================================
# EXERCISE 5  Precautionary savings premium vs sigma and rho
# ============================================================
println("\n=== Precautionary savings premium ===")

sigma_vals = [0.2, 0.4]
rho_vals   = [0.0, 0.3, 0.6, 0.9]
beta_rate  = (1 - para.β) / para.β * 100
r_matrix   = zeros(length(sigma_vals), length(rho_vals))

for (i, sv) in enumerate(sigma_vals), (j, rv) in enumerate(rho_vals)
    p_tmp = Para(b=0.0, NS=7, σ=sv, ρ=rv)
    r_eq, = general_equilibrium(p_tmp; use_egm=true, verbose=false)
    r_matrix[i,j] = r_eq * 100
    @printf "  sigma=%.1f  rho=%.1f  r*=%.3f%%  premium=%.3f%%\n" sv rv r_eq*100 (beta_rate-r_eq*100)
end

fig5 = Plots.plot(xlabel="Income persistence rho",
            ylabel="Liquidity premium (pp)",
            title="Precautionary Savings Premium by Income Risk",
            size=(600,380))
for (i, sv) in enumerate(sigma_vals)
    Plots.plot!(fig5, rho_vals, beta_rate .- r_matrix[i,:],
          marker=:circle, lw=2,
          color=(i==1 ? C_MAIN : C_ACC),
          label="sigma = $sv")
end
Plots.savefig(fig5, "fig5_precautionary_premium.pdf")
Plots.display(fig5)

# ============================================================
# EXERCISE 6  Aiyagari (1994) Table I — extended with Gini
# ============================================================
println("\n=== Table: GE statistics across rho (sigma = 0.4) ===")
para_tab = Para(b=0.0, NS=7)
_, _, _, t = generate_stats_table(rho_vals, 0.4, para_tab; use_egm=true)
println(t)
CSV.write("aiyagari_table.csv", t)

# ============================================================
# EXERCISE 7  Borrowing limit and social insurance
# ============================================================
println("\n=== Borrowing limit comparative statics ===")
b_vals = [0.0, 0.5, 1.0, 2.0]
r_b = Float64[]; K_b = Float64[]; g_b = Float64[]

for bv in b_vals
    p_tmp = Para(b=bv, NS=7)
    r_eq, _, _, ap_eq, _, K_eq, _, _, _, _, p_out =
        general_equilibrium(p_tmp; use_egm=true, verbose=false)
    g = compute_gini(p_out.a, ap_eq)
    push!(r_b, r_eq*100); push!(K_b, K_eq); push!(g_b, g)
    @printf "  b=%.1f  r*=%.3f%%  K*=%.3f  Gini=%.3f\n" bv r_eq*100 K_eq g
end

pb1 = Plots.plot(b_vals, r_b, marker=:circle, color=C_MAIN, lw=2, legend=false,
           xlabel="Borrowing limit b", ylabel="r* (%)",
           title="Equilibrium interest rate")
pb2 = Plots.plot(b_vals, K_b, marker=:circle, color=C_ACC,  lw=2, legend=false,
           xlabel="Borrowing limit b", ylabel="K*",
           title="Aggregate capital")
pb3 = Plots.plot(b_vals, g_b, marker=:circle, color=C_MED,  lw=2, legend=false,
           xlabel="Borrowing limit b", ylabel="Gini (wealth)",
           title="Wealth inequality")
fig6 = Plots.plot(pb1, pb2, pb3, layout=(1,3), size=(1050,600),
            plot_title="Effect of borrowing limit (social insurance proxy)")
Plots.savefig(fig6, "fig6_borrowing_limit.pdf")
Plots.display(fig6)

# ============================================================
# EXERCISE 8  Welfare cost of business cycles — Lucas vs Aiyagari
# ============================================================
println("\n=== Welfare costs of business cycles ===")
gamma_p  = para.γ
cost_agg = welfare_cost_business_cycles(CV_C / 100, gamma_p)

bot10     = cumsum(asset_probs) .<= 0.10
c_b10     = c_pol[bot10, :]
p_b10     = phi[bot10, :] ./ sum(phi[bot10, :])
C_b10     = sum(c_b10 .* p_b10)
sig_b10   = sqrt(sum((c_b10 .- C_b10).^2 .* p_b10)) / C_b10
cost_b10  = welfare_cost_business_cycles(sig_b10, gamma_p)

@printf "\n  Lucas (aggregate)  welfare cost = %.4f%% of consumption\n" cost_agg*100
@printf "  Bottom-10%%         welfare cost = %.4f%% of consumption\n"  cost_b10*100
@printf "  Ratio bottom / aggregate        = %.1fx\n" cost_b10 / max(cost_agg, 1e-10)

println("\n=== All figures saved. Run complete. ===")