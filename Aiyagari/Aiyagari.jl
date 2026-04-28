# ============================================================
# Aiyagari.jl  —  Main script (Plots.jl backend)
# Depends on: Aiyagari_functions.jl
# ============================================================

using Plots
using Parameters, CSV, Random, QuantEcon
using LinearAlgebra, LinearInterpolations
using DataFrames
using Printf
using LaTeXStrings
using Distributions

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
para = Para(b=0.0, NS=11)
# ========================================
y        = para.mc.state_values                     # 11 evenly-spaced log-z nodes
NS       = para.NS                                  # 11
σ_y      = para.σ                                   # unconditional std = 0.2
Δy       = y[2] - y[1]                             # uniform grid spacing

# Binomial(NS-1, 0.5) masses converted to density: divide by Δy
bin_dens = [pdf(Binomial(NS - 1, 0.5), k) for k in 0:NS-1] ./ Δy

# Normal density on fine grid
y_fine   = range(y[1] - 2Δy, y[end] + 2Δy; length = 400)
norm_dens = pdf.(Normal(0.0, σ_y), y_fine)

pc = Plots.bar(y, bin_dens;
    label      = "Rouwenhorst — Binomial($(NS-1), 0.5) / Δy",
    color      = :teal,
    alpha      = 0.6,
    bar_width  = Δy * 0.9,
    xlabel     = "Log productivity  z",
    ylabel     = "Density",
    legend    = :topright,
    #title      = "Stationary distribution: NS = $NS vs N(0, $(σ_y)²)",
    fontsize  = 8)

plot!(y_fine, norm_dens;
    label     = "N(0, $(σ_y)²)",
    lw        = 2.5,
    color     = :firebrick,
    linestyle = :dash)

fig0 = Plots.plot(pc, layout=(1,1), size=(500, 300), margin=1Plots.mm)
Plots.savefig(fig0, "rouwenhorst_approximation_stationary.pdf")
Plots.display(fig0)
# ========================================
# Solve for GE equilibrium: r, w, phi, C, K, CVs, policy functions, and output struct
# ========================================
    
r, w, phi, asset_probs, C, K, CV_C, CV_K, a_pol, c_pol, para =
    general_equilibrium(para; use_egm=true)

@printf "\nBaseline equilibrium\n"
@printf "  r* = %.4f%%   K* = %.4f   C* = %.4f\n"  r*100 K C
@printf "  CV(c) = %.2f%%   CV(w) = %.2f%%\n"       CV_C CV_K
@printf "  Liquidity premium = %.4f%%\n"             ((1-para.β)/para.β - r)*100

###############################################

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
z_lo  = 1
z_mid = ceil(Int, para.NS / 2)
z_hi  = para.NS

pc = Plots.plot(xlabel="Assets", ylabel="Consumption",
          title="Consumption policy c(a,z)", legend=:topleft)
ps = Plots.plot(xlabel="Assets", ylabel="Next-period assets",
          title="Savings policy a'(a,z)", legend=:topleft)

for (z_i, zlabel, zcol) in [(z_lo, "z = low", C_ACC), (z_mid, "z = mid", C_MAIN), (z_hi, "z = high", C_MED)]
    plot!(pc, para.a[ind], c_pol[ind, z_i], color=zcol, lw=1.8, label=zlabel)
    plot!(ps, para.a[ind], a_pol[ind, z_i], color=zcol, lw=1.8, label=zlabel)
end
plot!(ps, para.a[ind], para.a[ind],
      color=:black, ls=:dash, lw=0.8, alpha=0.5, label="45°")

fig2 = Plots.plot(pc, layout=(1,1), size=(400, 300), margin=5Plots.mm)
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
mpc_bins = range(0, 1; length=40)
mpc_hist = [sum(all_wts[(all_mpc .>= mpc_bins[i]) .& (all_mpc .< mpc_bins[i+1])])
            for i in 1:(length(mpc_bins)-1)]
bin_mids = collect(0.5 .* (mpc_bins[1:end-1] .+ mpc_bins[2:end]))
pm2 = Plots.bar(bin_mids, mpc_hist,
          bar_width=step(mpc_bins)*0.85, color=C_MAIN, alpha=0.7,
          xlabel="MPC", ylabel="Population share",
          title="Distribution of MPC across agents", legend=false)

fig3 = Plots.plot(pm1, layout=(1,1), size=(450, 320), margin=0.1Plots.mm)
Plots.savefig(fig3, "fig3_mpc_distribution.pdf")
Plots.display(fig3)

# ============================================================
# EXERCISE 4  Lorenz curves and Gini
# ============================================================
println("\n=== Wealth distribution statistics ===")

gini_w   = compute_gini(para.a, asset_probs)
phi_flat = vec(phi)
c_flat   = vec(c_pol)
pi_z     = vec(sum(phi, dims=1))   # marginal over z: length NS

# Analytical income Gini: lognormal with unconditional std σ_y
σ_y    = para.σ / sqrt(1 - para.ρ^2)
gini_y = 2 * cdf(Normal(), σ_y / sqrt(2)) - 1

gini_c = compute_gini(c_flat, phi_flat)

@printf "  Gini wealth              = %.3f\n" gini_w
@printf "  Gini income (analytical) = %.3f\n" gini_y
@printf "  Gini consumption         = %.3f\n" gini_c
@printf "  σ_y (unconditional)      = %.4f\n" σ_y

s1, s10, s50 = compute_wealth_shares(para.a, asset_probs)
@printf "  Top 1%%  wealth share = %.1f%%\n" s1*100
@printf "  Top 10%% wealth share = %.1f%%\n" s10*100
@printf "  Top 50%% wealth share = %.1f%%\n" s50*100

# ── Wealth shares LaTeX table ─────────────────────────────────────
io_ws = open("wealth_shares_table.tex", "w")
write(io_ws, """
\\begin{tabular}{lc}
\\toprule
Percentile & Wealth share (\\%) \\\\
\\midrule
Top 1\\%  & $(round(s1*100, digits=1))\\% \\\\
Top 10\\% & $(round(s10*100, digits=1))\\% \\\\
Top 50\\% & $(round(s50*100, digits=1))\\% \\\\
\\bottomrule
\\end{tabular}
""")
close(io_ws)

# Lorenz curves
cw, lw_c = lorenz_curve(para.a, asset_probs)
cc, lc   = lorenz_curve(c_flat, phi_flat)

# Analytical lognormal Lorenz curve for income: L(p) = Φ(Φ⁻¹(p) − σ_y)
p_grid   = range(0.001, 0.999; length=500)
lorenz_y_analytical = cdf.(Normal(), quantile.(Normal(), p_grid) .- σ_y)
cy_an    = collect(p_grid)
ly_an    = lorenz_y_analytical

fig4 = Plots.plot(cw, lw_c, color=C_ACC, lw=2,
            label=@sprintf("Wealth (Gini=%.2f)", gini_w),
            xlabel="Cumulative population share",
            ylabel="Cumulative value share",
            title="Lorenz Curves with Ginis",
            size=(500,500))
Plots.plot!(fig4, cy_an, ly_an, color=C_MAIN, lw=2,
            label=@sprintf("Income — analytical lognormal (Gini=%.2f)", gini_y))
Plots.plot!(fig4, cc, lc, color=C_MED, lw=2,
            label=@sprintf("Consumption (Gini=%.2f)", gini_c))
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
    p_tmp = Para(b=0.0, NS=11, σ=sv, ρ=rv)
    r_eq, = general_equilibrium(p_tmp; use_egm=true, verbose=false)
    r_matrix[i,j] = r_eq * 100
    @printf "  sigma=%.1f  rho=%.1f  r*=%.3f%%  premium=%.3f%%\n" sv rv r_eq*100 (beta_rate-r_eq*100)
end

fig5 = Plots.plot(xlabel=L"Income persistence $\rho_z$",
            ylabel="Liquidity premium (pp)",
            title="Precautionary Savings Premium by Income Risk",
            size=(600,380))
for (i, sv) in enumerate(sigma_vals)
    Plots.plot!(fig5, rho_vals, beta_rate .- r_matrix[i,:],
          marker=:circle, lw=2,
          color=(i==1 ? C_MAIN : C_ACC),
          label=L"$\sigma_z$ = %$sv")
end
Plots.savefig(fig5, "fig5_precautionary_premium.pdf")
Plots.display(fig5)

# ============================================================
# EXERCISE 6  Aiyagari (1994) Table I — extended
#             Columns: rho, r*, K*, liq. premium, MPC, Gini, %constrained
#             Two panels: sigma=0.2 and sigma=0.4
#             One AD benchmark column (sigma irrelevant for AD)
# ============================================================

println("\n=== Table: GE statistics (sigma = 0.2 and sigma = 0.4) ===")
para_tab  = Para(b=0.0, NS=11)
rho_star  = (1 - para_tab.β) / para_tab.β
rho_vals_tab = [0.0, 0.3, 0.6, 0.9]

_, _, _, t02 = generate_stats_table(rho_vals_tab, 0.2, para_tab; use_egm=true)
_, _, _, t04 = generate_stats_table(rho_vals_tab, 0.4, para_tab; use_egm=true)

# AD benchmark: no idiosyncratic risk → r = ρ, K from firm FOC, MPC≈0, Gini=0
K_AD = d(rho_star, para_tab)
ad = (r=rho_star*100, K=K_AD, liq=0.0, mpc=0.0, gini=0.0, pct=0.0)

println("\n--- sigma = 0.2 ---"); println(t02)
println("\n--- sigma = 0.4 ---"); println(t04)
CSV.write("aiyagari_table_02.csv", t02)
CSV.write("aiyagari_table_04.csv", t04)

# ── LaTeX table ────────────────────────────────────────────────────────────────
io = open("aiyagari_table.tex", "w")
write(io, raw"""
\begin{center}
{\small \textbf{Aiyagari (1994) Table I --- Extended}\\[3pt]
Baseline: $\beta=0.96$, $\gamma=2$, $\alpha=0.36$, $\delta=0.08$, $b=0$, $A=1$}\\[6pt]
\begin{tabular}{lcccc|cccc|c}
\toprule
& \multicolumn{4}{c|}{$\sigma_z = 0.2$}
& \multicolumn{4}{c|}{$\sigma_z = 0.4$}
& AD \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
$\rho_z$ & 0.0 & 0.3 & 0.6 & 0.9 & 0.0 & 0.3 & 0.6 & 0.9 & --- \\
\midrule
""")

fmt2(x) = @sprintf("%.2f", x)
fmt3(x) = @sprintf("%.3f", x)

rows = [
    (raw"r^* (\%)",              t02.r_star_pct,          t04.r_star_pct,          fmt2, ad.r),
    (raw"K^*",                   t02.K_eq,                t04.K_eq,                fmt2, ad.K),
    (raw"\text{Liq. prem. (pp)}",t02.liquidity_premium_pp,t04.liquidity_premium_pp,fmt2, ad.liq),
    (raw"\text{Agg. MPC}",       t02.agg_MPC,             t04.agg_MPC,             fmt3, ad.mpc),
    #(raw"\text{Gini (wealth)}",  t02.Gini_wealth,         t04.Gini_wealth,         fmt3, ad.gini),
    (raw"\% \text{ at constr.}", t02.pct_constrained,     t04.pct_constrained,     fmt2, ad.pct),
]

for (label, v02, v04, fmtfn, ad_val) in rows
    cells02 = join([fmtfn(v) for v in v02], " & ")
    cells04 = join([fmtfn(v) for v in v04], " & ")
    write(io, "\$$(label)\$ & $(cells02) & $(cells04) & $(fmtfn(ad_val)) \\\\\n")
end

write(io, raw"""
\bottomrule
\multicolumn{10}{l}{\footnotesize AD = Arrow-Debreu. $\sigma_z$ irrelevant for AD: $r^*=\rho$, MPC$\approx r^*$.}
\end{tabular}
\end{center}
""")
close(io)
println("  LaTeX table written to aiyagari_table.tex")
print(read("aiyagari_table.tex", String))


# ============================================================
# EXERCISE 7  Borrowing limit and social insurance
# ============================================================
println("\n=== Borrowing limit comparative statics ===")
b_vals = [0.0, 0.5, 1.0, 2.0]
r_b = Float64[]; K_b = Float64[]; g_b = Float64[]

for bv in b_vals
    p_tmp = Para(b=bv, NS=7)
    r_eq, _, phi_tmp, _, _, K_eq, _, _, _, _, p_out =
        general_equilibrium(p_tmp; use_egm=true, verbose=false)
    ap_marg = dropdims(sum(phi_tmp; dims=2), dims=2)   # marginal asset distribution
    g = compute_gini(p_out.a, ap_marg)
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
# EXERCISE 8  Welfare cost of consumption volatility — Lucas vs Aiyagari
# ============================================================
println("\n=== Welfare costs of consumption volatility ===")
V_im = compute_value_function(c_pol, a_pol, para)
λ_grid, λ_agg, λ_bot10 = welfare_cost_exact(V_im, C, phi, asset_probs, para)

@printf("Lucas (aggregate)  welfare cost = %.4f%% of consumption\n", λ_agg*100)
@printf("Bottom-10%%         welfare cost = %.4f%% of consumption\n", λ_bot10*100)
@printf("Ratio bottom / aggregate        = %.1fx\n", λ_bot10/λ_agg)

println("\n=== All figures saved. Run complete. ===")

###########################
# ============================================================
# EXERCISE 0b  GE diagram: K^s(r) vs K^d(r)
# ============================================================
println("\n=== Computing GE diagram: capital supply and demand ===")

rho_rate  = (1 - para.β) / para.β   # ρ = time preference rate
r_lo_scan = -0.03                    # start well above -δ: K^d stays finite
r_hi_scan = rho_rate - 0.002        # stay just below ρ (transversality)
r_scan    = range(r_lo_scan, r_hi_scan; length=30)

# K^s(r): solve HH problem at each r, aggregate distribution
K_supply = Float64[]
for rv in r_scan
    p_tmp      = deepcopy(para)
    p_tmp.r    = rv
    p_tmp.w    = r_to_w(rv, para)
    update_params!(p_tmp)
    c_init     = 0.5 .* repeat(p_tmp.a, 1, p_tmp.NS)
    a_p, _     = solve_model_egm(c_init, p_tmp; verbose=false)
    a_p        = clamp.(a_p, -p_tmp.b + 1e-10, p_tmp.grid_max - 1e-10)
    ab_pol     = histc(a_p, p_tmp.a)
    wei        = (a_p .- p_tmp.a[ab_pol]) ./ (p_tmp.a[ab_pol .+ 1] .- p_tmp.a[ab_pol])
    phi_tmp    = fill(1.0 / (p_tmp.NA * p_tmp.NS), p_tmp.NA, p_tmp.NS)
    invariant_dist!(phi_tmp, ab_pol, wei, p_tmp)
    ap_marg    = dropdims(sum(phi_tmp; dims=2), dims=2)
    push!(K_supply, dot(ap_marg, p_tmp.a))
end

# K^d(r): dense grid for smooth firm-demand curve
K_RA     = d(rho_rate, para)           # RA: firm FOC at r = ρ — defined FIRST
K_cap    = max(K_RA, K) * 1.8          # show up to 1.8x the larger equilibrium
r_lo_cap = para.α * para.A * K_cap^(para.α - 1) - para.δ  # r that yields K_cap
r_dense  = range(r_lo_cap, rho_rate - 0.0005; length=300)
K_d_plot = [d(rv, para) for rv in r_dense]

# IM equilibrium already in (K, r) from Exercise 0

# Axis limits: pad around the interesting region
K_min = minimum(K_supply) * 0.7
K_max = max(K_RA, K) * 1.5
r_pad = 0.008


fig_ge = Plots.plot(
    size        = (640, 430),
    xlabel      = "log(K)",
    ylabel      = "Interest rate  r",
    title       = "General Equilibrium: Capital Market",
    legend      = :bottomright,
    framestyle  = :box,
    #xscale      = :log10,
    xlims       = (log(K_min), log(K_max)),
    ylims       = (r_lo_scan - r_pad, rho_rate + r_pad * 1.5),
    tickfontsize = 9,
    guidefontsize = 10,
    legendfontsize = 9)

# Firm capital demand
Plots.plot!(fig_ge, log.(K_d_plot), collect(r_dense),
    color = "#2c7bb6", lw = 2.2,
    label = "r*(K): Firm demand")

# HH capital supply (IM)
Plots.plot!(fig_ge, log.(K_supply), collect(r_scan),
    color = "#d7191c", lw = 2.2, marker = :none,
    label = "K^s(r): HH savings (IM)")

# ρ horizontal asymptote (RA equilibrium condition)
Plots.hline!(fig_ge, [rho_rate],
    color = :black, ls = :dot, lw = 1.5,
    label = "rho  (RA: r = rho)")

# RA equilibrium dot
Plots.scatter!(fig_ge, [log(K_RA)], [rho_rate],
    color = :gray, ms = 7, msw = 0,
    label = "K*_RA")
Plots.plot!(fig_ge, [log(K_RA), log(K_RA)], [r_lo_scan - r_pad, rho_rate],
    color = :gray, ls = :dash, lw = 0.9, label = nothing)

# IM equilibrium dot
Plots.scatter!(fig_ge, [log(K)], [r],
    color = "#2c7bb6", ms = 7, msw = 0,
    label = "K*_IM, r*_IM")
Plots.plot!(fig_ge, [log(K), log(K)], [r_lo_scan - r_pad, r],
    color = "#2c7bb6", ls = :dash, lw = 0.9, label = nothing)

# Annotations — plain text, offset to avoid marker overlap
Plots.annotate!(fig_ge, log(K_RA) * 1.08, rho_rate - 0.005,
    Plots.text("K*_RA", 9, :gray, :left))
Plots.annotate!(fig_ge, log(K) * 1.08, r - 0.005,
    Plots.text("K*_IM", 9, "#2c7bb6", :left))

@printf "  K_RA = %.3f   K_IM = %.3f   r*_IM = %.4f%%\n" K_RA K r*100
Plots.savefig(fig_ge, "fig0b_ge_diagram.pdf")
Plots.display(fig_ge)
println("  GE diagram saved to fig0b_ge_diagram.pdf")