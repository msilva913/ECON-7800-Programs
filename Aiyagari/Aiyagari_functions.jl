# ============================================================
# Aiyagari_functions.jl
# Core functions for Aiyagari (1994) incomplete-markets model.
# Backend-agnostic: plotting handled in main script.
# ============================================================

using Parameters, CSV, Random, QuantEcon
using LinearAlgebra, LinearInterpolations
using DataFrames
using Printf

# ── Grid ──────────────────────────────────────────────────────────────────────

"""
    grid_cons_grow(n, left, right, g)
n gridpoints on [left, right] with geometrically growing spacing.
Denser near `left` — clusters points near the borrowing constraint.
"""
function grid_cons_grow(n, left, right, g)
    x = zeros(n)
    for i in 0:(n-1)
        x[i+1] = left + (right - left) / ((1+g)^(n-1) - 1) * ((1+g)^i - 1)
    end
    return x
end

"""
    histc(x, binranges)
Map values in x to indices of their left bin in binranges (sorted).
"""
function histc(x, binranges)
    return searchsortedlast.(Ref(binranges), x)
end

# ── Parameters ────────────────────────────────────────────────────────────────

@with_kw mutable struct Para{T1,T2,T3}
    β::Float64 = 0.96
    γ::Float64 = 2.0
    ρ::Float64 = 0.9               # persistent income component (Flodén-Lindé 2001)
    σ::Float64 = 0.2               # conditional std of log-income (Flodén-Lindé 2001)
    NS::Int64  = 11                 # number of income states (Rouwenhorst approximation)
    b::Float64 = 0.0               # borrowing limit (a' ≥ -b)
    grid_max::Float64 = 50.0
    NA::Int64  = 200
    A::Float64 = 1.0
    N::Float64 = 1.0
    α::Float64 = 0.36
    δ::Float64 = 0.08
    r::Float64 = 0.01
    w::Float64 = 1.0
    R::Float64 = 1 + r
    u           = (c, γ=γ) -> c^(1-γ)/(1-γ)
    u_prime     = c -> c^(-γ)
    u_prime_inv = c -> c^(-1/γ)
    mc::T1 = rouwenhorst(NS, ρ, σ*(1-ρ^2)^0.5, 0.0)
    P::T2  = mc.p
    y::Vector{Float64} = exp.(mc.state_values)
    @assert R * β < 1 "Transversality violated: β*(1+r) ≥ 1"
    a::T3 = grid_cons_grow(NA, -b, grid_max, 0.10)
end


"""
    update_params!(para)
Refresh all derived fields. Call after setfield!(para, :field, value).
"""
function update_params!(para)
    @unpack r, γ, σ, ρ, NS, β, b, grid_max, NA = para
    para.R           = 1 + r
    para.u           = c -> c^(1-γ)/(1-γ)
    para.u_prime     = c -> c^(-γ)
    para.u_prime_inv = c -> c^(-1/γ)
    para.mc = rouwenhorst(NS, ρ, σ*(1-ρ^2)^0.5, 0.0)
    para.P           = para.mc.p
    para.y           = exp.(para.mc.state_values)
    @assert para.R * β < 1 "Transversality violated"
    para.a           = grid_cons_grow(NA, -b, grid_max, 0.10)
    return nothing
end

# ── Firm optimality ───────────────────────────────────────────────────────────

r_to_w(r, para) = para.A * (1-para.α) * (para.A*para.α / (r+para.δ))^(para.α/(1-para.α))
rd(K, para)     = para.A * para.α * (para.N/max(K, 1e-10))^(1-para.α) - para.δ
d(r, para)      = para.N * (para.A*para.α / (r+para.δ))^(1/(1-para.α))

# ── Household solvers ─────────────────────────────────────────────────────────

"""
    time_iter(a_prime, para; omega=0.7)
Coleman operator. Bug fix: a1 computed outside z_hat loop.
"""
function time_iter(a_prime, para; omega=0.7)
    @unpack R, P, y, β, w, u_prime, u_prime_inv, a, b, NS = para
    a_prime_fun(a_i, z) = Interpolate(a, @view(a_prime[:,z]), extrapolate=:reflect)(a_i)
    expect      = zero(a_prime)
    c           = zero(a_prime)
    a_prime_new = zero(a_prime)
    for z in 1:NS
        for (i, a_i) in enumerate(a)
            a1 = a_prime_fun(a_i, z)          # outside z_hat loop
            for z_hat in 1:NS
                a2 = a_prime_fun(a1, z_hat)
                c_prime = w*y[z_hat] + R*a1 - a2
                expect[i,z] += β * R * u_prime(c_prime) * P[z, z_hat]
            end
            c[i,z]           = u_prime_inv(expect[i,z])
            a_prime_new[i,z] = w*y[z] + R*a_i - c[i,z]
        end
    end
    a_prime_new = omega .* a_prime_new .+ (1-omega) .* a_prime
    a_prime_new = max.(a_prime_new, -b)
    return a_prime_new, c
end

"""
    egm_step(c_on_grid, para)
One step of the Endogenous Grid Method (Carroll 2006).
No dampening needed; analytically inverts the Euler equation.
"""
function egm_step(c_on_grid, para)
    @unpack R, P, y, β, w, u_prime, u_prime_inv, a, b, NS, NA = para
    MU_exp = zeros(NA, NS)
    for z in 1:NS, z_hat in 1:NS
        @. MU_exp[:,z] += β * R * P[z,z_hat] * u_prime(c_on_grid[:,z_hat])
    end
    c_unc       = u_prime_inv.(MU_exp)
    c_new       = zeros(NA, NS)
    a_prime_new = zeros(NA, NS)
    for z in 1:NS
        a_endog_z = @. (c_unc[:,z] + a - w*y[z]) / R
        for (i, a_i) in enumerate(a)
            if a_i <= a_endog_z[1]
                a_prime_new[i,z] = -b
                c_new[i,z]       = max(w*y[z] + R*a_i - (-b), 1e-10)
            else
                c_new[i,z]       = Interpolate(a_endog_z, c_unc[:,z],
                                               extrapolate=:reflect)(a_i)
                a_prime_new[i,z] = w*y[z] + R*a_i - c_new[i,z]
            end
        end
    end
    a_prime_new = clamp.(a_prime_new, -b, a[end-1])  # a[end] would cause OOB in histc
    return a_prime_new, c_new
end

function solve_model_time_iter(a_prime, para;
        tol=1e-7, max_iter=2000, verbose=true, print_skip=25, omega=0.6)
    i = 1; error = tol + 1.0; c = similar(a_prime)
    while (i < max_iter) && (error > tol)
        a_prime_new, c = time_iter(a_prime, para; omega)
        error = maximum(abs.(a_prime_new .- a_prime))
        i += 1
        verbose && (i % print_skip == 0) &&
            @printf("  Coleman iter %d: error = %.2e\n", i, error)
        a_prime = a_prime_new
    end
    i == max_iter && @warn "time_iter: did not converge"
    return a_prime, c
end

function solve_model_egm(c_init, para;
        tol=1e-7, max_iter=2000, verbose=true, print_skip=25)
    i = 1; error = tol + 1.0; c = copy(c_init)
    while (i < max_iter) && (error > tol)
        _, c_new = egm_step(c, para)
        error = maximum(abs.(c_new .- c))
        i += 1
        verbose && (i % print_skip == 0) &&
            @printf("  EGM iter %d: error = %.2e\n", i, error)
        c = c_new
    end
    i == max_iter && @warn "EGM: did not converge"
    a_prime, c = egm_step(c, para)
    return a_prime, c
end

# ── Stationary distribution ───────────────────────────────────────────────────

function update_dist(phi, ab_pol, wei, para)
    @unpack P, NA, NS = para
    phi_new = zero(phi)
    for is in 1:NS, ia in 1:NA
        a_p = ab_pol[ia, is]
        for is_p in 1:NS
            w_up = wei[ia, is]
            phi_new[a_p,   is_p] += (1 - w_up) * P[is, is_p] * phi[ia, is]
            phi_new[a_p+1, is_p] +=      w_up  * P[is, is_p] * phi[ia, is]
        end
    end
    return phi_new
end

function invariant_dist!(phi, ab_pol, wei, para; tol_dist=1e-8, max_iter=5000)
    dif  = 1.0
    iter = 0
    while dif > tol_dist && iter < max_iter
        iter   += 1
        phi_new = update_dist(phi, ab_pol, wei, para)
        dif     = maximum(abs.(phi_new .- phi))
        phi    .= phi_new ./ sum(phi_new)
    end
    iter == max_iter && @warn "invariant_dist!: did not converge after $max_iter iterations (dif=$dif)"
end

# ── Summary statistics ────────────────────────────────────────────────────────

function sum_stats(phi, c_pol, para)
    @unpack a, NA, NS = para
    asset_probs = dropdims(sum(phi; dims=2), dims=2)
    K     = dot(asset_probs, a)
    C     = sum(c_pol .* phi)
    std_C = sqrt(sum((c_pol .- C).^2 .* phi))
    std_K = sqrt(sum((a .- K).^2 .* asset_probs))
    CV_C  = std_C / C * 100
    CV_K  = std_K / K * 100
    return asset_probs, C, K, CV_C, CV_K
end

# ── Additional analyses ───────────────────────────────────────────────────────

"""
    compute_mpc(c_pol, para; da=1e-3)
MPC at each state via finite difference on the consumption policy.
MPC ≈ 1 near constraint; ≈ 0 for wealthy (PIH).
"""
function compute_mpc(c_pol, para; da=1e-3)
    @unpack a, NA, NS = para
    mpc = zeros(NA, NS)
    for z in 1:NS
        c_interp = Interpolate(a, c_pol[:,z], extrapolate=:reflect)
        for (i, a_i) in enumerate(a)
            mpc[i,z] = (c_interp(a_i + da) - c_interp(a_i)) / da
        end
    end
    return clamp.(mpc, 0.0, 1.0)
end

"""
    compute_gini(values, weights)
Gini coefficient in [0,1] from values with probability weights.
Sorts ascending, normalises weights, computes area under Lorenz curve.
Note: standard formula assumes non-negative values. When borrowing
is allowed (b>0) some asset values may be negative — the returned
Gini is then the extended Gini (can exceed 1) and should be
interpreted with care.
"""
function compute_gini(values, weights)
    idx    = sortperm(values)
    v      = values[idx]
    w      = weights[idx] ./ sum(weights)
    wv_sum = dot(w, v)
    wv_sum ≈ 0.0 && return NaN   # guard: all values zero (e.g. b=0, everyone at constraint)
    cum_w  = cumsum(w)
    cum_wv = cumsum(w .* v) ./ wv_sum
    area   = sum(0.5 .* diff(cum_w) .* (cum_wv[1:end-1] .+ cum_wv[2:end]))
    return 1.0 - 2.0 * area
end

"""
    lorenz_curve(values, weights)
Returns (cum_pop_share, cum_value_share) for Lorenz curve plot.
"""
function lorenz_curve(values, weights)
    idx    = sortperm(values)
    v      = values[idx]
    w      = weights[idx] ./ sum(weights)
    cum_w  = [0.0; cumsum(w)]
    cum_wv = [0.0; cumsum(w .* v) ./ dot(w, v)]
    return cum_w, cum_wv
end

"""
    compute_wealth_shares(a, asset_probs)
Share of total wealth held by top 1%, top 10%, top 50%.
"""
function compute_wealth_shares(a, asset_probs)
    probs = asset_probs ./ sum(asset_probs)
    total = dot(probs, a)
    total == 0 && return (NaN, NaN, NaN)
    idx     = sortperm(a)
    a_s     = a[idx]; p_s = probs[idx]
    cum_pop = cumsum(p_s)
    s_top1  = dot(a_s[cum_pop .>= 0.99], p_s[cum_pop .>= 0.99]) / total
    s_top10 = dot(a_s[cum_pop .>= 0.90], p_s[cum_pop .>= 0.90]) / total
    s_top50 = dot(a_s[cum_pop .>= 0.50], p_s[cum_pop .>= 0.50]) / total
    return s_top1, s_top10, s_top50
end

"""
    compute_value_function(c_pol, para; tol=1e-8, max_iter=10000)
Compute V(a,z) by iterating V = u(c) + β E[V'] given converged c_pol.
"""
function compute_value_function(c_pol, a_pol, para; tol=1e-8, max_iter=10000)
    @unpack β, u, P, NA, NS, a = para
    V     = u.(c_pol) ./ (1 - β)   # initialise at myopic value
    V_new = similar(V)
    for _ in 1:max_iter
        for z in 1:NS, i in 1:NA
            EV = 0.0
            for z_p in 1:NS
                # interpolate V at a_pol[i,z] for each z_p
                EV += P[z, z_p] * Interpolate(a, V[:, z_p],
                                              extrapolate=:reflect)(a_pol[i,z])
            end
            V_new[i,z] = u(c_pol[i,z]) + β * EV
        end
        maximum(abs.(V_new .- V)) < tol && break
        V .= V_new
    end
    return V
end

"""
    welfare_cost_exact(V_im, C_star, phi, para)
Exact consumption-equivalent welfare cost λ(a,z) such that
    V_IM(a,z) = u((1-λ)C*)/(1-β)
For CRRA with γ≠1:
    λ(a,z) = 1 - [-V_IM(a,z)*(1-β)*(γ-1)]^(1/(1-γ)) / C_star

Returns λ_grid (NA×NS) and distribution-weighted aggregates.
"""
function welfare_cost_exact(V_im, C_star, phi, asset_probs, para)
    @unpack γ, β, NA, NS = para

    λ_grid = zeros(NA, NS)
    for z in 1:NS, i in 1:NA
        # For CRRA γ > 1: V < 0, so (1-γ)*(1-β)*V_im > 0
        ratio = (1 - γ) * (1 - β) * V_im[i, z]
        ratio <= 0 && continue          # ← was ||, must be &&
        λ_grid[i, z] = 1.0 - ratio^(1/(1-γ)) / C_star
    end

    # Aggregate welfare cost: E_Φ[λ]
    λ_agg = sum(λ_grid .* phi)

    # Bottom 10% of wealth distribution (by marginal asset probabilities)
    cum_pop   = cumsum(asset_probs ./ sum(asset_probs))
    bot10_mask = cum_pop .<= 0.10
    phi_bot   = phi[bot10_mask, :]
    denom     = sum(phi_bot)
    λ_bot10   = denom > 0 ? sum(λ_grid[bot10_mask, :] .* phi_bot) / denom : NaN

    return λ_grid, λ_agg, λ_bot10
end

# ── General equilibrium ───────────────────────────────────────────────────────

"""
    general_equilibrium(para; tol_r, use_egm, verbose)
Bisection on r: find r* where HH capital supply = firm capital demand.
"""
function general_equilibrium(para; tol_r=1e-5, use_egm=true, verbose=true)
    @unpack β, δ, NA, NS, a, b, grid_max = para
    r_max = (1 - β) / β
    r_min = -δ
    err   = 1.0
    c_init      = 0.5 .* repeat(a, 1, NS)
    a_pol       = similar(c_init)
    c_pol       = similar(c_init)
    phi         = fill(1.0 / (NA * NS), NA, NS)
    asset_probs = zeros(NA)
    C = K_supply = CV_C = CV_K = r = w = 0.0
    p2   = deepcopy(para)   # initialize before loop; overwritten each iteration
    iter = 0
    max_iter = 200

    while abs(err) > tol_r && iter < max_iter
        iter += 1
        r  = 0.5 * (r_min + r_max)
        w  = r_to_w(r, para)
        p2.r = r
        p2.w = w
        p2.R = 1 + r
        update_params!(p2)

        if use_egm
            a_pol, c_pol = solve_model_egm(c_init, p2; verbose=false)
        else
            a_pol, c_pol = solve_model_time_iter(repeat(p2.a,1,NS), p2;
                                                  verbose=false, omega=0.5)
        end
        c_init = c_pol   # warm-start next bisection iteration

        a_pol  = clamp.(a_pol, -b + 1e-10, grid_max - 1e-10)
        ab_pol = histc(a_pol, p2.a)
        wei    = (a_pol .- p2.a[ab_pol]) ./ (p2.a[ab_pol .+ 1] .- p2.a[ab_pol])
        invariant_dist!(phi, ab_pol, wei, p2)
        asset_probs, C, K_supply, CV_C, CV_K = sum_stats(phi, c_pol, p2)

        r1  = rd(K_supply, p2)
        err = r1 - r
        err < 0 ? (r_max = r) : (r_min = r)
        verbose && @printf("  K=%.4f  r=%.5f  r_firm=%.5f  err=%.2e\n",
                            K_supply, r, r1, err)
    end
    iter == max_iter && @warn "general_equilibrium: bisection did not converge after $max_iter iterations (err=$err)"
    return r, w, phi, asset_probs, C, K_supply, CV_C, CV_K, a_pol, c_pol, p2  # p2 has correct r,w
end

# ── Table generator ───────────────────────────────────────────────────────────

"""
    generate_stats_table(rho_vals, σ_val, para; use_egm=true)
Replicate Aiyagari (1994) Table I, extended with Gini and liquidity premium.
"""
function generate_stats_table(rho_vals, σ_val, para; use_egm=true)
    para = deepcopy(para); para.σ = σ_val; update_params!(para)
    rho_star = (1 - para.β) / para.β

    r_v   = similar(rho_vals); K_v   = similar(rho_vals)
    CVC   = similar(rho_vals); CVK   = similar(rho_vals)
    G_v   = similar(rho_vals); Liq   = similar(rho_vals)
    MPC_v = similar(rho_vals); PCT_v = similar(rho_vals)

    for (i, ρ) in enumerate(rho_vals)
        para.ρ = ρ; update_params!(para)
        r, _, phi, asset_probs_i, _, K, CV_C, CV_K, _, c_pol, p_out =
            general_equilibrium(para; use_egm, verbose=false, tol_r=1e-5)
        r_v[i]   = r; K_v[i] = K; CVC[i] = CV_C; CVK[i] = CV_K
        G_v[i]   = compute_gini(p_out.a, asset_probs_i)   # marginal asset distribution
        Liq[i]   = (rho_star - r) * 100
        mpc      = compute_mpc(c_pol, p_out)
        MPC_v[i] = sum(mpc .* phi)
        pct_mask = p_out.a .<= p_out.b
        PCT_v[i] = sum(asset_probs_i[pct_mask]) * 100
    end

    t = DataFrame(rho=rho_vals, r_star_pct=r_v.*100, K_eq=K_v,
                  liquidity_premium_pp=Liq, agg_MPC=MPC_v,
                  Gini_wealth=G_v, pct_constrained=PCT_v)
    return CVC, K_v, r_v, t
end

# ── Comparative statics helpers ───────────────────────────────────────────────
# Note: these functions use Plots.jl — requires `using Plots; gr()` in main script.

"""
    comp_statics_plot(field, vals, para; z_idx=1)
Partial-equilibrium: consumption policy vs assets for varying parameter.
"""
function comp_statics_plot(field, vals, para; z_idx=1)
    par = deepcopy(para)
    p   = plot(xlabel="Assets", ylabel="Consumption (lowest income state)",
               title="Comparative statics: $field", legend=:topleft)
    for val in vals
        setfield!(par, field, val)
        update_params!(par)
        c_init = 0.5 .* repeat(par.a, 1, par.NS)
        _, c_star = solve_model_egm(c_init, par; verbose=false)
        plot!(p, par.a, c_star[:,z_idx], label="$field=$val", lw=1.8, alpha=0.75)
    end
    display(p)
    return p
end

"""
    comp_statics_GE_plot(field, vals; use_egm=true)
GE comparative statics: r*, K*, C* as functions of a parameter.
"""
function comp_statics_GE_plot(field, vals; use_egm=true)
    r_v = similar(vals); K_v = similar(vals); C_v = similar(vals)
    for (i, val) in enumerate(vals)
        par = Para(); setfield!(par, field, val); update_params!(par)
        r, _, _, _, C, K, _, _, _, _, _ = general_equilibrium(par; use_egm, verbose=false)
        r_v[i], K_v[i], C_v[i] = r, K, C
    end
    p1 = plot(vals, r_v.*100, xlabel="$field", ylabel="r* (%)",  legend=false, lw=2)
    p2 = plot(vals, K_v,      xlabel="$field", ylabel="K*",       legend=false, lw=2)
    p3 = plot(vals, C_v,      xlabel="$field", ylabel="C*",       legend=false, lw=2)
    fig = plot(p1, p2, p3, layout=(1,3), size=(1100,360),
               plot_title="GE comparative statics: $field")
    display(fig)
    return r_v, K_v, C_v
end