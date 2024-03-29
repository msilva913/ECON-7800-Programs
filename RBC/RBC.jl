
using PyPlot
using BenchmarkTools
using LaTeXStrings
using Parameters, CSV, Random, QuantEcon
using NLsolve
using LinearAlgebra, Roots, LinearInterpolations, Interpolations, Dierckx
using Printf
using DataFrames
#include("time_series_fun.jl")
cd(@__DIR__)
parent_dir = dirname(@__DIR__)
cd(parent_dir)
include("time_series_fun.jl")

columns(M) = [view(M, :, i) for i in 1:size(M, 2)]

"""
    calibrate(r, α, l, δ, ρ_x, σ_x)
Calibrate RBC model given long-run targets.
"""
function calibrate(r=0.03, α=0.33, l=0.33, δ=0.025, ρ_x=0.974, σ_x=0.009)
    " Convert r to quarterly "
    r_q = (1+r)^(1.0/4.0)-1.0
    β = 1/(1+r_q)
    " Ratio of capital to labor supply "
    k_l = (α/(1/β-(1-δ)))^(1/(1-α))
    " Solve for theta "
    θ = ((1-l)/l*(1-α)*k_l^α)/(k_l^α-δ*k_l)
    CalibratedParameters = (α=α, β=β, δ=δ, θ=θ, ρ_x=ρ_x, σ_x=σ_x)
    return CalibratedParameters
end

"""
    steady_state(params)
Calculate steady state of model
"""
function steady_state(params)
    @unpack α, β, δ, θ, ρ_x, σ_x = params
    " capital-labor ratio "
    k_l = (α/(1/β-(1-δ)))^(1/(1-α))
    " wage and rental rate "
    w = (1-α)*k_l^α
    R = α*k_l^(α-1)
    " consumption-labor ratio "
    c_l = k_l^α-δ*k_l
    " other variables "
    l = ((1-α)/θ*k_l^(α-1))/((θ+1-α)/θ*k_l^(α-1)-δ)
    c = l*c_l
    k = k_l*l
    y = k^α*l^(1-α)
    i = y-c
    lab_prod = y/l

    SteadyState = (l=l, c=c, k=k, y=y, i=i, w=w, R=R, lab_prod=lab_prod)
    return SteadyState
end


@with_kw mutable struct Para{T1, T2, T3, T4}
    # model parameters
    α::Float64 = 0.33
    β::Float64 = 0.99
    δ::Float64 = 0.025
    θ::Float64 = 1.82
    # Shocks
    ρ_x::Float64 = 0.974
    σ_x::Float64 = 0.009

    # numerical parameter
    k_l::Float64 = 5.0 # lower bound for capital
    k_u::Float64 = 15.0 # upper bound for capital
    NK::Int64 = 50 # number of gridpoints for capital
    NS::Int64 = 9 # number of elements of Markov chain for productivity
    mc::T1 = rouwenhorst(NS, ρ_x, σ_x, 0) # Markov chain object
    P::T2 = mc.p # Transition matrix
    A::T3 = exp.(mc.state_values) # Levels of productivity
    k_grid::T4 = range(k_l, stop=k_u, length =NK) # grid of capital
end

function update_params!(self, cal)
    @unpack α, β, δ, θ = cal
    self.α = α
    self.β = β
    self.δ = δ
    self.θ = θ
    nothing
end

"""
    RHS_fun_cons(l_pol::Function, para::Para)
Compute the conditional expectation at each state (k,z) for a given labor supply policy
"""
function RHS_fun_cons(l_pol::Function, para::Para)
    @unpack α, β, δ, θ, P, NK, NS, k_grid, A = para
    # consumption given state and labo
    
    RHS = zeros(NK, NS)
    @inbounds Threads.@threads for z in 1:NS
        for (i, k) in collect(enumerate(k_grid))
            # labor policy
            l_i = l_pol(k, z)
            # consumption and output
            y = A[z]*k^α*l_i^(1-α)
            c = (1-l_i)/θ*(1-α)*y/l_i
            #c = min(c, y)
            # update capital
            k_p = (1-δ)*k + y - c
            for z_hat in 1:NS #possible future technology (determined by Markov chain)
                # update labor supply via interpolation
                l_p = l_pol(k_p, z_hat)
                # update consumption
                y_p = A[z_hat]*k_p^α*l_p^(1-α)
                c_p = (1-l_p)/θ*(1-α)*y_p/l_p
                #c_p = min(c_p, y_p)
                # future marginal utility
                RHS[i, z] +=β*P[z, z_hat]*((1/c_p)*(α*y_p/k_p+1-δ))
            end
        end
    end
    # Calculates right-hand side of Euler for given (k, z), where k may be off the grid
    rhs_fun(k, z) = LinearInterpolation(k_grid, RHS[:, z], extrapolation_bc=Line())(k)
    return rhs_fun
end

"""
    labor_supply_loss(l_i, k, z, RHS_fun, para)
Calculate discrepancy between consumption implied by intratemporal condition and
consumption implied by Euler equation
"""
# function labor_supply_loss(l_i, k, z, RHS_fun,  para::Para)

#     @unpack A, α, θ = para
#     # optimal consumption
#     y = A[z]*k^α*l_i^(1-α)
#     c = (1-l_i)/θ*(1-α)*y/l_i
#     #c = min(c, y)
#     error =  1/c - RHS_fun(k, z)
#     return error
# end

function labor_supply_update(l_i, k, z, RHS_fun, para)
    @unpack A, α, θ, = para
    # optimal consumption 
    y = A[z]*k^α*l_i^(1-α)
    c = 1/RHS_fun(k, z)
    l_new = (1-l_i)/θ*(1-α)*y/c
    return l_new
end

"""
    solve_model_time_iter(l, para::Para, tol=1e-8, max_iter=1000, verbose=true, print_skip=25)
Solve RBC model (find correct policies) given initial guess of labor supply policy
"""
function solve_model_time_iter(l, para::Para; tol=1e-8, max_iter=1000, verbose=true, 
                                print_skip=25)
    # Set up loop 
    @unpack k_grid, NS, A, α, θ= para
    # Initial consumption level
    l_new = similar(l)

    err = 1.0
    iter = 1
    while (iter < max_iter) && (err > tol)
        # interpolate given labor grid l
        l_pol(k, z) = LinearInterpolation(k_grid, @view(l[:, z]), extrapolation_bc=Line())(k)
        RHS_fun = RHS_fun_cons(l_pol, para)
        @inbounds Threads.@threads for z in 1:NS
            for (i, k) in collect(enumerate(k_grid))
                # solve for labor supply
                #l_i = find_zero(l_i -> labor_supply_loss(l_i, k, z, RHS_fun, para), (1e-10, 0.99), Bisection() )
                l_i = labor_supply_update(l[i, z], k, z, RHS_fun, para)
                l_new[i, z] = l_i
            end
        end
        #@printf(" %.2f", (mean(c)))
        err = maximum(abs.(l_new-l)/max.(abs.(l), 1e-10))
        if verbose && iter % print_skip == 0
            print("Error at iteration $iter is $err. \n")
        end
        iter += 1
        l .= l_new
    end

    # Get convergence level
    if iter == max_iter
        print("Failed to converge!")
    end

    # if verbose && (iter < max_iter)
    #     print("Converged in $iter iterations")
    # end
    y = similar(l)
    c = similar(l)
    for (i, k) in enumerate(k_grid)
        for z in 1:NS
            y[i, z] = A[z]*k^α*l[i, z]^(1-α)
            c[i, z] = (1-l[i, z])/θ*(1-α)*y[i, z]/l[i, z]
        end
    end

    inv = y - c
    w = (1-α).*y./l # you can also write w = @. (1-α)*y/l
    R = α.*y./k_grid
    l_pol(k, z) = LinearInterpolation(k_grid, l[:, z], extrapolation_bc=Line())(k)
    return l, l_pol, c, y, inv, w, R
end


"""
    simulate_series(l_mat, para::Para, burn_in=200, capT=10_000)
Simulate time series from model in levels
"""
function simulate_series(l_mat::Array, para::Para, burn_in=200, capT=10_000)

    @unpack ρ_x, σ_x, P, mc, A, α, θ, δ, k_grid = para
    l_pol(k, z) = Interpolate(k_grid, @view(l_mat[:, z]), extrapolate=:reflect)(k)
    capT = capT + burn_in + 1

    # Extract indices of simualtes shocks
    z_indices = simulate_indices(mc, capT)
    z_series = A[z_indices]
    # Simulate shocks
    
    k = ones(capT+1)
    var = ones(capT, 3)
    l, y, c = columns(var)

    for t in 1:capT
        l[t] = l_pol(k[t], z_indices[t])
        y[t] = z_series[t]*k[t]^α*l[t]^(1-α)
        c[t] = (1-l[t])/θ*(1-α)*y[t]/l[t]
        # update capital stock for next iteration
        k[t+1] = (1-δ)*k[t] + y[t] - c[t]
    end

    k = k[1:(end-1)] #pop!(k)
    i = y - c
    w = (1-α).*y./l
    R = α.*y./k
    lab_prod = y./l
    Simulation = (l=l, y=y, c=c, k=k, i=i, w=w, R=R,
                 lab_prod=lab_prod, η_x=log.(z_series), z_indices)
    return Simulation
end

"""
    impulse_response(l_mat, para, k_init; irf_length=40, scale=1.0)
Calculate impulse response to productivity shock 
"""
function impulse_response(l_mat::Matrix{Float64}, para, k_init; irf_length=40, scale=1.0)

    @unpack ρ_x, σ_x, P, mc, A, α, θ, δ, k_grid = para

    # Bivariate interpolation (AR(1) shocks, so productivity can go off grid)
    L = Spline2D(k_grid, A, l_mat)

    η_x = zeros(irf_length)
    η_x[1] = σ_x*scale

    for t in 1:(irf_length-1)
        η_x[t+1] = ρ_x*η_x[t]
    end
    z = exp.(η_x)
    z_bas = ones(irf_length)

    function impulse(z_series)

        k = zeros(irf_length+1)
        l = zeros(irf_length)
        c = zeros(irf_length)
        y = zeros(irf_length)

        k[1] = k_init

        for t in 1:irf_length
            # labor
            l[t] = L(k[t], z_series[t])
            y[t] = z_series[t]*k[t]^α*l[t]^(1-α)
            c[t] = (1-l[t])/θ*(1-α)*y[t]/l[t]
            k[t+1] = (1-δ)*k[t] + y[t] - c[t]
        end

        #k = k[1:(end-1)]
        pop!(k)
        i = y - c
        w = (1-α).*y./l
        R = α.*y./k
        lab_prod = y./l
        out = [c k l i w R y lab_prod]
        return out
    end

    out_imp = impulse(z) # collect values under impulse
    out_bas = impulse(z_bas) # collect baseline values (no shock occurs)

    irf_res = similar(out_imp)
    @. irf_res .= 100*log(out_imp/out_bas)
    #out = [log.(x./mean(getfield(simul, field))) for (x, field) in
    #zip([c, k[1:(end-1)], l, i, w, R, y, lab_prod], [:c, :k, :l, :i, :w, :R, :y, :lab_prod])]
    c, k, l, i, w, R, y, lab_prod = columns(irf_res) 

    irf = (l=l, y=y, c=c, k=k, i=i, w=w, R=R,
                 lab_prod=lab_prod, η_x=100*log.(z))
    return irf
end


function residual(l_pol, simul, para::Para; burn_in=200)
    capT = size(simul.c)[1]
    @unpack A, α, θ, P = para
    @unpack k, z_indices = simul

    " Pre-allocate arrays "
    #rhs_fun = RHS_fun_cons(l_pol, para)
    rhs_fun = RHS_fun_cons(l_pol, para)

    " Right-hand side of Euler equation "
    #rhs = RHS_fun.(k, z_indices)
    rhs = rhs_fun.(k, z_indices)
    loss = 1.0 .- simul.c .* rhs
    return loss[burn_in:end]
end  

function impulse_response_plot(irf; fig_title="rbc_irf.pdf")
    fig, ax = subplots(1, 3, figsize=(20, 5))
    ax[1].plot(irf.c, label="c")
    ax[1].plot(irf.i, label="i")
    ax[1].plot(irf.l, label="l")
    ax[1].plot(irf.y, label="y")
    ax[1].set_title("Consumption, investment, output, and labor supply")
    ax[1].set_ylabel("%Δ")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(irf.w, label="w")
    ax[2].plot(irf.R, label="R")
    ax[2].set_title("Wage and rental rate of capital")
    ax[2].set_ylabel("%Δ")
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(irf.η_x, label="x")
    ax[3].plot(irf.lab_prod, label="Labor productivity")
    ax[3].plot(irf.k, label="Capital stock")
    ax[3].set_title("Total factor and labor productivity and capital stock")
    ax[3].legend()
    ax[3].set_ylabel("%Δ")
    ax[3].grid()
    ax[3].legend()
    plt.tight_layout()
    display(fig)
    PyPlot.savefig(fig_title)
end

   

para = Para()

" Calibrate "
cal = calibrate()
" Solve for steady state "
steady = steady_state(cal)

update_params!(para, cal)
@unpack NK, NS, A, k_grid, α = para

# Initialize labor supply
l = ones(NK, NS)*steady.l

l_mat, l_pol, c_pol, y_pol, inv_pol, w_pol, R_pol = solve_model_time_iter(l, para, verbose=true)
@btime solve_model_time_iter($l, $para, tol=1e-6)


# Plot policies
fig, ax = subplots(ncols=3, figsize=(16, 4))
for (i, policy) in enumerate([l_mat, c_pol, inv_pol])
    ax[i].plot(k_grid, policy[:, 1], label="low productivity", alpha=0.6)
    ax[i].plot(k_grid, policy[:, 9], label="high productivity", alpha=0.6)
    ax[i].set_xlabel(L"k")
    ax[i].grid()
    ax[i].legend()
end
ax[1].set_title("Labor policy")
ax[2].set_title("Consumption policy")
ax[3].set_title("Investment policy")
plt.tight_layout()
display(fig)
plt.savefig("Policies.pdf")

" Simulation "
simul = simulate_series(l_mat, para) 
@unpack c, k, l, i, w, R, y, lab_prod, η_x, z_indices = simul

" Log deviations from stationary mean "
#out = [100*log.(getfield(simul, x)./mean(getfield(simul,x))) for x in keys(steady)]
fields = [:y, :c, :i, :w, :R, :l, :lab_prod]
out = reduce(hcat, [100 .*log.(getfield(simul, x)./mean(getfield(simul,x))) for x in fields])
#l, c, k, y, i, w, R, lab_prod = columns(out)
#simul_dat = DataFrames.DataFrame(l=l, c=c, k=k, y=y, i=i, w=w, R=R, lab_prod=lab_prod)
simul_dat = DataFrames.DataFrame(out, :auto)
DataFrames.rename!(simul_dat, fields)

# Moments 
cycle = mapcols(col -> hamilton_filter(col, h=8), simul_dat)
fields_mom = [:y, :c, :i, :l, :w, :R]
#select!(cycle, fields)
# Extract 
mom_mod = moments(cycle, :y, [:y], var_names=fields)
print(mom_mod)


" Residuals "
res = residual(l_pol, simul, para)
res_norm = log10.(abs.(res))
@show mean(res_norm), maximum(res_norm)


" Simulated data"
fig, ax = subplots(1, 3, figsize=(20, 5))
t = 250:1000
ax[1].plot(t, simul_dat.c[t], label="c")
ax[1].plot(t, simul_dat.l[t], label="l")
ax[1].plot(t, simul_dat.i[t], label="i")
ax[1].plot(t, simul_dat.y[t], label="y")
ax[1].set_title("Consumption, investment, output, and labor supply")
ax[1].set_ylabel("%Δ")
ax[1].legend()

ax[2].plot(t, simul_dat.w[t], label="w")
ax[2].plot(t, simul_dat.R[t], label="R")
ax[2].set_title("Wage and rental rate of capital")
ax[1].set_ylabel("%Δ")
ax[2].legend()

ax[3].plot(t, 100*η_x[t], label="x")
ax[3].plot(t, simul_dat.lab_prod[t], label="labor productivity")
ax[3].set_title("Total factor and labor productivity")
ax[1].set_ylabel("%Δ")
ax[3].legend()
plt.tight_layout()
display(fig)
PyPlot.savefig("simulations.pdf")


" Impulse responses "
k_1 = mean(simul.k)
irf = impulse_response(l_mat, para, k_1, irf_length=60)
impulse_response_plot(irf)

# no persistence in productivity
para_np = Para(ρ_x = 0.0)
# re-solve model
l = ones(NK, NS)*steady.l
l_mat, l_pol, c_pol, y_pol, inv_pol, w_pol, R_pol = solve_model_time_iter(l, para_np, verbose=true)

# irf_np = impulse_response(l_mat, para_np, k_1, irf_length=10)
# impulse_response_plot(irf_np, fig_title="rbc_irf_np.pdf")


function capital_destruction(l_mat::Matrix{Float64}, para, k_init; irf_length=40, Δ=0.2)

    # Destruction shock to capital is completely unanticipated--not incorporated into agents' policies
    @unpack ρ_x, σ_x, P, mc, A, α, θ, δ, k_grid = para

    # Bivariate interpolation (AR(1) shocks, so productivity can go off grid)
    L = Spline2D(k_grid, A, l_mat)

    # assume TFP at steady-state throughout
    z_series = ones(irf_length)

    # One-time shock to capital
    k_bas = k_init
    k_shock = k_init*(1-Δ)

    function impulse(k_init)
        # trajectory given initial capital stock (fixed technology process)
        k = zeros(irf_length+1)
        k[1] = k_init
        l = zeros(irf_length)
        c = zeros(irf_length)
        y = zeros(irf_length)


        for t in 1:irf_length
            # labor
            l[t] = L(k[t], z_series[t])
            y[t] = z_series[t]*k[t]^α*l[t]^(1-α)
            c[t] = (1-l[t])/θ*(1-α)*y[t]/l[t]
            k[t+1] = (1-δ)*k[t] + y[t] - c[t]
        end

        #k = k[1:(end-1)]
        pop!(k)
        i = y - c
        w = (1-α).*y./l
        R = α.*y./k
        lab_prod = y./l
        out = [c k l i w R y lab_prod]
        return out
    end

    out_imp = impulse(k_shock) # series under shock to capital
    out_bas = impulse(k_bas) # baseline series

    irf_res = similar(out_imp)
    @. irf_res .= 100*(out_imp-out_bas)/out_bas
    #out = [log.(x./mean(getfield(simul, field))) for (x, field) in
    #zip([c, k[1:(end-1)], l, i, w, R, y, lab_prod], [:c, :k, :l, :i, :w, :R, :y, :lab_prod])]
    c, k, l, i, w, R, y, lab_prod = columns(irf_res) 

    irf = (l=l, y=y, c=c, k=k, i=i, w=w, R=R,
                 lab_prod=lab_prod)
    return irf
end

function capital_destruction_plot(irf; fig_title="rbc_irf_k_shock.pdf")
    fig, ax = subplots(1, 3, figsize=(20, 5))
    ax[1].plot(irf.c, label="c")
    ax[1].plot(irf.i, label="i")
    ax[1].plot(irf.l, label="l")
    ax[1].plot(irf.y, label="y")
    ax[1].set_title("Consumption, investment, output, and labor supply")
    ax[1].set_ylabel("%Δ")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(irf.w, label="w")
    ax[2].plot(irf.R, label="R")
    ax[2].set_title("Wage and rental rate of capital")
    ax[2].set_ylabel("%Δ")
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(irf.lab_prod, label="Labor productivity")
    ax[3].plot(irf.k, label="Capital stock")
    ax[3].set_title("Total factor and labor productivity and capital stock")
    ax[3].legend()
    ax[3].set_ylabel("%Δ")
    ax[3].grid()
    ax[3].legend()
    plt.tight_layout()
    display(fig)
    PyPlot.savefig(fig_title)
end

irf_k_shock = capital_destruction(l_mat, para, steady.k, irf_length=40, Δ=0.2)

capital_destruction_plot(irf_k_shock)