
using PyPlot
using BenchmarkTools
using LaTeXStrings
using Parameters, CSV, Random, QuantEcon
using NLsolve, LeastSquaresOptim
using LinearAlgebra, Roots, LinearInterpolations, Interpolations, Dierckx
using Printf
using DataFrames
#include("time_series_fun.jl")
cd(@__DIR__)
parent_dir = dirname(@__DIR__)
cd(parent_dir)
#include("time_series_fun.jl")

columns(M) = [view(M, :, i) for i in 1:size(M, 2)]

"""
    calibrate(r, α, l, δ, ρ_x, σ_x)
Calibrate RBC model given long-run targets.
"""


@with_kw mutable struct Para{T1, T2, T3, T4}
    # model parameters
    β::Float64 = 0.99
    δ::Float64 = 0.025
    χ::Float64 = 0.9241
    ψ::Float64 = 2.0
    ε::Float64 = 4.3
    f_E::Float64 = 1
    σ::Float64 = 2.0
    Z::Float64 = 1.0

    # numerical parameter
    N_l::Float64 = 1.0 # lower bound for number of firms
    N_u::Float64 = 15.0 # upper bound for number of firms
    NN::Int64 = 50 # number of gridpoints for number of firms
    NS::Int64 = 7 # number of elements of Markov chain for productivity

    ρ_x::Float64 = 0.979
    σ_x::Float64 = 0.0072
    mc::T1 = rouwenhorst(NS, ρ_x, σ_x, 0) # Markov chain object
    P::T2 = mc.p # Transition matrix
    A::T3 = exp.(mc.state_values) # Levels of productivity
    N_grid::T4 = range(N_l, stop=N_u, length=NN) # grid for number of firms
end

"""
    steady_state(params)
Calculate steady state of model
"""
function steady_state(para)
    @unpack β, δ, χ, ψ, ε, σ, Z, f_E = para

    r = (1-β)/β
    Γ = ε/(ε-1)

    function from_L(L)
        N = (1-δ)*Z*L/(f_E*(r+δ)*(ε-1)+δ)
        ρ = N^(1/(ε-1))
        w = ρ*Z/Γ
        L_C = (ε-1)*(r+δ)*L/((ε-1)*(r+δ)+δ)
        C = Z*ρ*L_C
        loss = χ*L^(1/ψ)-w*C^(-σ)
        return loss, N, ρ, L_C, C, w
    end

    f(L) = from_L(L)[1]
    L = fzero(f, (1e-4, 20.0))

    loss, N, ρ, L_C, C, w = from_L(L)
    # Obtain remaining variables

    # Entrants
    N_E = δ*N/(1-δ)
    # Individual profits
    d = C/(ε*N)
    # Firm value
    ν = w*f_E/Z
    # Aggregate output 
    Y = C + ν*N_E
    # Aggregate profits 
    D = N*d
    # Individual output 
    y = C/(N*ρ)
    # Investment
    I = ν*N_E
    SteadyState = (C=C, Y=Y, N_E=N_E, N=N, d=d, ν=ν, w=w, L=L, L_C=L_C,
    ρ=ρ, y=y, D=D, I=I)
    return SteadyState
end



# function update_params!(self, cal)
#     @unpack α, β, δ, θ = cal
#     self.α = α
#     self.β = β
#     self.δ = δ
#     self.θ = θ
#     nothing
# end

"""
    RHS_fun_cons(C_pol, I_pol, para, numparams)
Compute the conditional expectation at each state (K,N,z) for a given labor supply policy
"""
function update_variables(C, N, Z, para)
    @unpack  β, δ, χ, ψ, ε, σ, f_E, A = para
    @unpack N_l, N_u = para
    Γ = ε/(ε-1)
    ρ = N^(1/(ε-1))
    
    w = ρ*Z/Γ
    L_C = C/(Z*ρ)

    L = (w*C^(-σ)/χ)^ψ
    ν = w*f_E/Z
    d = C/(N*ε)
    N_E = max((w*L+N*d-C)/ν, 0)  
    # Enforce bounds   
    N_p = min((1-δ)*(N+N_E), N_u)
    N_p = max(N_p, N_l)
    return ρ, C, L_C, w, L, ν, d, N_E, N_p
end


function interpolating_function(mat::Array, para)
    @unpack N_grid = para
     # Create interpolating functions given C_mat, I_mat
     pol(N, z) = LinearInterpolation(N_grid, @view(mat[:, z]), extrapolation_bc=Line())(N)
    return pol
end

function RHS_EE_cons(C_pol, para::Para)
    @unpack  β, δ, χ, ψ, ε, σ, Z, f_E = para
    @unpack P, NN, NS, N_grid, A = para

    # consumption given state and labor
    
    RHS = zeros(NN, NS)
    #@inbounds Threads.@threads for (i, K) in collect(enumerate(K_grid))
    for z in 1:NS
    #@inbounds Threads.@threads for z in 1:NS
        for (i, N) in enumerate(N_grid)
        # relative price
            # consumption and investment policies
            C = C_pol(N, z)
            ρ, C, L_C, w, L, ν, d, N_E, N_p = update_variables(C, N, A[z], para)
            #N_p = min((1-δ)*(N+N_E), maximum(N_grid))

            for z_hat in 1:NS #possible future technology (determined by Markov chain)
                # update consumption and investment
                C_p = C_pol(N_p, z_hat)
                ρ_p, C_p, L_C_p, w_p, L_p, ν_p, d_p, N_E_p, _ = update_variables(C_p, N_p, A[z_hat], para)

                RHS[i, z] += P[z, z_hat]*(1-δ)*C_p^(-σ)*(ν_p+d_p)/ν
            end
        end
    end

    RHS .= (β.*RHS).^(-1/σ) # C
    return RHS
end


function initialize(para)
    @unpack NN, NS, A, N_grid = para
    C_mat = zeros(NN, NS)
    for z in 1:NS
        for i in 1:NN
            C_mat[i, z] = ss.C*A[z]*N_grid[i]/ss.N
        end
    end
    return C_mat
end



"""
    solve_model_time_iter(l, para::Para, tol=1e-8, max_iter=1000, verbose=true, print_skip=25)
Solve RBC model (find correct policies) given initial guess of labor supply policy
"""
function solve_model_time_iter(C_mat::Array, para::Para; ω=0.7, tol=1e-7, max_iter=1000, verbose=true, 
                                print_skip=25)
    # Set up loop 
    @unpack  β, δ, χ, ψ, ε, σ, Z, f_E = para
    @unpack P, NN, NS, N_grid, A = para

    ss = steady_state(para)

    # Initialize arrays for updating policies
    C_new = similar(C_mat)

    error = 1.0
    iter = 1.0
    while (iter < max_iter) && (error > tol)
    
        C_pol = interpolating_function(C_mat, para)
    
        rhs =  RHS_EE_cons(C_pol, para)
        #x0 = [ss.C, ss.I]
        #@inbounds Threads.@threads for (i, K) in collect(enumerate(K_grid))
        
        for z in 1:NS
            for (i, N) in enumerate(N_grid)
                C_new[i,  z] = rhs[i, z]
            end
        end
      
        error = mean(abs.(C_new-C_mat)./abs.(C_mat))
        if verbose && iter % print_skip == 0
            print("Error at iteration $iter is $error. \n")
        end
        iter += 1
        # update arrays
        C_mat .= @. ω*C_new + (1-ω)*C_mat
    end

    # Get convergence level
    if iter == max_iter
        print("Failed to converge!")
    end

    if verbose && (iter < max_iter)
        print("Converged in $iter iterations")
    end
    ρ = @. N_grid^(1/(ε-1))
    A_mat = similar(C_mat)

    for z in 1:NS
        A_mat[:, z] .= A[z]
    end

    Γ = ε/(ε-1)

    L_C = @. (C_mat/(ρ*A_mat))
    w = @. ρ*A_mat/Γ
    L = @. (w*C_mat^(-σ)/χ)^ψ
    ν = @. w*f_E/A_mat
    d = @. C_mat/(N_grid*ε)
    N_E = @. max((w*L+N_grid*d-C_mat)/ν, 0)     

    C_pol = interpolating_function(C_mat, para)
    return C_mat, C_pol, L_C, L, w, ν, d, N_E
end


function simulate_series(C_mat::Array, para::Para; burn_in=200, capT=10_000)

    @unpack  β, δ, χ, ψ, ε, σ, Z, f_E = para
    @unpack P, mc, NN, NS, N_grid, A = para

    C_pol = interpolating_function(C_mat, para)
    capT = capT + burn_in + 1

    # Extract indices of shocks
    z_indices = simulate_indices(mc, capT)
    z_series = A[z_indices]
    
    # Initialize arrays
    ss = steady_state(para)
    N = ones(capT+1)*ss.N
    C = similar(z_series)
    var = ones(capT, 7)

    ρ, L_C, w, L, ν, d, N_E = columns(var)

    for t in 1:capT
        C[t] = C_pol(N[t], z_indices[t])
        out = update_variables(C[t], N[t], z_series[t], para)
        ρ[t], C[t], L_C[t], w[t], L[t], ν[t], d[t], N_E[t], N[t+1] = out
    end

    pop!(N)
    I = @. ν*N_E
    Y = @. C + I
    lab_prod = @. Y/L
    L_E = L - L_C

    Simulation = (ρ=ρ, C=C, N=N, Y=Y, lab_prod=lab_prod, w=w, 
    L_C=L_C, L_E=L_E, L=L, ν=ν, d=d, N_E=N_E, I=I, η_x=log.(z_series), z_indices=z_indices)
    return Simulation
end


"""
    impulse_response(l_mat, para, k_init; irf_length=40, scale=1.0)
Calculate impulse response to productivity shock 
"""
function impulse_response(C_mat::Array, para, N_init; irf_length=40, scale=1.0, perc=false)

    @unpack  β, δ, χ, ψ, ε, σ, Z, f_E = para
    @unpack P, mc, NN, NS, N_grid, A = para
    @unpack ρ_x, σ_x = para

    # 2-dimensional interpolation (AR(1) shocks, so productivity can go off grid)
    
    #C_fun = LinearInterpolation((N_grid, A), C_mat)
    C_fun = Spline2D(N_grid, A, C_mat)
    η_x = zeros(irf_length)
    η_x[1] = σ_x*scale
    if perc
        η_x[1] = scale/100
    end

    for t in 1:(irf_length-1)
        η_x[t+1] = ρ_x*η_x[t]
    end
    z = exp.(η_x)
    z_bas = ones(irf_length)

    function impulse(z_series)
        N = zeros(irf_length+1)
        C = zeros(irf_length)
        var = ones(irf_length, 7)

        ρ, L_C, w, L, ν, d, N_E = columns(var)

        N[1] = N_init

        for t in 1:irf_length
            C[t] = C_fun(N[t], z_series[t])
            out = update_variables(C[t], N[t], z_series[t], para)
            ρ[t], C[t], L_C[t], w[t], L[t], ν[t], d[t], N_E[t], N[t+1] = out
        end

        pop!(N)
        I = @. ν*N_E
        Y = @. C + I
        lab_prod = @. Y/L
        L_E = L - L_C


        out =   [N C I ρ L_C L_E w L ν d N_E Y lab_prod]
        return out
    end

    out_imp = impulse(z) # collect values under impulse
    out_bas = impulse(z_bas) # collect baseline values (no shock occurs)

    irf_res = similar(out_imp)
    @. irf_res .= 100*log(out_imp/out_bas)
    #out = [log.(x./mean(getfield(simul, field))) for (x, field) in
    #zip([c, k[1:(end-1)], l, i, w, R, y, lab_prod], [:c, :k, :l, :i, :w, :R, :y, :lab_prod])]
    N, C, I, ρ, L_C, L_E, w, L, ν, d, N_E, Y, lab_prod = columns(irf_res) 

    irf = (N=N, C=C, I=I, ρ=ρ, L_C=L_C, L_E=L_E, w=w, L=L, ν=ν, d=d, N_E=N_E, Y=Y, lab_prod=lab_prod,  η_x=100*log.(z))
    return irf
end


function residual(C_pol, sim, para::Para; burn_in=200)
    capT = size(sim.C)[1]
    @unpack  α, β, δ, χ, ψ, ε, σ, Z, f_E = para
    @unpack P, mc, NN, NS, N_grid, A = para

    @unpack C, N, z_indices = sim

    " Pre-allocate arrays "
    #rhs_fun = RHS_fun_cons(l_pol, para)
    rhs = RHS_EE_cons(C_pol, para)
    rhs_fun = interpolating_function(rhs, para)

    rhs_array = rhs_fun.(N, z_indices)
    loss = zeros(1, capT)
    loss[1, :] .= @. 1.0 - rhs_array/C
    return loss[:,burn_in:end]
end  

function simulation_plot(sim)
    " Simulated data"
    fig, ax = subplots(2, 2, figsize=(12, 12))
    t = 250:1000
    ax[1,1].plot(t, sim.C[t], label="C")
    ax[1,1].plot(t, sim.I[t], label="I")
    ax[1,1].plot(t, sim.Y[t], label="Y")
    ax[1,1].plot(t, sim.L[t], label="L")
    ax[1,1].set_title("Output measures and labor supply")
    ax[1,1].set_ylabel("%Δ")
    ax[1,1].legend()

    ax[1,2].plot(t, sim.w[t], label="w")
    ax[1,2].plot(t, sim.ν[t], label="r_K")
    ax[1,2].set_title("Wage and value of firm")
    ax[1,2].set_ylabel("%Δ")
    ax[1,2].legend()

    ax[2,1].plot(t, sim.L_C[t], label="L_C")
    ax[2,1].plot(t, sim.L_E[t], label="L_E")
    ax[2,1].plot(t, sim.L[t], label="L")
    ax[2,1].set_title("Labor in each sector")
    ax[2,1].set_ylabel("%Δ")
    ax[2,1].legend()

    ax[2,2].plot(t, sim.η_x[t], label="x")
    ax[2,2].plot(t, sim.lab_prod[t], label="labor productivity")
    ax[2,2].plot(t, sim.N[t], label="Capital")
    ax[2,2].set_title("Productivity and number of firms")
    ax[2,2].set_ylabel("%Δ")
    ax[2,2].legend()
    plt.tight_layout()
    display(fig)
    PyPlot.savefig("simulations.pdf")
end

function impulse_response_plot(irf; fig_title="BGM_irf.pdf")
    fig, ax = subplots(2, 2, figsize=(12, 8))
    ax[1,1].plot(irf.C, label="C")
    #ax[1].plot(irf.I, label="I")
    ax[1,1].plot(irf.N_E, label="N_E")
    ax[1,1].plot(irf.Y, label="Y")
    ax[1,1].plot(irf.L, label="L")
    ax[1,1].set_title("Consumption, entry, output, and labor supply")
    ax[1,1].set_ylabel("%Δ")
    ax[1,1].grid()
    ax[1,1].legend()

    ax[1,2].plot(irf.w, label="w")
    ax[1,2].plot(irf.ν, label="ν")
    ax[1,2].set_title("Wage and firm value")
    ax[1,2].set_ylabel("%Δ")
    ax[1,2].grid()
    ax[1,2].legend()

    ax[2,1].plot(irf.L_C, label="L_C")
    ax[2,1].plot(irf.L_E, label="L_E")
    ax[2,1].plot(irf.L, label="L")
    ax[2,1].set_title("Labor composition")
    ax[2,1].legend()
    ax[2,1].set_ylabel("%Δ")
    ax[2,1].grid()

    ax[2,2].plot(irf.η_x, label="x")
    ax[2,2].plot(irf.lab_prod, label="Labor productivity")
    ax[2,2].plot(irf.N, label="Number of firms")
    ax[2,2].set_title("Total factor and labor productivity and number of firms")
    ax[2,2].legend()
    ax[2,2].set_ylabel("%Δ")
    ax[2,2].grid()
  
    plt.tight_layout()
    display(fig)
    PyPlot.savefig(fig_title)
end



para = Para(ε=3.8, ψ=4.0, σ=1.0,  N_l=1, N_u=15)
ss = steady_state(para)
C_mat= initialize(para)
out = solve_model_time_iter(C_mat, para, max_iter=1000, ω=0.5)

C_mat, C_pol, L_C, L, w, ν, d, N_E = out
sim = simulate_series(C_mat, para, burn_in=200, capT=10_000)


irf = impulse_response(C_mat, para, ss.N, irf_length=60, perc=true)
impulse_response_plot(irf, fig_title="BGM_irf.pdf")


" Residuals "
res = residual(C_pol, sim, para, burn_in=200)
res_norm = log10.(abs.(res))
@show mean(res_norm), maximum(res_norm)

########### Low δ ##############################################################
