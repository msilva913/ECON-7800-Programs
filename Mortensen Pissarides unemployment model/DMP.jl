
using PyPlot
using Plots
using BenchmarkTools
using LaTeXStrings
using Parameters, CSV, Random, QuantEcon
using Distributions
using LinearAlgebra, Roots, Optim, LinearInterpolations, Interpolations
using Dierckx
using Printf, PrettyPrint
using DataFrames
include("time_series_fun.jl")

#job finding probability
function jf(θ, η_L)
    return θ/(1+θ^(η_L))^(1/η_L)
end

#vacancy filling probability
function vf(θ, η_L)
    return 1/(1+θ^η_L)^(1/η_L)
end

function theta_invert(q, η_L)
    θ = (1/q^(η_L) - 1)^(1/η_L)
    return θ
end

function columns(M)
    return (view(M, :, i) for i in 1:size(M, 2))
end

"""
    grid_cons_grow(n, left, right, g)
Create n+1 gridpoints with growing distance on interval [a, b]
according to the formula
x_i = a + (b-a)/((1+g)^n-1)*((1+g)^i-1) for i=0,1,...n 
"""
function grid_cons_grow(n, left, right, g)
  
    x = zeros(n)
    for i in 0:(n-1)
        x[i+1] = @. left + (right-left)/((1+g)^(n-1)-1)*((1+g)^i-1)
    end
    return x
end


function calibrate(;β=exp(-4.0/1200.0), α_L=0.5, z=0.71, η_L=1.25,
    s=0.035, u_target=0.058, ρ_x=0.95^(1/3), stdx=0.00625)

    r = 1/β - 1.0
    function err(x, k)
        return k/vf(x, η_L) - ((1-α_L)*(1-z)-α_L*k*x)/(r+s)
    end

    function loss(k)
        θ = find_zero(x -> err(x, k), (0.0, 10.0), Bisection())
        f = jf(θ, η_L)
        return s/(s+f) - u_target
    end

    k = find_zero(loss, (0.1, 1), Bisection())
    CalibratedParameters = (k=k, α_L=α_L, z=z, s=s, η_L=η_L, β=β, ρ_x=ρ_x, stdx=stdx)
    return CalibratedParameters
end


function steady_state(para)
    @unpack k, α_L, z, s, η_L, β, ρ_x, stdx = para
    x = 1.0
    r = 1/β - 1.0

    function err(x, k)
        return k/vf(x, η_L) - ((1-α_L)*(1-z)-α_L*k*x)/(r+s)
    end

    θ = find_zero(x ->err(x, k), (0.01, 10), Bisection())
    f = jf(θ, η_L)
    q = f/θ
    u = s/(s+f)
    w = α_L*(x+k*θ) + (1-α_L)*z
    v = θ*u
    # Average hiring cost/ expected value of a filled job
    E = k/q

    J = x-w + (1-s)*k/q
    M = (1-u)*J
    Y = (1-u)*x

    SteadyState = (E=E, θ=θ, q=q, f=f, u=u, w=w, v=v, M=M, Y=Y)
    return SteadyState
end


@with_kw struct Para{T1, T2, T3, T4}
    # model parameters
    k::Float64 
    α_L::Float64
    z::Float64
    s::Float64
    η_L::Float64
    β::Float64
    ρ_x::Float64
    stdx::Float64

    # numerical parameter
    u_low::Float64 = 0.02
    u_high::Float64 = 0.35
    max_iter::Int64 = 1000
    NU::Int64 = 50
    NS::Int64 = 20
    T::Float64 = 1e5
    mc::T1 = rouwenhorst(NS, ρ_x, stdx, 0)
    P::T2 = mc.p
    A::T3 = exp.(mc.state_values)
    u_grid::T4 = grid_cons_grow(NU, u_low, u_high, 0.02)
end


function get_policies(E, u, x, para)
    " obtain policies from right-hand side of Euler equation "

    @unpack k, α_L, z, s, η_L, β, NU, NS, u_grid, A, u_low, u_high = para
    q = k/E
    q = min(q, 1.0)
    θ = theta_invert(q, η_L)
    f = θ*q
    v = θ*u 
    w = α_L*(x+k*θ) + (1-α_L)*z
    u_p = s*(1-u) + (1-f)*u
    # enforce bounds
    u_p = min(u_p, u_high)
    u_p = max(u_p, u_low)

    J = x-w + (1-s)*k/q
    M = (1-u)*J
    Y = (1-u)*x
    return q, θ, f, v, w, M, Y, u_p
end

function rhs_jcc(E_pol, iu, ix, para)
    # Right-hand side for particular state
    @unpack k, α_L, z, s, η_L, β, NU, NS, u_grid, A, P = para
    # Reconstruct right-hand side of Euler
    u = u_grid[iu]
    x = A[ix]
    # current right-hand side
    E = E_pol(u, ix)
    # extract policies and next-period unemployment
    q, θ, f, v, w, M, Y, u_p= get_policies(E, u, x, para)
    E_new = 0.0
    @inbounds @simd for ix_p in 1:NS
        x_p = A[ix_p]
        E_p = E_pol(u_p, ix_p)
        q_p, θ_p, f_p, v_p, w_p, M_p, Y_p, u_p2 = get_policies(E_p, u_p, x_p, para)
        # add job surplus under realization ix_p
        E_new += P[ix, ix_p]*β*(x_p-w_p + (1-s)*(k/q_p))
    end
    return E_new
end


function solve_model_time_iter(E_mat, para; tol=1e-7, max_iter=1000, verbose=true, 
                                print_skip=25, ω=0.7)
    # Set up loop 
    @unpack k, α_L, z, s, η_L, β, NU, NS, u_grid, A, P = para
    
    err = 1
    iter = 1
    while (iter < max_iter) && (err > tol)
        E_pol(u, ix) = Interpolate(u_grid, @view(E_mat[:, ix]), extrapolate=:reflect)(u)
        # interpolate given grid on EE
        E_new = zeros(NU, NS)
        @inbounds for (iu, u) in collect(enumerate(u_grid))
            for ix in 1:NS
                # new right-hand side of EE given iu, ix
                E_new[iu, ix] = rhs_jcc(E_pol, iu, ix, para)
            end
        end
        E_new .= ω*E_new +(1-ω)*E_mat
        err = maximum(abs.(E_new-E_mat)/max.(abs.(E_mat), 1e-10))
        if verbose && iter % print_skip == 0
            print("Error at iteration $iter is $err.")
        end
        iter += 1
        # update grid of rhs of EE
        E_mat = E_new
    end
    E_pol(u, ix) = Interpolate(u_grid, @view(E_mat[:, ix]), extrapolate=:reflect)(u)
    # Get convergence level
    if iter == max_iter
        print("Failed to converge!")
    end

    if verbose && (iter < max_iter)
        print("Converged in $iter iterations")
    end
    # Get remainding variables
    # Productivity on (NS, NU) grid
    X = repeat(A, 1, NU)'
    q = k./E_mat
    q = min.(q, 1.0)
    θ = theta_invert.(q, η_L)
    f = θ.*q
    v = θ.*u_grid
    w =  α_L.*(X.+k.*θ) .+ (1-α_L)*z

    return E_mat, E_pol, X, q, f, v, w
end


function simulate_series(E_mat, para, burn_in=200, capT=10000)

    @unpack k, α_L, z, s, η_L, β, ρ_x, stdx, NU, NS, A, mc, u_grid = para
    capT = capT + burn_in + 1

    # bivariate interpolation
    E_pol = Spline2D(u_grid, A, E_mat)

    # draw shocks
    η_x = randn(capT).*stdx
    x = zeros(capT)
    for t in 1:(capT-1)
        x[t+1] = ρ_x*x[t] + η_x[t+1]
    end
    x = exp.(x)

    u = ones(capT+1)*0.05
    vars = ones(capT, 8)
    f, θ, v, w, M, Y, E = columns(vars)
    
    for t in 1:capT
        # interpolate E
        E[t] = E_pol(u[t], x[t])
        # recover policies and next-period unemployment from interpolation
        __, θ[t], f[t], v[t], w[t], M[t], Y[t], u[t+1] = get_policies(E[t], u[t], x[t], para)
    end
    # remove last element of u
    pop!(u)
    # remove burn-in
    out = [θ f v w u x M Y][(burn_in+1):end, :]
    θ, f, v, w, u, x, M, Y = columns(out)
    labor_share = w./x

    Simulation = (θ=θ, f=f, v=v, w=w, u=u, x=x, M=M, Y=Y, labor_share=labor_share)
    return Simulation
end


function impulse_response(E_mat, para; u_init=0.07, irf_length=60, scale=1.0)

    @unpack k, α_L, z, s, η_L, β, ρ_x, stdx, NU, NS, A, mc, u_grid = para

    # bivariate interpolation
    E_pol = Spline2D(u_grid, A, E_mat)

    # draw shocks
    x = zeros(irf_length)
    x[1] = stdx*scale
    for t in 1:(irf_length-1)
        x[t+1] = ρ_x*x[t]
    end
    x = exp.(x)
    # counterfactual
    x_bas = ones(irf_length)

    function impulse(x_series)
        u = ones(irf_length+1)
        u[1] = u_init
        vars = ones(irf_length, 7)
        f, θ, v, w, M, Y, E = columns(vars)
        
        for t in 1:irf_length
            # interpolate E
            E[t] = E_pol(u[t], x_series[t])
            # recover policies and next-period unemployment from interpolation
            __, θ[t], f[t], v[t], w[t], M[t], Y[t], u[t+1] = get_policies(E[t], u[t], x_series[t], para)
        end
        # remove last element of u
        pop!(u)
        labor_share = w./x_series

        out = [θ f v w u x_series M Y labor_share]
        return out
    end

    out_imp = impulse(x)
    out_bas = impulse(x_bas)

    irf_res = similar(out_imp)
    @. irf_res = 100*log(out_imp/out_bas)
    θ, f, v, w, u, x, M, Y, labor_share = columns(irf_res)
    irf = (θ=θ, f=f, v=v, w=w, u=u, x=x, M=M, Y=Y, labor_share=labor_share)
    return irf
end

   
# calibrate model
cal = calibrate(stdx=0.015)
@unpack k, α_L, z, s, η_L, β, ρ_x, stdx = cal

ss = steady_state(cal)
# Stock market cap
ss.M/ss.Y 

# form struct using calibrated parameters
para = Para(k=k, α_L=α_L, z=z, s=s, η_L=η_L, β=β, ρ_x=ρ_x, stdx=stdx)
@unpack NU, NS, u_grid = para

# initial guess of rhs jcc
E_mat = zeros(NU, NS)
E_mat .= k/ss.q

# Solve model: iterating on the job creation condition
E_mat, E_pol, X, q, f, v, w = solve_model_time_iter(E_mat, para)

# Simulate model
simul = simulate_series(E_mat, para)

fields = [:x, :u, :θ, :Y, :M, :labor_share]
# Log deviations from stationary mean
out = reduce(hcat, [100 .*log.(getfield(simul, x)./mean(getfield(simul,x))) for x in fields])

# Create time series object
df = time_series_object(out, fields);

# convert to quarterly
df_q = collapse(df, eoq(df.index), fun=mean);
#cycle = mapcols(col -> hamilton_filter(col, h=8), DataFrames.DataFrame(df_q.values,:auto))
cycle = mapcols(col -> hp_filter(col, 10^5)[1], DataFrames.DataFrame(df_q.values,:auto))

DataFrames.rename!(cycle, fields)

#Extract moments
mom_mod = moments(cycle, :x, fields, var_names=fields)
pprint(mom_mod)



fig, ax = subplots(1, 3, figsize=(20, 5))
t = 250:1000
ax[1].plot(t, simul.x[t], label="x")
ax[1].plot(t, simul.w[t], label="w")
ax[1].set_title("Subplot a: Productivity and wages")
ax[1].legend()

ax[2].plot(t, simul.f[t], label="f")
ax[2].plot(t, simul.θ[t], label="θ")
ax[2].set_title("Subplot b: Job finding rate and tightness")
ax[2].legend()

ax[3].plot(t, simul.u[t], label="u")
ax[3].set_title("Subplot c: Unemployment")
ax[3].legend()
display(fig)
PyPlot.savefig("simulations.pdf")

" Impulse responses "
irf =impulse_response(E_mat, para, u_init=ss.u)

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[1].plot(irf.θ, label="θ")
    ax[1].plot(irf.u, label="u")
    ax[1].legend(loc="best")
    ax[1].set_title("Market tightness and vacancies")
    
    ax[2].plot(irf.M, label="M")
    ax[2].plot(irf.Y, label="Y")
    ax[2].legend(loc="best")
    ax[2].set_title("Output and the stock market cap")
    
    ax[3].plot(irf.x, label="x")
    ax[3].plot(irf.w, label="w")
    ax[3].plot(irf.labor_share, label="labor_share")
    ax[3].legend(loc="best")
    ax[3].set_title("Productivity shock, wages, and the labor share")
    tight_layout()
    display(fig)
    PyPlot.savefig("irfs.pdf")

" Beveridge curve "
fig, ax = plt.subplots()
ax.scatter(x=simul.u, y=simul.θ, alpha=0.6, s=0.3)
ax.set_title("Beveridge curve")
ax.set_xlabel("u")
ax.set_ylabel("θ")
ax.legend()
tight_layout()
display(fig)
PyPlot.savefig("Beveridge_curve.pdf")






      
