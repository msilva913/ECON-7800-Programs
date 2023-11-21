
using PyPlot
using Parameters, CSV, Random, QuantEcon
using LinearAlgebra,  LinearInterpolations
using BenchmarkTools
using DataFrames
using Printf

"""
    hist(x, binranges)
Map values to histogram bins
"""
function histc(x, binranges)
    indices = searchsortedlast.(Ref(binranges), x)
    return indices
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

@with_kw mutable struct Para{T1, T2, T3}

    r::Float64 = 0.01   # real interest rate
    β::Float64 = 0.96   # discount factor
    w::Float64 = 1      # Real wage
    γ::Float64 = 2.0      # CRRA parameter
    b::Float64 = 0.0      # Exogenous borrowing constraint
    ρ::Float64 = 0.6      # AR(1) persistence parameter
    σ::Float64 = 0.16^(0.5)    # Conditional standard deviation
    NS::Int64 = 7      # Number of states in Markov chain
    grid_max::Float64 = 100.0
    NA::Int64 = 200
    u_a::Float64 = 0.01
    A::Float64 = 1      # Level parameter in production function
    N::Float64 = 1      # Employment level
    α::Float64 = 0.36   # Elasticity of output with respect to capital
    δ::Float64 = 0.08   # depreciation rate
    R::Float64 = 1 +r
    u = (c, γ=γ) -> c^(1-γ)/(1-γ)
    u_prime = c -> c^(-γ)
    u_prime_inv = c -> c^(-1/γ)
    mc::T1 = rouwenhorst(NS, ρ, σ*(1-ρ^2)^(1/2), 0)
    P::T2 = mc.p
    y::Vector{Float64} = exp.(mc.state_values)
    @assert R* β < 1
    a::T3 = grid_cons_grow(NA, -b, grid_max, 0.02)
end

"""
Changes dependent fields in a mutable struct instance para
"""
function update_params!(self)
    @unpack r, γ, σ, ρ, NS, β, b, grid_max, NA = self
    self.R = 1+r
    self.u = c -> c^(1-γ)/(1-γ)
    self.u_prime = c -> c^(-γ)
    self.u_prime_inv = c -> c^(-1/γ)
    self.mc = rouwenhorst(NS, ρ, σ*(1-ρ^2)^(1/2))
    self.P = self.mc.p
    self.y = exp.(self.mc.state_values)
    @assert self.R* β < 1
    self.a = grid_cons_grow(NA, -b, grid_max, 0.02)
    nothing
end



### Basic functions
"""
    r_to_w(r, para)
Equilibrium wages associated with interest rate r 
"""
function r_to_w(r, para)
    @unpack A, α, δ = para
    return A*(1-α)*(A*α/(r+δ))^(α/(1-α))
end

"""
    rd(K, para)
Inverse demand curve for capital: interest rate associated with given level of capital K
"""
function rd(K, para)
    @unpack A, α, δ, N = para
    return A*α*(N/K)^(1-α)-δ
end


"""
    d(r, para)
Demand for capital at interest rate r
"""
function d(r, para)
    @unpack A, α, δ, N = para
    return N*(A*α/(r+δ))^(1/(1-α))
end


"""
    time_iter(a_prime, para; omega=0.7)
The Coleman operator
a_prime: current guess of next-period assets
mod: model instance
"""
function time_iter(a_prime, para; omega=0.7)
    @unpack σ, R, P, y, β, w, u_prime, u_prime_inv, a, b, NS = para
    # Convert policy into function by linear interpolation
    function a_prime_fun(a_i, z)
        return Interpolate(a, @view(a_prime[:, z]), extrapolate=:reflect)(a_i)
    end

    expect = zero(a_prime)
    c = zero(a_prime)
    a_prime_new = zero(a_prime)
    # Current state a, z
    for z in 1:NS
        for (i, a_i) in enumerate(a)
            for z_hat in 1:NS
                a1 = a_prime_fun(a_i, z) #a'
                a2 = a_prime_fun(a1, z_hat) #a''
                #Implied next-period consumption
                c_prime = w*y[z_hat] + R*a1 - a2
                # Right-hand side of Euler equations
                expect[i, z] += β*R*u_prime(c_prime)*P[z, z_hat]
            end
            # Implied current consumption
            c[i, z] = u_prime_inv(expect[i, z])
            # Update of next-period assets
            a_prime_new[i, z] = w*y[z] + R*a_i - c[i, z]
        end
    end
    # Dampening
    a_prime_new = omega*a_prime_new + (1-omega)*a_prime
    # Consistency with borrowing constraint
    a_prime_new = max.(a_prime_new, -b)
    return a_prime_new, c
end


"""
    solve_model_time_iter(a_prime, para; tol=1e-7, max_iter=1000, verbose=true, 
    print_skip=25, omega=0.6)
Solve for optimal consumption and savings policy
"""
function solve_model_time_iter(a_prime, para; tol=1e-7, max_iter=1000, verbose=true, 
                                print_skip=25, omega=0.6)
    # Set up loop 
    i = 1
    error = tol + 1
    c = similar(a_prime)
    while (i < max_iter) && (error > tol)
        a_prime_new, c = time_iter(a_prime, para, omega=omega)
        error = maximum(abs.(a_prime-a_prime_new))
        i += 1
        if verbose && i % print_skip == 0

            print("Error at iteration $i is $error.")
        end
        a_prime = a_prime_new
    end

    if i == max_iter
        print("Failed to converge!")
    end

    if verbose && (i < max_iter)
        print("Converged in $i iterations")
    end
    return a_prime, c
end


"""
    comp_statics_plot(field, vars, para)
General comparative statics function for asset holdings (to assess variation with, say, interest rate or wage)
"""
function comp_statics_plot(field, vals, para)
    " How do savings and aggregate asset holdings vary with the interest rate and wage? "
    fig, ax = subplots()
    # convert to dictionary
    par = deepcopy(para)
    for val in vals
        # Set interest rate in model instance
        # change the model instance with field = val
        setfield!(par, field, val)
        # update dependent parameters
        update_params!(par)
        #update_params(para)
        @unpack P, a = par
        z_size = size(P)[1]
        a_prime_init = repeat(a, 1, z_size)
        # Solve model
        a_prime_star, c_star = solve_model_time_iter(a_prime_init, par, verbose=false)
        ax.plot(a, c_star[:, 1], label= "$field=$val", alpha=0.6)
    end
    ax.set(xlabel="asset level", ylabel = "consumption (low income)")
    ax.legend()
    display(fig)
end

"""
    update_dist(phi, ab_pol, wei, para)
Update distribution given transition savings policy.
"""
function update_dist(phi, ab_pol, wei, para)
    # update distribution
    @unpack P, NA, NS = para
    phi_new = zero(phi)
    for is in 1:NS
        for ia in 1:NA
            # lower gridpoint on savings
            a_p = ab_pol[ia, is]
            for is_p in 1:NS
                phi_new[a_p, is_p] += (1-wei[ia, is])*P[is, is_p]*phi[ia, is]
                phi_new[a_p+1, is_p] += wei[ia, is]*P[is, is_p]*phi[ia, is]
            end
        end
    end
    return phi_new
end

"""
    invariant_dist!(phi, ab_pol, wei, para; tol_dist=1e-9)
Compute stationary distribution over assets and productivities
"""
function invariant_dist!(phi, ab_pol, wei, para; tol_dist=1e-9)
    @unpack NA, NS = para
    phi_new = similar(phi)
    dif = 1.0
    while dif > tol_dist
        # update distribution
        phi_new .= update_dist(phi, ab_pol, wei, para)
        # check convergence
        dif = maximum(abs.(phi_new-phi))
        # ensure probabilities sum to 1
        phi .= phi_new./sum(phi_new)
    end
end

"""
    sum_stats(phi, a_pol, c_pol, para)
Compute summary statistics from asset distribution
"""
function sum_stats(phi, a_pol, c_pol, para)
    @unpack a, NA, NS = para

    # asset distribution
    asset_probs = sum(phi, dims=[2])
    # capital supply
    K = 0.0
    # aggregate consumption
    C = 0.0
    K  = sum(asset_probs.*a)
    for ia in 1:NA 
        for is in 1:NS
            C += c_pol[ia, is]*phi[ia, is]
            #K += a_pol[ia, is]*phi[ia, is]
        end
    end
    # standard deviation of consumption
    c_var = zeros(NA, NS)
    a_var = zeros(NA, NS)
    for ia in 1:NA
        for is in 1:NS
            c_var[ia, is] = (c_pol[ia, is]-C)^2*phi[ia, is]
            a_var[ia, is] = (a[ia]-K)^2*phi[ia, is]
        end
    end
    # standard deviations
    std_C = sqrt(sum(c_var))
    std_K = sqrt(sum(a_var))

    CV_C = std_C/C*100
    CV_K = std_K/K*100
    return asset_probs, C, K, CV_C, CV_K
end


"""
    general_equilibrium(para)
Apply fixed point iteration of r to find level of capital at which demand equals supply
"""
function general_equilibrium(para; T=100_000)
    """
 
    """
    @unpack a, grid_max, b, δ, β, NA, NS = para
    r_max = (1-β)/β
    # initial guess of savings
    a_pol = 0.8.*repeat(a, 1, NS)
    c_pol = similar(a_pol)
    # Generate fixed Markov chain
    phi = similar(a_pol)
    phi .= 1.0/(NA*NS)
    asset_probs = similar(a)
    C, K_supply, CV_C, CV_K, r, w = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    maxrate = r_max
    minrate = -δ
    err = 1.0
    tol_r = 1e-6

    while abs(err) > tol_r
        r = 0.5*(minrate + maxrate)
        w = r_to_w(r, para)
        para_new = deepcopy(para)
        para_new.r = r 
        para_new.w = w
        update_params!(para_new)
        # solve HH problem (partial equilibrium)
        a_pol, c_pol = solve_model_time_iter(a_pol, para_new, verbose=false, omega=0.5)
        
        a_pol = max.(a_pol, -b+1e-10)
        a_pol = min.(a_pol, grid_max-1e-10)
        # invariant distribution
        # Assign bins to savings policies
        ab_pol = histc(a_pol, a)
        # Calculate weights on adjacent bins
        wei = (a_pol- a[ab_pol]) ./ (a[(ab_pol .+1)] - a[ab_pol])
      
        invariant_dist!(phi, ab_pol, wei, para_new)
        asset_probs, C, K_supply, CV_C, CV_K = sum_stats(phi, a_pol, c_pol, para_new)

        # Inverse demand at capital supplied by HH
        r1 = rd(K_supply, para_new) #interest rate at which firms demand k supplied by HH
        err = r1 - r 
        if err < 0 # lower rate
            maxrate = r 
        else # raise rate
            minrate = r
        end
        @printf("\n K_supply = %.5f, r = %.5f, r_1 = %.5f", K_supply, r, r1)
        # Difference between supply and demand
    end
    return r, w, phi, asset_probs, C, K_supply, CV_C, CV_K, a_pol, c_pol, para
end

"""
    comp_statics_GE_plot(field, vals)
Plot GE comparative statics over different parameters
"""
function comp_statics_GE_plot(field, vals)


    r_vals = similar(vals)
    K_vals = similar(vals)
    c_vals = similar(vals)

    for (i, val) in enumerate(vals)
        # Instantiate and change field to value
        para = Para()
        setfield!(para, field, val)
        # Update parameters
        update_params(para)
        # Solve for general equilibrium
        r, w, K, c = general_equilibrium(para)
        r_vals[i], K_vals[i], c_vals[i] = r, K, c
    end

    fig, ax = subplots(ncols=3, figsize=(12, 4))
    ax[1].plot(vals, r_vals)
    ax[1].set_ylabel("r")
    ax[2].plot(vals, K_vals)
    ax[2].set_ylabel("K")
    ax[3].plot(vals, c_vals)
    ax[3].set_ylabel("c")

    for i in 1:3
        ax[i].set_xlabel("$field")
    end
    tight_layout()
    display(fig)
    return r_vals, K_vals, c_vals
end


function main()

    para = Para(b=0.0)
    @unpack b, grid_max, NA, NS, σ, ρ, R, β, a, mc, P, y = para


    a_prime_init = repeat(a, 1, NS)
    a_pol, c_pol = solve_model_time_iter(a_prime_init, para)
    a_pol = max.(a_pol, -b+1e-10)
    a_pol = min.(a_pol, grid_max-1e-10)
    # invariant distribution
    # Assign bins to savings policies
    ab_pol = histc(a_pol, a)
    # Calculate weights on adjacent bins
    wei = (a_pol- a[ab_pol]) ./ (a[(ab_pol .+1)] - a[ab_pol])
    phi = similar(a_pol)
    phi .= 1.0/(NA*NS)
    invariant_dist!(phi, ab_pol, wei, para)
    asset_probs, C, K_supply, CV_C, CV_K = sum_stats(phi, a_pol, c_pol, para)

     # histogram
     fig, ax = subplots(ncols=1, figsize=(6, 6))
     ax.plot(a, asset_probs, label="asset distribution")
     #ax.fill_between(a, asset_probs, 0.0, alpha=0.2)
     ax.legend()
     tight_layout()
     display(fig)

    fig, ax = subplots()
    for z in 1:NS
        label =  L"c(\cdot, $z)"
        ax.plot(a, c_pol[:, z], label=label)
    end
    ax.set(xlabel="assets", ylabel="consumption")
    ax.legend()
    display(fig)

    # Standard deviation
    # σ
    sigma_vals = range(0.2, stop=0.4; length=3)
    comp_statics_plot(:σ, sigma_vals, para)

    # Borrowing constraint
    #b 
    b_vals = range(0.0, stop=2, length=4)
    comp_statics_plot(:b, b_vals, para)

    # Interest rate
    r_vals = range(0, stop=0.04, length=4)
    comp_statics_plot(:r, r_vals, para)

    # wages
    w_vals = range(0.8, stop=1.2, length=4)
    comp_statics_plot(:w, w_vals, para)

    # curvature of preferences
    gam_vals = range(1.01, stop=4, length=4)
    comp_statics_plot(:γ, gam_vals, para)
  
end

"""
    generate_table(ρ_vals, σ, para)
generate table of statistics for different values of σ and ρ
"""
function generate_stats_table(rho_vals, σ, para)
    r_vals = similar(rho_vals)
    K_vals = similar(rho_vals)
    CV_C_vals = similar(rho_vals)
    CV_K_vals = similar(rho_vals)
    para.σ=σ
    update_params!(para)

    for (i, ρ) in collect(enumerate(rho_vals))
        para.ρ=ρ
        update_params!(para)
        r, w, phi, asset_probs, C, K, CV_C, CV_K = general_equilibrium(para)
        r_vals[i], K_vals[i], CV_C_vals[i], CV_K_vals[i] = r, K, CV_C, CV_K
    end

    t = DataFrame(rho_vals=rho_vals,
        K=K_vals,
        CV_consumption= CV_C_vals,
        CV_capital = CV_K_vals,
        r = r_vals*100)

    return CV_C_vals, K_vals, r_vals, t
end