include("Aiyagari_functions.jl")

para = Para()

function supply_demand_curves(para)
    """
 
    """
    @unpack a, grid_max, b, δ, β, NA, NS = para
    r_max = (1-β)/β
    # initialize policies
    a_pol = 0.8.*repeat(a, 1, NS)
    c_pol = similar(a_pol)

    # initialize distribution
    phi = similar(a_pol)
    phi .= 1.0/(NA*NS)

    asset_probs = similar(a)
    maxrate = r_max
    r_vals = range(0.01, r_max*0.999, length=25)
    K_vals = similar(r_vals)

    for (i, r) in enumerate(r_vals)
        # get wage
        w = r_to_w(r, para)
        # create new instance of mutable struct
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
        __, __, K_vals[i], __, __ = sum_stats(phi, a_pol, c_pol, para_new)
    end

        # Inverse demand at capital supplied by HH
    r_demand = rd.(K_vals, Ref(para_new))

    return r_vals, K_vals, r_demand
end

r_vals, K_vals, r_demand = supply_demand_curves(para)

fig, ax = plt.subplots()
ax.plot(K_vals, r_vals, label="Supply curve")
ax.plot(K_vals, r_demand, label="Demand curve")
ax.legend()
display(fig)
  
