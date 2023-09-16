
using Parameters, NLsolve, DataFrames, Optim

# Finance a given amount of government expenditure using different taxes 
# Examine effect on labor supply, consumption, and output

Para = @with_kw (
    α = [0.3, 0.4], # prefernce parameters
    β = [0.3, 0.6],
    G = 3.0,
    Kbar = 10.0,
    Tbar = 30.0
)

function markets(x, para; tax=1)
    @unpack α, β, G, Tbar, Kbar = para
    q = zeros(2)
    q[1] = 1.0
    q[2] = x[1]
    w = x[2]
    r = x[3]

    # set tax rates based on case
    τ_c = zeros(2)
    τ_w, τ_r = 0.0, 0.0
    if tax == 1 # only tax capital
        τ_r = x[4]
    elseif tax == 2 # high consumption tax with labor subsidy
         τ_w = -x[4]
         τ_c[1] = x[4]
         τ_c[2] = x[4]
    elseif tax == 3 # pure income tax
        τ_w = x[4]
        τ_r = x[4]
    elseif tax == 4 # uniform consumption tax
        τ_c .= x[4]
    elseif tax == 5 #consumption tax and capital tax
        τ_c .= 0.18
        τ_r = x[4]
    elseif tax == 6 #asymmetric consumption tax
        τ_c[1] = x[4]
        τ_c[2] = 0.5*x[4]
    end

   # Calculate consumer prices and total income
   p = q.*(1.0.+τ_c)
   wn = w*(1.0-τ_w)
   rn = r*(1.0-τ_r)
   Ybarn = wn*Tbar + rn*Kbar

   # market clearing conditions
   out = similar(x)

   # goods markets (1 and 2)
   out[1] = α[1]*Ybarn/p[1] + G - (β[1]/w)^(β[1])*((1-β[1])/r)^(1-β[1])*q[1]*(α[1]*Ybarn/p[1]+G)
   out[2] = 1.0 - (β[2]/w)^(β[2])*((1-β[2])/r)^(1-β[2])*q[2]

   # labor market conditions
   out[3] = β[1]/w*q[1]*(α[1]*Ybarn/p[1]+G) + β[2]/w*q[2]*α[2]*Ybarn/p[2] + (1-α[1]-α[2])*Ybarn/wn - Tbar

   # government budget constraint
   out[4] = q[1]*G -sum((τ_c./(1.0 .+ τ_c).* α))*Ybarn - τ_w*w*(Tbar - (1-α[1]-α[2])/wn*Ybarn) - τ_r*r*Kbar

    return out, q, p, w, r, τ_c, τ_w, τ_r, wn, rn, Ybarn
end
    # Market equations

para = Para()
x0 = [0.25; 0.25; 0.25; 0.25]
summ = zeros(6, 11)

# Loop over tax regimes and compute equilibria
for tax in 1:6
    f(x) = markets(x, para, tax=tax)[1]
    res = nlsolve(f, x0)
    # update starting point
    x0 = res.zero
    @show out, q, p, w, r, τ_c, τ_w, τ_r, wn, rn, Ybarn = markets(res.zero, para, tax=tax)
    # Calculate other economic variables
    @unpack α, β, G, Tbar, Kbar = para
    Ybarn = wn*Tbar + rn*Kbar
    X = @. α*Ybarn/p
    Y = similar(X)

    Y[1] = X[1] + G
    Y[2] = X[2]
    leis = (1-sum(α))*Ybarn/wn
    # labor supplied in each market
    L = @. β*q*Y/w
    # Capital supplied in each market
    K = @. (1.0-β)*q*Y/r
    # Utility
    U = X[1]^(α[1])*X[2]^(α[2])*leis^(1-sum(α))

    # organize output: each tax regime in separate row
    summ[tax, :] = [τ_c[1], τ_c[2], τ_w, τ_r, w, r, q[2], X[1], X[2], leis, U]
end

summ_dat = DataFrames.DataFrame(summ, :auto)
summ_dat = round.(summ_dat, digits=2)
rename!(summ_dat, [:τ_1, :τ_2, :τ_w, :τ_r, :w, :r, :q_2, :X_1, :X_2, :leis, :U])
print(summ_dat)
#summ_dat_tex = latexify(summ_dat, env=:table, latex=true)
