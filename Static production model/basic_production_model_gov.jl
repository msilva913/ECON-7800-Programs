
using PyPlot
using Parameters, Random
using NLsolve, Optim


# using named tuple (can also use mutable struct)
Para = @with_kw (
    α = [0.3, 0.4], # prefernce parameters
    β = [0.3, 0.6],
    G = 3.0,
    Kbar = 10.0,
    Tbar = 30.0
)

function markets(x, para)
    @unpack α, β, Kbar, Tbar, G = para
    q = zeros(2)

    q[1] = 1.0
    q[2] = x[1]
    w = x[2]
    r = x[3]
    # Assume capital income tax adjusts to clear market
    τ_c = zeros(2)
    τ_w = 0.0
    τ_r = x[4]

    # Consumer prices and total income
    p = q.*(1.0 .+ τ_c)
    wn = w*(1.0 - τ_w)
    rn = r*(1.0 - τ_r)
    Ybarn = wn*Tbar + rn*Kbar

    out = similar(x)

    #out[1] = α[1]*Ybarn/p[1]+ G - (β_1/w)^(β_1)*((1-β_1)/r)^(1-β_1)*q[1]*(α[1]*Ybarn/p[1]+G) 
    out[1] = 1.0 - (β[1]/w)^(β[1])*((1-β[1])/r)^(1-β[1])*q[1]
    out[2] = 1.0 - (β[2]/w)^(β[2])*((1-β[2])/r)^(1-β[2])*q[2]

    out[3] = (β[1]/w)*q[1]*(α[1]*Ybarn/p[1]+G) + (β[2]/w)*q[2]*α[2]*Ybarn/p[2] + (1-α[1]-α[2])/wn*Ybarn - Tbar

    out[4] = q[1]*G -sum((τ_c./(1.0 .+ τ_c).* α))*Ybarn - τ_w*w*(Tbar - (1-α[1]-α[2])/wn*Ybarn) - τ_r*r*Kbar
    
    X_1 = α[1]*Ybarn/p[1]
    X_2 = α[2]*Ybarn/p[2]
    l = (1-α[1]-α[2])*Ybarn/wn
    u = X_1^(α[1])*X_2^(α[2])l^(1-α[1]-α[2])

    return out, q, w, r, p, wn, rn, τ_r, Ybarn, X_1, X_2, l, u, G
end
    # Market equations

para = Para(G=3.0)
f(x) = markets(x, para)[1]
x0 = [0.5; 0.5; 0.5; 0.5]
res = nlsolve(f, x0)
out, q, w, r, p, wn, rn, τ_r, Ybarn, X_1, X_2, l, u, G = markets(res.zero, para)
@show X_1, G, q[1]
@show X_2, q[2]
@show u
println("Capital income tax rate is ", 100*round(τ_r,digits=3), "%")

