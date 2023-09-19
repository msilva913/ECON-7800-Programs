
using PyPlot
using Parameters, Random
using NLsolve, Optim


@with_kw mutable struct Para

    α::Array{Float64, 1} = [0.3, 0.4]  # 
    β_1::Float64 = 0.3   # discount factor
    β_2::Float64 = 0.6
    Kbar::Float64 = 20
    Tbar::Float64 = 30
    G::Float64 = 3
    # Taxes
    τ_c::Array{Float64, 1} = [0.0, 0.0]
    τ_w::Float64 = 0.0
    τ_r::Float64 = 0.0

    #u = (L_1, K_1) -> (L_1^β_1*K_1^(1-β_1))^α*((Lbar-L_1)^β_2*(Kbar-K_1)^(1-β_2))^(1-α)
end

function markets(x, para)
    @unpack α, β_1, β_2, Kbar, Tbar, G, τ_c, τ_w, τ_r = para
    q = zeros(2)

    q[1] = 1.0
    q[2] = x[1]
    w = x[2]
    r = x[3]
    # Assume capital income tax adjusts to clear market
    τ_r = x[4]

    # Consumer prices and total income
    p = q.*(1.0 .+ τ_c)
    wn = w.*(1.0 .- τ_w)
    rn = r.*(1.0 .- τ_r)
    Ybarn = wn*Tbar + rn*Kbar

    out = similar(x)

    #out[1] = α[1]*Ybarn/p[1]+ G - (β_1/w)^(β_1)*((1-β_1)/r)^(1-β_1)*q[1]*(α[1]*Ybarn/p[1]+G) 
    out[1] = 1.0 - (β_1/w)^β_1*((1-β_1)/r)^(1-β_1)*q[1]

    out[2] = 1.0 - (β_2/w)^β_2*((1-β_2)/r)^(1-β_2)q[2]

    out[3] = (β_1/w)*q[1]*(α[1]*Ybarn/p[1]+G) + (β_2/w)*q[2]*α[2]*Ybarn/p[2] + (1-α[1]-α[2])/wn*Ybarn - Tbar

    out[4] = q[1]*G -sum((τ_c./(1.0 .+ τ_c).* α))*Ybarn - τ_w*w*(Tbar - (1-α[1]-α[2])/wn*Ybarn) - τ_r*r*Kbar
    
    X_1 = α[1]*Ybarn/p[1]
    X_2 = α[2]*Ybarn/p[2]
    l = (1-α[1]-α[2])*Ybarn/wn
    u = X_1^(α[1])*X_2^(α[2])l^(1-α[1]-α[2])

    return out, q, w, r, p, wn, rn, τ_r, Ybarn, X_1, X_2, l, u, G
end
    # Market equations

para = Para(Kbar=10)
f(x) = markets(x, para)[1]
x0 = [0.5; 0.5; 0.5; 0.5]
res = nlsolve(f, x0)
out, q, w, r, p, wn, rn, τ_r, Ybarn, X_1, X_2, l, u, G = markets(res.zero, para)
@show X_1, G, q[1]
@show X_2, q[2]
@show u
println("Capital income tax rate is ", 100*round(τ_r,digits=3), "%")

