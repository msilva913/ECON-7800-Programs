
# Static general equilibrium model with intermediate goods #
using Parameters, LinearAlgebra
using NLsolve, DataFrames

Para = @with_kw (
    Kbar = 10.0,
    Tbar = 30.0,
    α = [0.3, 0.4],
    β = [0.3, 0.6],
    a0 = [0.2, 0.2],
    a = [0.0 0.3;
         0.2 0.0],
    G = 3.0
)

#u = (L_1, K_1) -> (L_1^β_1*K_1^(1-β_1))^α*((Lbar-L_1)^β_2*(Kbar-K_1)^(1-β_2))^(1-α)

function markets(x, para)
    @unpack α, a0, a, β, G, Tbar, Kbar = para
    w = 1.0
    r = x[1]
    τ_c = zeros(2)
    τ_w, τ_r = 0.0, 0.0
    τ_c[1] = x[2]
    τ_c[2] = τ_c[1]

    # calculate K/Y and L/Y
   ky = @. a0*((1.0-β)/β*w/r)^β
   ly = @. a0*(β/(1.0-β)*r/w)^(1-β)
   # determine producer prices
   b = @. w*ly + r*ky
   # Solve matrix system for q
   A = I(2) - a'
   q = A\b
   #(I(2) - a')*q - @.w*ly + r*ky
   # consumer prices and demands
    p = @. q*(1.0+τ_c)
    wn = w*(1.0-τ_w)
    rn = r*(1.0-τ_r)

    Ybarn = wn*Tbar + rn*Kbar
    X = @. α/p*Ybarn
    leis = (1.0-α[1]-α[2])/wn*Ybarn

    # determine output levels
    Y = similar(X)
    Y[1] = X[1] + G
    Y[2] = X[2]
    A = I(2) - a
    Y .= A\Y

    # Compute K and L
    K = ky.*Y
    L = ly.*Y
    # Utility
    U = X[1]^(α[1])*X[2]*(α[2])*leis^(1-α[1]-α[2])
    # check markets and budget
    out = similar(X)
    out[1]  = sum(K) - Kbar
    out[2] = q[1]*G - sum(τ_c.*q.*X) - τ_w*w*(Tbar-leis) - τ_r*r*Kbar
    return out, w, r, τ_c, τ_w, τ_r, q, p, Y, X, K, L, U
end


summ = zeros(2, 9)
for (i, Kbar) in enumerate([10.0, 8.0])
    # set parameters
    para = Para(Kbar=Kbar)
    # set function in terms of one variable
    f(x) = markets(x, para)[1]
    x0 = [0.2, 0]
    res = nlsolve(f, x0)
    out, w, r, τ_c, τ_w, τ_r, q, p, Y, X, K, L, U = markets(res.zero, para)
    summ[i, :] = [w, r, q[1], q[2], τ_c[1], τ_c[2], p[1], p[2], U]
end
    

summ_dat = DataFrames.DataFrame(summ, :auto)
summ_dat = round.(summ_dat, digits=2)
rename!(summ_dat, [:w, :r, :q_1, :q_2, :τ_1, :τ_2, :p_1, :p_2, :U])
print(summ_dat)

#summ_dat_tex = latexify(summ_dat, env=:table, latex=true)
