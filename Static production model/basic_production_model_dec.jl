
using Parameters, NLsolve, DataFrames, Printf

# can used named tuple instead of mutable struct
Para = @with_kw (
    α = 0.3,   # 
    β_1 = 0.3,   # discount factor
    β_2  = 0.6,
    Kbar = 10.0,
    Lbar = 20.0
)

function markets(x, para)
    @unpack α, β_1, β_2, Kbar, Lbar = para
    p = zeros(2)
    p[1] = 1.0
    p[2] = x[1]
    w = x[2]
    r = x[3]

    # Calculate total income
    Ybar = w*Lbar + r*Kbar

    out = similar(x)

    # Market clearing conditions for goods 1 and 2
    out[1] = 1.0/p[1] - (β_1/w)^(β_1)*((1-β_1)/r)^(1-β_1) 
    out[2] = 1.0/p[2] - (β_2/w)^(β_2)*((1-β_2)/r)^(1-β_2) 

    # Market clearing condition for capital
    out[3] = (β_1/w)*α*Ybar+β_2/w*(1-α)*Ybar - Lbar

    return out, p, w, r
end
    # Market equations

para = Para()
f(x) = markets(x, para)[1]
x0 = [0.5; 0.5; 0.5]
res = nlsolve(f, x0)
@show out, p, w, r = markets(res.zero, para)
# Calculate other economic variables
@unpack α, β_1, β_2, Kbar, Lbar = para
Ybar = w*Lbar + r*Kbar
X_1 = α*Ybar/p[1]
X_2 = (1-α)*Ybar/p[2]
L_1 = β_1/w*p[1]*X_1
L_2 = β_2/w*p[2]*X_2

K_1 = (1-β_1)/r*p[1]*X_1
K_2 = (1-β_2)/r*p[2]*X_2
# organize output
summ = zeros(2, 3)
summ[1, :] = [L_1, K_1, X_1]
summ[2, :] = [L_2, K_2, X_2]


summ_dat = DataFrame(summ, :auto)
summ_dat = round.(summ_dat, digits=2)
rename!(summ_dat, [:Labor, :Capital, :Output])
print(summ_dat)

# Print prices using Printf

@printf "\n p1 = %.2f" p[1]
@printf "\n p2 = %.2f" p[2]
@printf "\n w = %.2f" w
@printf "\n r = %.2f" r


