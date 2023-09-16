
using Parameters, NLsolve, DataFrames, Printf

Para = @with_kw (
    α_1 = 0.3,   # 
    α_2 = 0.4,
    β_1 = 0.3,   # discount factor
    β_2  = 0.6,
    Kbar = 10.0,
    Tbar = 30.0
)


function markets(x, para)
    @unpack α_1, α_2, β_1, β_2, Kbar, Tbar = para
    p = zeros(2)
    p[1] = 1.0
    p[2] = x[1]
    w = x[2]
    r = x[3]

    # Calculate potential output
    Ybar = w*Tbar + r*Kbar

    out = similar(x)

    out[1] = 1.0/p[1] - (β_1/w)^(β_1)*((1-β_1)/r)^(1-β_1) 
    out[2] = 1.0/p[2] - (β_2/w)^(β_2)*((1-β_2)/r)^(1-β_2) 
    # labor market equilibrium changes with variable labor
    out[3] = β_1*α_1*Ybar/w + β_2*α_2*Ybar/w + (1.0-α_1-α_2)*Ybar/w - Tbar

    return out, p, w, r
end
    # Market equations

para = Para()
# loss function

f(x) = markets(x, para)[1]
x0 = [0.5; 0.5; 0.5]
res = nlsolve(f, x0)
@show out, p, w, r = markets(res.zero, para)

# Calculate other economic variables
@unpack α_1, α_2, β_1, β_2, Kbar, Tbar = para
Ybar = w*Tbar + r*Kbar
X_1 = α_1*Ybar/p[1]
X_2 = α_2*Ybar/p[2]
L_1 = β_1/w*p[1]*X_1
L_2 = β_2/w*p[2]*X_2
leis = (1-α_1-α_2)/w*Ybar

K_1 = (1-β_1)/r*p[1]*X_1
K_2 = (1-β_2)/r*p[2]*X_2
# organize output
summ = zeros(2, 3)
summ[1, :] = [L_1, K_1, X_1]
summ[2, :] = [L_2, K_2, X_2]


summ_dat = DataFrame(summ, :auto)
summ_dat = round.(summ_dat, digits=3)
rename!(summ_dat, [:Labor, :Capital, :Output])
print(summ_dat)

@printf "\n p1 = %.2f" p[1]
@printf "\n p2 = %.2f" p[2]
@printf "\n w = %.2f" w
@printf "\n r = %.2f" r



