using Parameters, Optim, DataFrames

@with_kw mutable struct Para

    α::Float64 = 0.3   # 
    β_1::Float64 = 0.3   # discount factor
    β_2::Float64 = 0.6
    Kbar::Float64 = 10
    Lbar::Float64 = 20

    # define anonymous function
    # utility function to be maximized
    u = (L_1, K_1) -> (L_1^β_1*K_1^(1-β_1))^α*((Lbar-L_1)^β_2*(Kbar-K_1)^(1-β_2))^(1-α)
end

para = Para()
@unpack u, Lbar, Kbar, β_1, β_2 = para
# define negative of utility function
f(x) = -u(x[1], x[2])
x = [5.0, 5.0]

# LBFGS method with auto-differentiation
result = optimize(f, x, LBFGS(), autodiff=:forward)
sol = result.minimizer
@show L_1, K_1 = sol
@show L_2 = Lbar - L_1
@show K_2 = Kbar - K_1
X_1 = L_1^(β_1)*K_1^(1-β_1)
X_2 = L_2^(β_2)*K_2^(1-β_2)

# organize output
summ = zeros(2, 3)
summ[1, :] = [L_1, K_1, X_1]
summ[2, :] = [L_2, K_2, X_2]


summ_dat = DataFrame(summ, :auto)
summ_dat = round.(summ_dat, digits=2)
rename!(summ_dat, [:Labor, :Capital, :Output])
print(summ_dat)
