# Closely based on 
# https://julia.quantecon.org/more_julia/optimization_solver_packages.html

using LinearAlgebra, Statistics
using ForwardDiff, Optim, JuMP, Ipopt, BlackBoxOptim, Roots, NLsolve, LeastSquaresOptim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

# Calculating derivatives of functions
# 1) Analytically (by hand): invaluable, but sometimes tedious and error-prone
# 2) Finite differences: challenge of maintaining both accuracy and stability
# 3) Automatic differentiation--algorithmic differentiation with calculation of the chain rule

h(x) = sin(x[1]) + x[1] * x[2] + sinh(x[1] * x[2]) # multivariate.
x = [1.4 2.2]
@show ForwardDiff.gradient(h,x) # use AD, seeds from x

#Or, can use complicated functions of many variables
f(x) = sum(sin, x) + prod(tan, x) * sum(sqrt, x)
g = (x) -> ForwardDiff.gradient(f, x); # g() is now the gradient
g(rand(5)) # gradient at a random point
# ForwardDiff.hessian(f,x') # or the hessian

# Optimization 
#Optim.jl works well for unconstrained or box-bounded optimization of univariate and multivariate functions
f(x) = x^2
x_range = -2:0.1:1.0
plot(x_range, f.(x_range))

# returns various fields holding output
result = optimize(x ->x^2, -2.0, 1.0)
converged(result) || error("Failed to converge in $(iterations(result)) iterations")

# can use maximize function instead
result = maximize(x -> x^2, -2.0, 1.0)
converged(result) || error("Failed to converge in $(iterations(result)) iterations")

# Unconstrained multivariate optimization
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x_iv = [0.0, 0.0]

# Default uses Nelder-Mead (derivative-free method)
results = optimize(f, x_iv) # i.e. optimize(f, x_iv, NelderMead())
@show results.minimizer

# Can change algorithm t ype to L-BFGS
# Can use automatic differentiation

results = optimize(f, x_iv, LBFGS(), autodiff=:forward)
println("minimum = $(results.minimum) with argmin = $(results.minimizer) in "*
"$(results.iterations) iterations")

# Can also specify analytic gradient
function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

results = optimize(f, g!, x_iv, LBFGS())
println("minimum = $(results.minimum) with argmin = $(results.minimizer) in "*
"$(results.iterations) iterations")

# See also  BlackBoxOptim.jl for global deriative-free optimization and JuMP

# System of equations and least LeastSquares 

f(x) = sin(4(x-1/4)) + x +x^20 - 1
x_vals = 0.0:0.05:1.0
plot(x_vals, f.(x_vals))

# Find root 
@show sol = fzero(f, 0, 1)

# Roots of multivariate systems of equations

using NLsolve

f(x) = [(x[1]+3)*(x[2]^3-7)+18
   sin(x[2]*exp(x[1])-1)] # returns an array

results = nlsolve(f, [ 0.1; 1.2], autodiff=:forward)
println("converged=$(NLsolve.converged(results)) at root=$(results.zero) in "*
"$(results.iterations) iterations and $(results.f_calls) function calls")

# Each function evaluation requires creating copies 
# Functions which modify arguments can sometimes help

function f!(F, x) # modifies the first argument
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

results = nlsolve(f!, [ 0.1; 1.2], autodiff=:forward)

println("converged=$(NLsolve.converged(results)) at root=$(results.zero) in "*
"$(results.iterations) iterations and $(results.f_calls) function calls")

# Least squares optimization
# of the form x ∈ R^N, F(x): R^N -> R^M 
# min_x F(x)^T F(x)

using LeastSquaresOptim
function rosenbrock(x)
    [1 - x[1], 100 * (x[2]-x[1]^2)]
end
sol = LeastSquaresOptim.optimize(rosenbrock, zeros(2), Dogleg())
@show sol.minimizer
