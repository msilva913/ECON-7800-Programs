
# Illustrate AR(2) process
using PyPlot

#3) Impulse response of an ARMA(1, 1) processes
function ψ_fun(ρ_1, ρ_2, T::Int)
    X = zeros(T)
    X[1] = 1 # initial shock
    X[2] = ρ_1*X[1]
    for t in 3:T
        X[t] = ρ_1*X[t-1] + ρ_2*X[t-2]
    end
    return X
end

T = 20
series_1 = ψ_fun(0.9, 0.05, T)
series_2 = ψ_fun(0.9, -0.1, T)


fig, ax = subplots()
ax.plot(1:T, series_1, linewidth=2, alpha=0.6, label="ρ_1=0.9, ρ_2=0.05")
ax.plot(1:T, series_2, linewidth=2, alpha=0.6, label="ρ_1=0.9, ρ_2=-0.1")
ax.legend()
tight_layout()
ax.set_title("A(2) process")
display(fig)
plt.savefig("AR2_process.pdf")



