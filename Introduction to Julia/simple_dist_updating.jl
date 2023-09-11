using PyPlot
using LinearAlgebra

cd(@__DIR__)

π_0= [0.8 0.2]
P = [0.965 0.035; 0.45 0.55]

println("The initial employment mass is ", π_0)

π_1 = π_0*P
println("At t=1, the employment mass is ", π_1)
π_2 = π_1*P
println("At t=2, the employment mass is ", π_2)

# Check for long run limit

function simulate_markov(P::Matrix, π_0::Array, T=50)
    # Store distributions over time
    π_array = zeros(T, 2)
    #Initialize
    π_array[1, :] = π_0
    # Update
    for t in 1:(T-1)
        π_array[t+1, :] = π_array[t, :]'*P
    end
    return π_array
end

π_array = simulate_markov(P, π_0, 20)

fig, ax = plt.subplots(nrows=2, figsize=(10, 5))
ax[1].plot(π_array[:, 1], label="Employed", linewidth=2.5)
ax[2].plot(π_array[:, 2], label="Unemployed", linewidth=2.5)
ax[1].legend()
ax[2].legend()
plt.tight_layout()
display(fig)
plt.savefig("evolution_employment.pdf")
plt.show()

println("At t=20, employment mass is ", π_array[end, :])

# Find by computing eigenvectors 

# Stationary distribution satisfies (I-P')π=0
# Find π by computing an associated eingevector of P'
evals, evecs = eigvals(transpose(P)), eigvecs(transpose(P))

@show π_∞ = evecs[:, 2]/sum(evecs[:, 2])
