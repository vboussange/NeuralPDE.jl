using Flux, Zygote, LinearAlgebra, Statistics
# println("NNPDE_deepsplitting_tests")
using Test, StochasticDiffEq
# println("Starting Soon!")
using Revise
include("/Users/victorboussange/.julia/dev/NeuralPDE/src/pde_solve_deepsplitting.jl")
# using NeuralPDE

using Random
Random.seed!(100)

d = 5 # number of dimensions
# one-dimensional heat equation
x0 = fill(0.0f0,d)  # initial points
tspan = (0.0f0,1.0f0)
dt = 0.1   # time step
time_steps = div(tspan[2]-tspan[1],dt)
batch_size = 8192
train_steps = 10
lr_boundaries = [100, 200, 300]
lr_values = [1e-1,1e-2,1e-3,1e-4]
σ_sampling = 0.1

g(X) = 2^(d/2) * exp(-2*π * sum(X.^2))   # terminal condition
a(X) = - sum(X.^2) /2.
f(y,z,v_y,v_z,dv_y,dv_z,p,t) = sin(v_y) - v_z * π^(d/2)*σ_sampling^2 # function from solved equation
μ_f(X,p,t) = 0.0
σ_f(X,p,t) = 1.0
mc_sample(x) = x + randn(d) * σ_sampling / sqrt(2)

## One should look at InitialPDEProble, this would surely be more appropriate
prob = PIDEProblem(g, f, μ_f, σ_f, x0, tspan)


hls = d + 50 #hidden layer size

## construction with batch size
#neural network approximating solutions at the desired point
# with batch normalisation
# u0 = Flux.Chain(Dense(d,hls),
#                 BatchNorm(hls,tanh),
#                 Dense(hls,hls,tanh),
#                 BatchNorm(hls,tanh),
#                 Dense(hls,1))
# without batch normalisation
nn = Flux.Chain(Dense(d,hls,tanh),
                Dense(hls,hls,tanh),
                Dense(hls,1))

opt = Optimiser(ExpDecay(η = 0.1,
                decay = 0.1,
                decay_step = 5,
                clip = 1e-4),
                ADAM())
alg = NNPDEDS(nn,K=1,opt = opt )

u1 = solve(prob, alg, mc_sample,
            dt=dt,
            verbose = true,
            abstol=1e-8,
            maxiters = train_steps,
            batch_size=batch_size)
