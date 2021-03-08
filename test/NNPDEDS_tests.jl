using Flux, Zygote, LinearAlgebra, Statistics
println("NNPDEHAN_tests")
using Test, StochasticDiffEq
println("Starting Soon!")
using NeuralPDE

using Random
Random.seed!(100)

# one-dimensional heat equation
x0 = [11.0f0]  # initial points
tspan = (0.0f0,5.0f0)
dt = 0.5   # time step
time_steps = div(tspan[2]-tspan[1],dt)
d = 1      # number of dimensions
m = 10     # number of trajectories (batch size)
batch_size = 8192
train_steps = 400
lr_boundaries = [100, 200, 300]
lr_values = [1e-1,1e-2,1e-3,1e-4]


g(X) = sum(X.^2)   # terminal condition
f(y,z,v_y,v_z,dv_y,dv_z,t) = sin(v_y) - v_z * π^(d/2)*σ_sampling^2 # function from solved equation
μ_f(X,p,t) = 0.0
σ_f(X,p,t) = 1.0
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)


hls = d + 50 #hidden layer size
#neural network approximating solutions at the desired point
# with batch normalisation
# u0 = Flux.Chain(Dense(d,hls),
#                 BatchNorm(hls,tanh),
#                 Dense(hls,hls,tanh),
#                 BatchNorm(hls,tanh),
#                 Dense(hls,1))
# without batch normalisation
u0 = Flux.Chain(Dense(d,hls,tanh),
                Dense(hls,hls,tanh),
                Dense(hls,1))
