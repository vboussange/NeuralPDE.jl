"""
    IntegroPDEProblem(f, μ, σ, x0, tspan)
A non local non linear PDE problem.
Consider `du/dt = l(u) + \int f(u,x) dx`; where l is the nonlinear Lipschitz function
# Arguments
* `f` : The function f(u)
* `μ` : The drift function of X from Ito's Lemma
* `μ` : The noise function of X from Ito's Lemma
* `x0`: The initial X for the problem.
* `tspan`: The timespan of the problem.
"""
struct IntegroPDEProblem{F,Mu,Sigma,X,T,P,A,UD,K} <: DiffEqBase.DEProblem
    f::F
    μ::Mu
    σ::Sigma
    X0::X
    tspan::Tuple{T,T}
    p::P
    A::A
    u_domain::UD
    kwargs::K
    IntegroPDEProblem(f,μ,σ,X0,tspan,p=nothing;A=nothing,u_domain=nothing,kwargs...) = new{typeof(f),
                                                         typeof(μ),typeof(σ),
                                                         typeof(X0),eltype(tspan),
                                                         typeof(p),typeof(A),typeof(u_domain),typeof(kwargs)}(
                                                         g,f,μ,σ,X0,tspan,p,A,u_domain,kwargs)
end

Base.summary(prob::IntegroPDEProblem) = string(nameof(typeof(prob)))

function Base.show(io::IO, A::IntegroPDEProblem)
  println(io,summary(A))
  print(io,"timespan: ")
  show(io,A.tspan)
end

"""
Deep splitting algorithm for solving non local non linear PDES.

Arguments:
* `chain`: a Flux.jl chain with a d-dimensional input and a 1-dimensional output,
* `strategy`: determines which training strategy will be used,
* `init_params`: the initial parameter of the neural network,
* `phi`: a trial solution,
* `derivative`: method that calculates the derivative.

"""

struct NNPDEDS{C1,O} <: NeuralPDEAlgorithm
    u0::C1
    K::Int
    opt::O
end
NNPDEDS(u0;K=1,opt=Flux.ADAM(0.1)) = NNPDEDS(u0,K,opt)

function DiffEqBase.solve(
    prob::TerminalPDEProblem,
    alg::NNPDEDS,
    sde,
    mc_sample;
    batch_size = 1,
    std_sampling_mc,
    lr_boundaries,
    lr_values,
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    save_everystep = false,
    dt,
    give_limit = false,
    trajectories,
    sdealg = EM(),
    ensemblealg = EnsembleThreads(),
    trajectories_upper = 1000,
    trajectories_lower = 1000,
    maxiters_upper = 10,
    )

    X0 = prob.X0
    ts = prob.tspan[1]:dt:prob.tspan[2]
    N = length(ts)
    d  = length(X0)
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p

    data = Iterators.repeated((), maxiters)


    #hidden layer
    opt = alg.opt
    u0 = alg.u0
    σᵀ∇u = alg.σᵀ∇u
    ps = Flux.params(u0, σᵀ∇u...)

    # this is the splitting model
    function sol()
        map(1:trajectories) do j
            u = u0(X0)[1]
            X = X0
            for i in 1:length(ts)-1
                t = ts[i]
                _σᵀ∇u = σᵀ∇u[i](X)
                dW = sqrt(dt)*randn(d)
                u = u - f(X, u, _σᵀ∇u, p, t)*dt + _σᵀ∇u'*dW
                X  = X .+ μ(X,p,t)*dt .+ σ(X,p,t)*dW
            end
            X,u
        end
    end

    function loss()
        mean(sum(abs2,g(X) - u) for (X,u) in sol())
    end

    iters = eltype(X0)[]

    cb = function ()
        save_everystep && push!(iters, u0(X0)[1])
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    for (i,t) in enumerate(ts)
        # this is equivalent to the sde_loop
        y1 = X0 .+ randn(d,batch_size) .* σ .* sqrt(ts[end] - t)
        y0 = X0 .+ randn(d,batch_size) .* σ .* sqrt(ts[end] - ts[i+1])
        y1 = Batch(c for eachcol(y1))
        y0 = Batch(c for eachcol(y0))
        loss, v_n, v_j = splitting_model()
        # Victor : you should make sure that the batch works
        # t
        Flux.train!(loss, ps, data, opt; cb = cb)
    end

    save_everystep ? iters : u0(X0)[1]

end #pde_solve
