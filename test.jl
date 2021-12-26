using DifferentialEquations, DiffEqFlux, Flux, Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)

sol = solve(prob,saveat=0.1)

plot(sol)
A = sol[1,:]
t = 0:0.1:10.0
scatter!(t,A)

p = [2.2, 1.0, 2.0, 0.4]
params = Flux.params(p)

function predict_rd()
  solve(prob,Tsit5(),p=p,saveat=0.1)[1,:]
end

loss_rd() = sum(abs2,x-1 for x in predict_rd())

data = Iterators.repeated((), 120)
opt = ADAM(0.1)

cb = function ()
  display(loss_rd())
  display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end

cb()

Flux.train!(loss_rd, params, data, opt, cb=cb)

using OrdinaryDiffEq
using ParameterizedFunctions

rob = @ode_def Rob begin
  dy₁ = -k₁*y₁+k₃*y₂*y₃
  dy₂ =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
  dy₃ =  k₂*y₂^2
end k₁ k₂ k₃

function rober(du,u,p,t)
  k₁, k₂, k₃ = p
  y₁, y₂, y₃ = u
  du[1] = dy₁ = -k₁*y₁+k₃*y₂*y₃
  du[2] = dy₂ =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
  du[3] = dy₃ =  k₂*y₂^2
end

prob = ODEProblem(rob,[1.0;0.0;0.0],(0.0,1e11),(0.04,3e7,1e4))
sol = solve(prob,KenCarp4())

plot(sol,xscale=:log10,tspan=(0.1,1e11))

function delay_lotka_volterra(du,u,h,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)*h(p,t-0.1)[1]
  du[2] = dy = (δ*x - γ)*y
end

h(p,t) = ones(eltype(p),2)
prob = DDEProblem(delay_lotka_volterra,[1.0,1.0],h,(0.0,10.0),constant_lags=[0.1])
p = [2.2, 1.0, 2.0, 0.4]
sol_d = solve(prob,p=p)
plot(sol_d)

params = Flux.params(p)

using DiffEqSensitivity

function predict_rd_dde()
  solve(prob,MethodOfSteps(Tsit5()),p=p,sensealg=TrackerAdjoint(),saveat=0.1)[1,:]
end

loss_rd_dde() = sum(abs2,x-1 for x in predict_rd_dde())
loss_rd_dde()

data = Iterators.repeated((), 100)
opt = ADAM(0.1)

cb_d = function ()
  display(loss_rd_dde())
  display(plot(solve(remake(prob,p=p),MethodOfSteps(Tsit5()),saveat=0.1),ylim=(0,6)))
end

cb_d()

Flux.train!(loss_rd_dde, [p], data, opt, cb = cb_d)

function lotka_volterra_noise(du,u,p,t)
  du[1] = 0.1u[1]
  du[2] = 0.1u[2]
end

prob = SDEProblem(lotka_volterra,lotka_volterra_noise,[1.0,1.0],(0.0,5.0))

sol_n = solve(prob, p=p)
plot(sol_n)

p = [2.2, 1.0, 2.0, 0.4]
params = Flux.params(p)
function predict_sde()
  solve(prob,SOSRI(),p=p,sensealg=TrackerAdjoint(),saveat=0.1,
                     abstol=1e-1,reltol=1e-1)[1,:]
end
loss_rd_sde() = sum(abs2,x-1 for x in predict_sde())
loss_rd_sde()

cb_s = function ()
  display(loss_rd_sde())
  display(plot(solve(remake(prob,p=p),SOSRI(),saveat=0.1),ylim=(0,15)))
end

data = Iterators.repeated((),150)
opt = Flux.ADAM(0.01)

Flux.train!(loss_rd_sde, [p], data, opt, cb=cb_s)

dudt = Chain(Dense(2,50,tanh),Dense(50,2))

tspan = (0.0f0,25.0f0)
node = NeuralODE(dudt,tspan,Tsit5(),saveat=0.1)
node = NeuralODE(gpu(dudt),tspan,Tsit5(),saveat=0.1)

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((sin.(u))'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

dudt = Chain(Dense(2,50,tanh),
             Dense(50,50,tanh),
             Dense(50,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(n_ode)

pred = n_ode(u0)
scatter(t,ode_data[1,:],label="data")
scatter!(t,pred[1,:],label="prediction")

function predict_n_ode()
  n_ode(u0)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

loss_n_ode()

data = Iterators.repeated((), 300)
opt = ADAM(0.05)
cb_n = function ()
  display(loss_n_ode())
  cur_pred = predict_n_ode()
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

cb_n()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb_n)

using Sundials, DiffEqBase

function lorenz(du,u,p,t)
 du[1] = 10.0*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob,CVODE_Adams(),reltol=1e-12,abstol=1e-12)
prob2 = ODEProblem(lorenz,sol[end],(100.0,0.0))
sol = solve(prob,CVODE_Adams(),reltol=1e-12,abstol=1e-12)
@show sol[end]-u0
