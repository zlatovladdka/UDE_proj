using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
using DifferentialEquations
gr()

function corona!(du,u,p,t)
    S,E,I,R,N,D,C = u
    F, β0,α,κ,μ,σ,γ,d,λ = p
    dS = -β0*S*F/N - β(t,β0,D,N,κ,α)*S*I/N -μ*S # susceptible
    dE = β0*S*F/N + β(t,β0,D,N,κ,α)*S*I/N -(σ+μ)*E # exposed
    dI = σ*E - (γ+μ)*I # infected
    dR = γ*I - μ*R # removed (recovered + dead)
    dN = -μ*N # total population
    dD = d*γ*I - λ*D # severe, critical cases, and deaths
    dC = σ*E # +cumulative cases

    du[1] = dS; du[2] = dE; du[3] = dI; du[4] = dR
    du[5] = dN; du[6] = dD; du[7] = dC
end

β(t,β0,D,N,κ,α) = β0*(1-α)*(1-D/N)^κ
S0 = 14e6
u0 = [0.9*S0, 0.0, 0.0, 0.0, S0, 0.0, 0.0]
p_ = [10.0, 0.5944, 0.4239, 1117.3, 0.02, 1/3, 1/5,0.2, 1/11.2]
R0 = p_[2]/p_[7]*p_[6]/(p_[6]+p_[5])
tspan = (0.0, 21.0)

prob = ODEProblem(corona!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 1)
scatter(solution)

tspan2 = (0.0,60.0)
prob = ODEProblem(corona!, u0, tspan2, p_)
solution_extrapolate = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 1)
scatter(solution_extrapolate[2:4, :]', label=["Exposed" "Infected" "Removed"])
savefig("repos\\UDE_proj\\pres\\initial1.png")

scatter(solution_extrapolate[1:1, :]', label="Susceptible")
scatter!(solution_extrapolate[5:end, :]', label=["Population" "Severe cases" "Cumulative cases"])
savefig("repos\\UDE_proj\\pres\\initial2.png")

# данные
tsdata = Array(solution)
noisy_data = tsdata + Float32(1e-5)*randn(eltype(tsdata), size(tsdata))

tsdata_extrapolate = Array(solution_extrapolate)
noisy_data_extrapolate = tsdata_extrapolate + Float32(1e-5)*randn(eltype(tsdata_extrapolate), size(tsdata_extrapolate))

plot(abs.(tsdata-noisy_data)')
scatter(noisy_data')

ann_node = FastChain(FastDense(7, 64, tanh),FastDense(64, 64, tanh), FastDense(64, 64, tanh), FastDense(64, 7))
p = Float64.(initial_params(ann_node))

function dudt_node(u,p,t)
    S,E,I,R,N,D,C = u
    F,β0,α,κ,μ,σ,γ,d,λ = p_
    dS,dE,dI,dR,dD = ann_node([S/N,E,I,R,N,D/N,C],p)

    dN = -μ*N # total population
    dC = σ*E # +cumulative cases

    [dS,dE,dI,dR,dN,dD,dC]
end

prob_node = ODEProblem(dudt_node, u0, tspan, p)
s = concrete_solve(prob_node, Tsit5(), u0, p, saveat = solution.t)

plot(s)
scatter!(solution)

function predict(θ)
    Array(concrete_solve(prob_node, Vern7(), u0, θ, saveat = 1,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

function loss(θ)
    pred = predict(θ)
    sum(abs2, (noisy_data[2:4,:] .- pred[2:4,:])), pred # + 1e-5*sum(sum.(abs, params(ann)))
end

loss(p)

losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
end

res1_node = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 500)
res2_node = DiffEqFlux.sciml_train(loss, res1_node.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 10000)

losses

prob_node2 = ODEProblem(dudt_node, u0, tspan, res2_node.minimizer)
s = solve(prob_node2, Tsit5(), saveat = 1)
scatter(solution, vars=[2,3,4], label=["Данные: Exposed" "Данные: Infected" "Данные: Recovered"])
plot!(s, vars=[2,3,4], label=["NODE: Exposed" "NODE: Infected" "NODE: Recovered"])

savefig("repos\\UDE_proj\\pres\\neuralode_train.png")

plot(losses, yaxis = :log, xaxis = :log, xlabel = "Итерации", ylabel = "Потери")
savefig("repos\\UDE_proj\\pres\\neuralode_loss.png")

prob_node_extrapolate = ODEProblem(dudt_node,u0, tspan2, res2_node.minimizer)
_sol_node = solve(prob_node_extrapolate, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 1)
p_node = scatter(solution_extrapolate, vars=[2,3,4], legend = :topleft, label=["Данные: Exposed" "Данные: Infected" "Данные: Recovered"], title="Экстраполяция Neural ODE")
plot!(p_node,_sol_node, lw=5, vars=[2,3,4], label=["NODE: Exposed" "NODE: Infected" "NODE: Recovered"])
plot!(p_node,[20.99,21.01],[0.0,maximum(hcat(Array(solution_extrapolate[2:4,:]),Array(_sol_node[2:4,:])))],lw=5,color=:black,label="Граница тренировочных данных")

savefig("repos\\UDE_proj\\pres\\neuralode_extrapolation.png")


### UODE

ann = FastChain(FastDense(3, 64, tanh),FastDense(64, 64, tanh), FastDense(64, 1))
p = Float64.(initial_params(ann))

function dudt_(u,p,t)
    S,E,I,R,N,D,C = u
    F, β0,α,κ,μ,σ,γ,d,λ = p_
    z = ann([S/N,I,D/N],p) # Exposure does not depend on exposed, removed, or cumulative!
    dS = -β0*S*F/N - z[1] -μ*S # susceptible
    dE = β0*S*F/N + z[1] -(σ+μ)*E # exposed
    dI = σ*E - (γ+μ)*I # infected
    dR = γ*I - μ*R # removed (recovered + dead)
    dN = -μ*N # total population
    dD = d*γ*I - λ*D # severe, critical cases, and deaths
    dC = σ*E # +cumulative cases

    [dS,dE,dI,dR,dN,dD,dC]
end

prob_nn = ODEProblem(dudt_,u0, tspan, p)
s = concrete_solve(prob_nn, Tsit5(), u0, p, saveat = 1)

plot(solution, vars=[2,3,4])
scatter!(s[2:4,:]')

function predict(θ)
    Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat = solution.t,
                         abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end


function loss(θ)
    pred = predict(θ)
    sum(abs2, noisy_data[2:4,:] .- pred[2:4,:]), pred # + 1e-5*sum(sum.(abs, params(ann)))
end

loss(p)

losses = []
callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println(losses[end])
    end
    false
end

res1_uode = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), cb=callback, maxiters = 500)
res2_uode = DiffEqFlux.sciml_train(loss, res1_uode.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 1000)
res2_uode = DiffEqFlux.sciml_train(loss, res2_uode.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters = 1000)


loss(res2_uode.minimizer)

prob_nn2 = ODEProblem(dudt_,u0, tspan, res2_uode.minimizer)
uode_sol = solve(prob_nn2, Tsit5(), saveat = 1)
scatter(solution, vars=[2,3,4], label="Данные")
plot!(uode_sol, vars=[2,3,4], label="Предсказание UODE")

savefig("repos\\UDE_proj\\pres\\uode_prediction_train.png")

plot(losses, yaxis = :log, xaxis = :log, xlabel = "Итерации", ylabel = "Потери")
savefig("repos\\UDE_proj\\pres\\uode_loss.png")


X = noisy_data
DX = Array(solution(solution.t, Val{1}))

prob_nn2 = ODEProblem(dudt_,u0, tspan2, res2_uode.minimizer)
_sol_uode = solve(prob_nn2, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 1)
p_uode = scatter(solution_extrapolate, vars=[2,3,4], legend = :topleft, label=["Данные: Exposed" "Данные: Infected" "Данные: Recovered"], title="Экстраполяция Universal ODE")
plot!(p_uode,_sol_uode, lw = 5, vars=[2,3,4], label=["UODE: Exposed" "UODE: Infected" "UODE: Recovered"])
plot!(p_uode,[20.99,21.01],[0.0,maximum(hcat(Array(solution_extrapolate[2:4,:]),Array(_sol_uode[2:4,:])))],lw=5,color=:black,label="Граница тренировочных данных")

savefig("repos\\UDE_proj\\pres\\universalode_extrapolation.png")

### UODE + SINDy

@variables u[1:3]
polys = []
for i ∈ 0:2, j ∈ 0:2, k ∈ 0:2
    push!(polys, u[1]^i * u[2]^j * u[3]^k)
end

polys
h = [cos.(u)...; sin.(u)...; unique(polys)...]
basis = Basis(h, u)

X = noisy_data
DX = Array(solution(solution.t, Val{1}))
S,E,i,R,N,D,C = eachrow(X)
F,β0,α,κ,μ,_,γ,d,λ = p_
L = β.(0:tspan[end],β0,D,N,κ,α).*S.*i./N
L̂ = vec(ann([S./N i D./N]',res2_uode.minimizer))
X̂ = [S./N i D./N]'


X_ext = noisy_data_extrapolate
DX_ext = Array(solution(solution_extrapolate.t, Val{1}))
S,E,i,R,N,D,C = eachrow(X_ext)
F,β0,α,κ,μ,_,γ,d,λ = p_
L_ext = β.(0:tspan2[end],β0,D,N,κ,α).*S.*i./N


scatter(L,title="UODE vs Исходная функция в системе",label="Исходное слагаемое")
plot!(L̂,label="Предсказание UODE")

savefig("repos\\UDE_proj\\pres\\uode_estimated_exposure.png")
# savefig("uode_estimated_exposure.pdf")


#### NOTE: реализовать то, что находится ниже, у меня пока что не удалось до конца

thresholds = Float32.(exp10.(-6:0.1:1))
opt = STLSQ(thresholds)


# тут код с X и L: определяем проблему с правой частью
# на тот самый подгоняемый член и делаем СИНДи
# возможно придётся добавить функции для выполнения СИНДи, а то че-то
# местные не робят....


problem_direct = ContinuousDataDrivenProblem(X[2:4, :],DX[2:4, :])
Ψ_direct = solve(problem_direct, basis, opt, maxiter = 50000, progress = true, denoise = true, normalize = true)
println(Ψ_direct.basis)
println(Ψ_direct.parameters)
println(result(Ψ_direct))

prob = ODEProblem(Ψ_direct, u0[2:4], tspan2, Ψ_direct.parameters)
sol_Ψ_direct = solve(prob)
scatter(X_ext[2:4,:]')
plot!(sol_Ψ_direct)


problem_ideal = ContinuousDataDrivenProblem(X_ext[2:4,:],DX_ext[2:4,:])
Ψ_ideal = solve(problem_ideal, basis, opt, maxiter = 50000, progress = true, denoise = true, normalize = true)
println(Ψ_ideal.basis)
println(Ψ_ideal.parameters)
println(result(Ψ_ideal))

prob = ODEProblem(Ψ_ideal, u0[2:4], tspan2, Ψ_ideal.parameters)
sol_Ψ_ideal = solve(prob)
scatter(X_ext[2:4,:]')
plot!(sol_Ψ_ideal)



problem_ideal_L = ContinuousDataDrivenProblem(X_ext[2:4,:], L_ext[:])
Ψ_ideal_L = solve(problem_ideal_L, basis, opt, maxiter = 50000, progress = true, denoise = true, normalize = true)
println(result(Ψ_ideal_L))



problem = ContinuousDataDrivenProblem(X̂[:, :],L̂[:])
Ψ = solve(problem, basis, opt, maxiter = 50000, progress = true, normalize=true, denoise=true, eval_expression=true)
println(Ψ.basis)
println(Ψ.parameters)
println(result(Ψ))

Ψ(u0[2:4],Ψ.parameters)
Ψ(u0[2:4])


## финал
function approx(u,p,t)
    S,E,i,R,N,D,C = u
    F,β0,α,κ,μ,σ,γ,d,λ = p
    z = Ψ([S/N,i,D/N])
    dS = -β0*S*F/N - z[1] - μ*S
    dE = β0*S*F/N + z[1] - (σ+μ)*E
    dI = σ*E - (γ+μ)*i
    dR = γ*i - μ*R
    dN = -μ*N
    dD = d*γ*i - λ*D
    dC = σ*E

    [dS,dE,dI,dR,dN,dD,dC]
end


a_prob = ODEProblem{false}(approx, u0, tspan2, p_)
a_solution = solve(a_prob, Tsit5())

# p_uodesindy = scatter(solution_extrapolate, vars=[2,3,4], legend = :topleft, label=["True Exposed" "True Infected" "True Recovered"])
# plot!(p_uodesindy,a_solution, lw = 5, vars=[2,3,4], label=["Estimated Exposed" "Estimated Infected" "Estimated Recovered"])
# plot!(p_uodesindy,[20.99,21.01],[0.0,maximum(hcat(Array(solution_extrapolate[2:4,:]),Array(_sol_uode[2:4,:])))],lw=5,color=:black,label="Training Data End")
