using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, Optim
using DiffEqFlux, Flux
using Plots
gr()
using JLD2, FileIO
using Statistics
using DelimitedFiles
using DifferentialEquations
# Set a random seed for reproduceable behaviour
using Random
Random.seed!(5443)

## работа с данными

hudson_bay_data = readdlm("repos\\UDE_proj\\hudson_bay_data.dat", '\t', Float32, '\n')

Xₙ = Matrix(transpose(hudson_bay_data[:, 2:3]))
t = hudson_bay_data[:, 1] .- hudson_bay_data[1, 1]

xscale = maximum(Xₙ, dims =2)
Xₙ .= 1f0 ./ xscale .* Xₙ

tspan = (t[1], t[end])

scatter(t, transpose(Xₙ), xlabel = "t", ylabel = "x(t), y(t)")
plot!(t, transpose(Xₙ), xlabel = "t", ylabel = "x(t), y(t)")

savefig("repos\\UDE_proj\\pres\\data.png")

## SINDY + сглаживание

full_problem = ContinuousDataDrivenProblem(Xₙ, t, DataDrivenDiffEq.GaussianKernel())

plot(full_problem.t, full_problem.X')
plot(full_problem.t, full_problem.DX')

@variables u[1:2]

b = [polynomial_basis(u, 2); sin.(u); cos.(u)]
basis = Basis(b, u)

λ = Float32.(exp10.(-7:0.1:5))
opt = STLSQ(λ)

full_res = solve(full_problem, basis, opt, maxiter = 10000, progress = true, denoise = true, normalize = true)

println(full_res)
println(result(full_res))

problem = ODEProblem(full_res, Xₙ[:,1], tspan, full_res.parameters)
sol = solve(problem)

scatter(t, transpose(Xₙ), xlabel = "t", ylabel = "x(t), y(t)")
plot!(sol)

savefig("repos\\UDE_proj\\pres\\sindy_solo.png")

## попробуем сетку

rbf(x) = exp.(-(x.^2))

U = FastChain(
    FastDense(2,5,rbf), FastDense(5,5, rbf), FastDense(5,5, tanh), FastDense(5,2)
)

p = [rand(Float32,2); initial_params(U)]

function ude_dynamics!(du,u, p, t)
    # считаем, что добыча экспоненциально растёт
    # а хищник наоборот экспоненциально умирает
    # и просто добавялем какое-то дополнительное слагаемое, которое не знаем
    û = U(u, p[3:end])
    du[1] = p[1]*u[1] + û[1]
    du[2] = -p[2]*u[2] + û[2]
end

prob_nn = ODEProblem(ude_dynamics!,Xₙ[:, 1], tspan, p)

function predict(θ, X = Xₙ[:,1], T = t)
    Array(solve(prob_nn, Vern7(), u0 = X, p=θ,
                tspan = (T[1], T[end]), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

group_size = 5
continuity_term = 200.0f0

function loss(data, pred)
	return sum(abs2, data - pred)
end

function shooting_loss(p)
    return multiple_shoot(p, Xₙ, t, prob_nn, loss, Vern7(),
                          group_size; continuity_term)
end

function loss(θ)
    X̂ = predict(θ)
    sum(abs2, Xₙ - X̂) / size(Xₙ, 2) + convert(eltype(θ), 1e-3)*sum(abs2, θ[3:end]) ./ length(θ[3:end])
end

losses = Float32[]

callback(θ,args...) = begin
	l = loss(θ) # Equivalent L2 loss
    push!(losses, l)
    if length(losses)%5==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end


res1 = DiffEqFlux.sciml_train(shooting_loss, p, ADAM(0.1f0), cb=callback, maxiters = 100)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

res2 = DiffEqFlux.sciml_train(shooting_loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 500)
println("Training loss after $(length(losses)) iterations: $(losses[end])")

res3 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 100000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

res4 = DiffEqFlux.sciml_train(loss, res3.minimizer, BFGS(initial_stepnorm=0.001f0), cb=callback, maxiters = 100000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")

res5 = DiffEqFlux.sciml_train(loss, res4.minimizer, BFGS(initial_stepnorm=0.00001f0), cb=callback, maxiters = 100000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")


losses

pl_losses = plot(1:101, losses[1:101], yaxis = :log10, xaxis = :log10, xlabel = "Итерации", ylabel = "Потери", label = "ADAM (Shooting)", color = :blue)
plot!(102:123, losses[102:123], yaxis = :log10, xaxis = :log10, xlabel = "Итерации", ylabel = "Потери", label = "BFGS ", color = :red)
plot!(123:length(losses), losses[123:end], color = :black, label = "BFGS (L2)")

savefig("repos\\UDE_proj\\pres\\loss_train_lotka.png")

p_trained = res4.minimizer

tsample = t[1]:0.5:t[end]
X̂ = predict(p_trained, Xₙ[:,1], tsample)

pl_trajectory = scatter(t, transpose(Xₙ), label = ["Измерения x(t)" "Измерения y(t)"], xlabel = "t", ylabel = "x(t), y(t)")
plot!(tsample, transpose(X̂),label = ["UDE x(t)" "UDE y(t)"])
savefig("repos\\UDE_proj\\pres\\ude_predict_lotka_new.png")

Ŷ = U(X̂,p_trained[3:end])

pl_reconstruction = scatter(tsample, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["Недостающие слагаемые" nothing])
plot!(tsample, transpose(Ŷ), color = :red, lw = 2, style = :dash, label = [nothing nothing])
savefig("repos\\UDE_proj\\pres\\ude_missing_term.png")
pl_missing = plot(pl_trajectory, pl_reconstruction, layout = (2,1))
savefig("repos\\UDE_proj\\pres\\ude_missing_term_predict_lotka_new.png")

## SINDy снова

λ = Float32.(exp10.(-7:0.1:5))
opt = STLSQ(λ)

nn_problem = ContinuousDataDrivenProblem(X̂, Ŷ, t=tsample)
nn_res = solve(nn_problem, basis, opt, maxiter = 20000, progress = true, normalize = true, denoise = true)
println(nn_res)
println(result(nn_res))
println(nn_res.parameters)


function recovered_dynamics!(du,u, p, t)
    û = nn_res(u, p[3:end])
    du[1] = p[1]*u[1] + û[1]
    du[2] = -p[2]*u[2] + û[2]
end

p_model = [p_trained[1:2];parameters(nn_res)]

estimation_prob = ODEProblem(recovered_dynamics!, Xₙ[:, 1], tspan, p_model)

sys = modelingtoolkitize(estimation_prob);
dudt = ODEFunction(sys);
estimation_prob = ODEProblem(dudt,Xₙ[:, 1], tspan, p_model)
estimate = solve(estimation_prob, Tsit5(), saveat = t)

plot(estimate)


##

prob = ODEProblem(recovered_dynamics!, Xₙ[:, 1], tspan, p_model)
sol_d = solve(prob,p=p_model)
plot(sol_d)

params = Flux.params(p_model)

using DiffEqSensitivity

function predict_func()
  solve(prob,MethodOfSteps(Tsit5()),p=p_model,saveat = t)
end

loss_func() = sum(abs2,Array(predict_func()) .- Xₙ)
loss_func()

data = Iterators.repeated((), 10)
opt = ADAM(0.1)

cb_func = function ()
  display(loss_func())
  display(plot(solve(remake(prob,p=p_model),MethodOfSteps(Tsit5()),saveat=t)))
end

cb_d()

Flux.train!(loss_func, [p_model], data, opt, cb = cb_func)



p_fitted = p_model

##
function loss_fit(θ)
    X̂ = Array(solve(estimation_prob, Tsit5(), p = θ, saveat = t))
    sum(abs2, X̂ .- Xₙ)
end

callback(θ,args...) = begin
	l = loss_fit(θ)
    println("Current loss $l")
	# println("training")
    false
end

res_fit = DiffEqFlux.sciml_train(loss_fit, p_model, BFGS(initial_stepnorm = 0.1f0), cb=callback, maxiters = 1000000)
p_fitted = res_fit.minimizer

estimate_rough = solve(estimation_prob, Tsit5(), saveat = 0.1*mean(diff(t)), p = [p_trained[1:2];parameters(nn_res)])
estimate = solve(estimation_prob, Tsit5(), saveat = 0.1*mean(diff(t)), p = p_fitted)

p_fitted

pl_fitted = plot(t, transpose(Xₙ), style = :dash, lw = 2, label = ["Данные x(t)" "Данные y(t)"], xlabel = "t", ylabel = "x(t), y(t)")
plot!(estimate_rough, label = ["UODE: x(t)" "UODE: y(t)"])
plot!(estimate, lw=3, label = ["UODE + fit: x(t)" "UODE + fit: y(t)"])

savefig("repos\\UDE_proj\\pres\\recover_fit.png")


t_long = (0.0f0, 50.0f0)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.25f0, tspan = t_long,p = p_fitted)
plot(estimate_long.t, transpose(estimate_long[:,:]), lw=3, label = ["Интерполяция UODE + fit: x(t)" "Интерполяция UODE + fit: y(t)"], xlabel = "t", ylabel = "x(t),y(t)")
plot!(t, transpose(Xₙ), lw = 2)
scatter!(t, transpose(Xₙ),label = ["Данные x(t)" "Данные y(t)"], xlabel = "t", ylabel = "x(t), y(t)")
plot!([19.99,20.01],[0.0,maximum(Xₙ)*1.25],lw=4,color=:black, label = nothing)
annotate!([(10.0,maximum(Xₙ)*1.25,text("Тренировочная \nВыборка",12 , :center, :top, :black, "Helvetica"))])

savefig("repos\\UDE_proj\\pres\\full_lotka.png")
