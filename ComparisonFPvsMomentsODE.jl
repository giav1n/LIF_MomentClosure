using HDF5, ProgressMeter, Interpolations, DifferentialEquations, LinearAlgebra
# Make sure "NeuralFields.jl" is in the same directory or provide the full path
include("NeuralFields.jl") 

# ============================================================================
# 1. SETUP & FOKKER-PLANCK INTEGRATION
# ============================================================================
println("Starting Fokker-Planck integration...")

# -- Physical Constants & Network Parameters --
dt = 0.01 / 1000  # [s]
K = 10000  
θ, H = 20.0, 0.0
J = 0.38 * θ / K  
τ, τ0 = 0.02, dt  
τD = τ * 0.1  

# -- Input Parameters --
μ0_val = 1.05 * θ / τ
σ0_val = 0.133 * θ / sqrt(τ)
ν0 = NeuralFields.Φ(μ0_val * τ, σ0_val * sqrt(τ), θ, H, τ, τ0)
println("Stationary firing rate: $ν0")

μExt = μ0_val - K * J * ν0
σExt2 = σ0_val^2 - K * (J^2) * ν0
Vmin = θ - 2 * abs(θ - H)
NV = 5000
params = Dict("dt"=>dt, "τ"=>τ, "τ0"=>τ0, "τD"=>τD, "τC"=>τ, 
              "δ"=>dt, "θ"=>θ, "H"=>H, "Vm"=>Vmin, "NV"=>NV, "g"=>0, "N"=>1)

pop = NeuralFields.DefineSinglePopulation(params)
S = NeuralFields.InitializeState(pop)

# -- Simulation Control --
Life = 40 * τ 
steps = Int(round(Life / dt))
νt_fp = 0.0

# -- Arrays for Storage --
rFP = zeros(steps)
mFP = zeros(steps)
vFP = zeros(steps)

# -- External Drive --
f_freq = 0.12 * ν0
external(t) = μExt * 0.12 * sin(2 * π * t * f_freq)
ΔV = pop.Vc[3] - pop.Vc[2]
tFP = range(0.0, stop=Life, length=steps)

# -- Movie Recording Storage --
# Storing snapshots of the probability density p(V)
tFpmovie = Float64[]
FPmovie = Vector{Float64}[]
kk = 1
ΔMovie = Int(round(0.005 / dt))

@showprogress "Fokker-Planck Loop" for n = 2:steps 
    global kk += 1
    global νt_fp
    
    μ_step = μExt + external(n * dt) + K * J * νt_fp
    σ2_step = σExt2 + K * (J^2) * νt_fp
    NeuralFields.IntegrateFP!(pop, μ_step, σ2_step, S, false)
    
    ν = S[pop.NV + 1]
    νt_fp += (dt / τD) * (ν - νt_fp)
    
    # Store scalar metrics
    rFP[n] = ν
    p = S[1:pop.NV]
    mFP[n] = sum(pop.Vc .* p) * ΔV / θ
    vFP[n] = sum((pop.Vc .- mFP[n] .* θ).^2 .* p) * ΔV / (θ^2)
    
    # Store movie frame
    if kk == ΔMovie
        push!(FPmovie, copy(p))
        push!(tFpmovie, n * dt)
        kk = 1
    end
end

# Normalize time and rate for plotting consistency
tFP_norm = tFP ./ τ
rFP_norm = rFP .* τ
##
# ============================================================================
# 2. LOW DIMENSIONAL DYNAMICS (ERE)
# ============================================================================
println("Starting Low Dimensional Dynamics...")

# -- Constants for LD --
α = 20
H_ld, θ_ld, τ_ld, τD_ld = 0, 1, 1, 0.1

# -- Paths to Interpolation Tables (Update these if paths change) --
isi_path = "ISImoments.h5"
spec_path = "quantities_spectral.h5"

# -- Setup Interpolations --
d_isi = read(h5open(isi_path))
μ_isi, σ_isi = d_isi["Mu"] ./ α, d_isi["Sig"] ./ α
T_isi, T2_isi = d_isi["T"], d_isi["T2"]

Φ_grid = zeros(size(T_isi))
for i in 1:size(T_isi, 1), j in 1:size(T_isi, 2)
    Φ_grid[i, j] = 1 / T_isi[i, j]
end
fΦ = LinearInterpolation((σ_isi, μ_isi), Φ_grid, extrapolation_bc=Flat())

qs = read(h5open(spec_path))
μA, σA = qs["mu"], qs["sigma"] * sqrt(α) / α
λall = qs["lambda_all"] * α
fhλ = [LinearInterpolation((σA, μA), λall[:, :, n], extrapolation_bc=Flat()) for n in 1:10]

# -- ERE Helper Functions --
function GetUₖₙ(H, θ, μ, σ, λ)
    nλ = length(λ)
    U = complex(zeros(nλ, nλ))
    for n in 1:nλ
        U[1, n] = (H - θ) / (λ[n] + 1)
        U[2, n] = (H^2 - θ^2 + 2 * μ * U[1, n]) / (λ[n] + 2)
    end
    if nλ > 2
        for k in 3:nλ, n in 1:nλ
            U[k, n] = (H^k - θ^k + k * μ * U[k-1, n] + (k * (k-1) / 2) * (σ^2) * U[k-2, n]) / (λ[n] + k)
        end
    end
    return U
end

function Getu₀(H, θ, μ, σ, Φ_val, τ, nk)
    u₀ = zeros(nk)
    u₀[1] = (H - θ) * Φ_val * τ + μ
    u₀[2] = μ * u₀[1] + ((H^2 - θ^2) * Φ_val * τ + (σ^2)) / 2
    for n in 3:nk
        u₀[n] = μ * u₀[n-1] + (H^n - θ^n) * Φ_val * τ / n + ((n - 1) / 2) * (σ^2) * u₀[n-2]
    end
    return u₀
end

Getλ(μ, σ, nλ) = complex([fhλ[k](σ, μ) for k in 1:nλ])

function Getγ(H, θ, μ, σ, λ)
    A = inv(GetUₖₙ(H, θ, μ, σ, λ))
    return [real(sum(A[:, n])) for n in 1:length(λ)]
end

function Flux(du, u, p, t)
    μ0, σ0, K, J, I0, ω0, nλ = p
    nλ = Int(round(nλ))
    
    μ = μ0 + K * J * u[end] + I0 * sin(ω0 * t)
    σ = sqrt(max(σ0^2 + K * (J^2) * u[end], 0))
    
    λ = Getλ(μ * τ_ld, σ * sqrt(τ_ld), 2 * nλ)
    ν0_curr = fΦ(σ * sqrt(τ_ld), μ * τ_ld)
    u₀ = Getu₀(H_ld, θ_ld, μ * τ_ld, σ * sqrt(τ_ld), ν0_curr, τ_ld, nλ * 2)
    δu = u[1:end-1] - u₀
    γ = Getγ(H_ld, θ_ld, μ * τ_ld, σ * sqrt(τ_ld), λ)
    
    ν = max(ν0_curr + (1 / τ_ld) * dot(γ, δu), 0)
    
    du[1] = (-u[1] + μ * τ_ld + (H_ld - θ_ld) * ν * τ_ld) / τ_ld
    du[2] = (-2 * u[2] + 2 * μ * τ_ld * u[1] + (H_ld^2 - θ_ld^2) * ν * τ_ld + τ_ld * (σ^2)) / τ_ld
    
    if nλ > 1
        for k in 3:(2*nλ)
            du[k] = (-k * u[k] + k * μ * τ_ld * u[k-1] + (H_ld^k - θ_ld^k) * ν * τ_ld + (k * (k - 1) * (σ^2) / 2) * u[k-2]) / τ_ld
        end
    end
    du[end] = (ν - u[end]) / τD_ld
end

function compute_ν(sol, p, τ, H, θ)
    μ0, σ0, K, J, I0, ω0, nλ = p
    nλ = Int(round(nλ))
    times = sol.t
    ν_values = zeros(length(times))
    
    for (i, t) in enumerate(times)
        u = sol.u[i]
        μ = μ0 + K * J * u[end] + I0 * sin(ω0 * t)
        σ = sqrt(max(σ0^2 + K * (J^2) * u[end], 0))
        λ = Getλ(μ * τ, σ * sqrt(τ), 2 * nλ)
        ν0_curr = fΦ(σ * sqrt(τ), μ * τ)
        u₀ = Getu₀(H, θ, μ * τ, σ * sqrt(τ), ν0_curr, τ, nλ * 2)
        δu = u[1:end-1] - u₀
        γ = Getγ(H, θ, μ * τ, σ * sqrt(τ), λ)
        ν_values[i] = max(ν0_curr + (1 / τ) * dot(γ, δu), 0)
    end
    return times, ν_values
end

# -- Run LD Simulations --
μ0_ld, σ0_ld = 1.05, 0.133
ν0_ld = 0.4
β_ld = 0.12 * ν0_ld
Life_ld = 300.0
K_ld, J_ld = 1000, 0.38 / 1000

function redefμσ(μ, σ, τ, K, J, ν0)
    μt = μ - K * J * ν0
    σt2 = σ^2 - K * (J^2) * ν0
    return μt, sqrt(σt2)
end

μx, σx = redefμσ(μ0_ld, σ0_ld, τ_ld, K_ld, J_ld, ν0_ld)
I0_ld = μx * 0.12
ω0_ld = 2π * β_ld
tLD = range(0, stop=Life_ld, step=0.001)

results_ld = Dict()

# We only need n=1 and n=3 for the plot, but you can add others if needed
for nλ_val in [1, 3] 
    println("  Running LD for nλ = $nλ_val")
    u0_ode = vcat(Getu₀(H_ld, θ_ld, μ0_ld, σ0_ld, ν0_ld, τ_ld, 2 * nλ_val), ν0_ld)
    p_ode = [μx, σx, K_ld, J_ld, I0_ld, ω0_ld, nλ_val]
    prob = ODEProblem(Flux, u0_ode, (0.0, Life_ld), p_ode)
    sol = solve(prob, saveat=tLD)
    
    x_sol = hcat(sol.u...)
    _, ν_inst = compute_ν(sol, p_ode, τ_ld, H_ld, θ_ld)
    
    # Store relevant fields
    results_ld[nλ_val] = Dict(
        "m" => x_sol[1, :],
        "v" => x_sol[2, :] - x_sol[1, :].^2,
        "nu" => ν_inst
    )
end 


##. 
using Plots
plot(tFP_norm, rFP_norm, label="Fokker-Planck", xlabel="Time (τ)", ylabel="Firing Rate (ντ)", title="Firing Rate Comparison",color=:black)
plot!(tLD, results_ld[1]["nu"] .* τ_ld, label="LD nλ=1", linestyle=:dash,linewidth=2)
plot!(tLD, results_ld[3]["nu"] .* τ_ld, label="LD nλ=3", linestyle=:dot,linewidth=2)
xlims!(15, 40)
ylims!(0, 1.0)