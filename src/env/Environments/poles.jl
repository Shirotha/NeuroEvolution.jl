using ...NeuroEvolution

using LinearAlgebra

export Poles

"""
`Environment` implementation that prepresents the N-pole problem.
Tests a neural network to balance `N` poles by moving the base in either direction.

# Interface
## Input
A list of numbers representing the coodinates of the poles relative to the base.
Pole 1 x, Pole 1 y, Pole 2 x, Pole 2 y, ...
## Output
A list with a single number representing the speed that the base should move at.
"""
mutable struct Poles{N} <: Environment
    "Positions of all poles alternating x and y."
    Position::Vector{Float64}
    "Velocity vectors of all poles alternating x and y."
    Velocity::Vector{Float64}
    "Strength of gravity per timestep."
    Gravity::Float64
    "Simulation timestep per step."
    DeltaTime::Float64
    "Number of steps to simulate before the neural network can respond again."
    StepsPerCycle::Int
    """
    When this is positive the inital angles after each reset will be choosen that number and its negative in radians (uniform distribution).
    When this is negative the initial angles after each reset will alternate between this negative number and its postive counterpart.
    """
    MaxInitialAngle::Float64
    "How much time was simulated since the last reset."
    Time::Float64
    "Total score that the neural network has reached, based on angle deviations from being straight."
    Score::Float64

    function Poles{N}(gravity, delta_time, steps_per_cycle, max_initial_angle) where {N}
        N <= 0 && error("Need at least one pole!")
        delta_time <= 0 && error("DeltaTime has to be positive!")
        steps_per_cycle <= 0 && error("Need at least one step per cycle!")
        pos = zeros(Float64, 2N)
        vel = zeros(Float64, 2N)
        delta_time /= steps_per_cycle
        new{N}(pos, vel, gravity * delta_time, delta_time, steps_per_cycle, max_initial_angle, 0.0, 0.0)
    end
end
function Poles{N}(; delta_time = 0.1, steps_per_cycle = 10, gravity = 9.81, max_initial_angle = 0.05, fixed_initial_angle = false) where {N}
    Poles{N}(gravity, delta_time, steps_per_cycle, (fixed_initial_angle ? -1 : 1) * max_initial_angle)
end

NeuroEvolution.inputform(::Type{Poles{N}}) where {N} = zeros(Float64, 2N)
NeuroEvolution.outputform(::Type{Poles{N}}) where {N} = zeros(Float64, 1)

NeuroEvolution.process!(env::Poles{N}, output) where {N} = isnothing(output) ? process!(env) : process!(env, convert(Vector{Float64}, output))
function NeuroEvolution.process!(env::Poles{N}, output::Union{Nothing, Vector{Float64}} = nothing) where {N}
    if isnothing(output)
        pos = Vector{Float64}(undef, 2N)
        if env.MaxInitialAngle > 0
            angles = (2 .* rand(N) .- 1) .* env.MaxInitialAngle
        else
            angles = [(iseven(i) ? 1 : -1) * env.MaxInitialAngle for i in 1:N]
        end
        for i in 1:N
            (s, c) = sincos(angles[i])
            pos[2i - 1] = s
            pos[2i] = c
        end
        env.Position = pos
        env.Velocity = zeros(Float64, 2N)
        env.Time = 0.0
        env.Score = 0.0
    else
        dx = -first(output) * env.DeltaTime
        for _ in 1:env.StepsPerCycle
            r = 1:2
            for _ in 1:N
                x = view(env.Position, r)
                v = view(env.Velocity, r)
                old_x = copy(x)
                v[1] = dx
                v[2] -= env.Gravity
                @. x += v * env.DeltaTime

                x[2] < 0 && return nothing

                normalize!(x)
                @. v = x - old_x
                env.Score -= abs(x[1]) * env.DeltaTime
                r = r .+ 2
            end
            env.Time += env.DeltaTime
        end
    end
    env.Position
end

NeuroEvolution.properties(::Type{<:Poles}) = [:Default, :Time, :Score]
NeuroEvolution.fitness(env::Poles, ::Property"Default") = env.Time
NeuroEvolution.fitness(env::Poles, ::Property"Score") = env.Score
NeuroEvolution.fitness(env::Poles, ::Property"Time") = env.Time
