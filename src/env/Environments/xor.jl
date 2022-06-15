using LinearAlgebra

using ...NeuroEvolution

export Xor

"""
`Environment` implementation that represents the xor-problem.
Tests a neural network to solve a xor-gate.

# Interface
## Input
List of two boolean numbers representing the gate inputs.
## Output
List of two numbers
1.  Certainty that the result should be false.
2.  Certainty that the result should be true.
"""
mutable struct Xor <: Environment
    "Total score that was reached by the neural network."
    Score::Float64
    "Expected result the neural network should return."
    Target::Bool

    Xor() = new(0.0, false)
end

NeuroEvolution.inputform(::Type{Xor}) = zeros(Float64, 2)
NeuroEvolution.outputform(::Type{Xor}) = zeros(Float64, 2)

NeuroEvolution.process!(env::Xor, output) = isnothing(output) ? process!(env) : process!(env, convert(Vector{Float64}, output))
function NeuroEvolution.process!(env::Xor, output::Union{Nothing, Vector{Float64}} = nothing)
    if isnothing(output)
        env.Score = 0.0
    else
        env.Score -= norm(output - [~env.Target, env.Target])
    end
    input = rand(Bool, 2)
    env.Target = xor(input...)
    return convert(Vector{Float64}, input)
end

NeuroEvolution.properties(::Type{Xor}) = [:Default, :Score]
NeuroEvolution.fitness(env::Xor, ::Property"Default") = env.Score
NeuroEvolution.fitness(env::Xor, ::Property"Score") = env.Score