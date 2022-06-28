export Genotype, Phenotype, Environment, Trainer, Evaluation, Selector, Mutator, Unit
export express, process!, reset!, fitness, train!, evaluate!, push!, mutate, inherit, nparents, nchildren, inputform, outputform, properties
export NoTrainer, Property, Average, Transform
export @Property_str

using Distributed, DistributedArrays
using DataStructures

using Random

# FIXME: docs: TYPES macro doesn't work with UnionAll types

"""
Base class for all genomes.

# Interface
`express(genome::G)::P`

should create a `P <: Phenotype` using a `G <: Genotype`.

- `genome`: the source genome.
"""
abstract type Genotype end

"""
Base class for all neural networks.

# Interface
`process!(network::P, input::T)::S`

should evaluate a `P <: Phenotype` for a given input of type `T` and return the result of type `S`.

- `network`: the network to evaluate.
- `input`: the given input.

`reset!(network::P)`

should reset the internal state of a `P <: Phenotype`.

- `network`: the network to reset.
"""
abstract type Phenotype end

"""
Base class for all environments.

# Interface
`process!(environment::E)::S`

should initialize the `E <: Environment` and return the initial state of type `S`.

- `environemnt`: the environment to initialize or reset.

`process!(environment::E, output::T)::Union{Nothing, S}`

should update the state of a `E <: Environment` for a given neural network result of type `T` and return the updated state of type `S` or `nothing` if a failstate is reached.

- `environment`: the environment to update.
- `output`: the output of the neural network.

`inputform(type::Type{E})::S`

should return an example of type `S` of the neural network input that a `E <: Environment` produces.

- `type`: the type of environment.

`outputform(type::Type{E})::T`

should return an example of type `T` of the neural network output that a `E <: Environment` expects as input.

- `type`: the type of environment.

`fitness(environment::E, evaluation::F)::Float64`

should return the value of a `F <: Evaluation` for a `E <: Environment`.
This should be at least implemented for `F = Property"Default"`.

- `environment`: the environment to query
- `evaluation`: the kind of evaluation of query for.

`properties(environment::E)::Vector{Symbol}`

should return a list of symbols that are valid in a `fitness(::E, Property{X}`) call for a `E <: Environment`.
By default this will return `[:Default]`.

- `environment`: the environment to query.
"""
abstract type Environment end

"""
Base class for all neural network trainers.

# Interface
`train!(trainer::T, network::P, environment::E)`

should prepare a `P <: Phenotype` to be evaluated in a `E <: Environment` using a `T <: Trainer`

- `trainer`: the trainer to use.
- `network`: the network to be trained.
- `environment`: the environment to train for.
"""
abstract type Trainer end

"""
Base class for all evaluations for a environemnt,

# Interface
`fitness(environment::Environment, evaluation::F)::Float64`

should return the value of a `F <: Evaluation` for a general `Environment`
This should be at least implemented for `F = Property"Default"`.

- `environment`: a general environment
- `evaluation`: the kind of evaluation of query for.
"""
abstract type Evaluation end

"""
Base class for all genome selectors.

# Interface
`Base.push!(selector::S, genome::G, fitness::Float64)`

should add a `G <: Genotype` with given fitness to a `S <: Selector`

- `selector`: the selector to add to.
- `genome`: the genome to add.
- `fitness`: the fitness of the genome.

`Base.iterate(selector::S, state=nothing)`

should iterate over all possible parents.

see `Base.iterate`

`Base.length(selector::S)`

should return the number of viable parents.

see `Base.length`

`Base.getindex(selector::S, index::Int)`

should get the `index`-th parent.

see `Base.getindex`

`Base.empty!(selector::S)`

should reset the `S <: Selector` for the next generation.

see `Base.empty!`

`Base.eltype(::Type{S})::G`

should return the type of parents stored.

see `Base.eltype`
"""
abstract type Selector end

"""
Base class for all mutators of genomes.
- `nP`: number of parents needed for inheritance.
- `nC`: number of children created in inheritance.

# Interface
`mutate(mutator::M, genome::G)::G`

should mutate a `G <: Genome` using a `M <: Mutator` and return the new genome.

- `mutator`: the mutator to use.
- `genome`: the genome to mutate

`inherit(mutator::M, selector::S)::Vector{G}`

should select `nP` parent from a `S <: Selector` and return `nC` children of type `G <: Genome` using a `M <: MUtator`.

- `mutator`: the mutator to use.
- `selector`: the source to select parent from.
"""
abstract type Mutator{nP, nC} end

"""
Creates a new `Phenotype` based on the given `Genotype`.

see `Genotype`
"""
function express(genome::G)::P where {G <: Genotype, P <: Phenotype}
    throw(MissingException("express(::$G) not implemented"))
end

"""
Evaluates a `Phenotype` given a specific input and returns the result.

see `Phenotype`
"""
function process!(network::P, input::T) where {P <: Phenotype, T}
    throw(MissingException("process!(::$P, ::$T) not implemented"))
end

"""
Evaluates a `Phenotype` given a specific input and returns the result.

see `Phenotype`
"""

(network::Phenotype)(input...) = 
    (res = process!(network, [input...]); length(res) == 1 ? res[1] : res)

"""
Evaluates a `Phenotype` given a specific input and returns the result.

see `Phenotype`
"""
(network::Phenotype)(input::AbstractArray{T}) where {T} = process!(network, input)

"""
Resets a `Phenotype`.

see `Phenotype`
"""
function reset!(network::P) where {P <: Phenotype}
    throw(MissingException("reset!(::$P) not implemented"))
end

"""
Simulates the `Environment` for a single time-step.
The second argument sould be the result of `process!(::P, ::T)`
or `nothing` for initialization/reset.
Returns the initial/updated state or `nothing` if a failstate is reached.

see `Environment`
"""
function process!(environment::E, output::Union{Nothing, T} = nothing) where {E <: Environment, T}
    isnothing(output) && throw(MissingException("process!(::$E) not implemented"))
    throw(MissingException("process!(::$E, ::$T) not implemented"))
end
(environment::Environment)(output::Union{Nothing, T} = nothing, rest...) where {T} = 
    process!(environment, isnothing(output) ? nothing : [output, rest...])
(environment::Environment)(output::AbstractArray{T}) where {T} = 
    process!(environment, output)

"""
Returns an example of neural network input produced by an `Environemnt`.

see `Environment`
"""
function inputform(environment::Type{E}) where {E <: Environment}
    throw(MissingException("inputform(::Type{$E}) not implemented"))
end

"""
Returns an example of neural network input produced by an `Environemnt`.

see `Environment`
"""
inputform(environment::E) where {E <: Environment} = inputform(E)

"""
Returns an example of neural network output expected by an `Environment`.

see `Environemnt`
"""
function outputform(environment::Type{E}) where {E <: Environment}
    throw(MissingException("outputform(::Type{$E}) not implemented"))
end

"""
Returns an example of neural network output expected by an `Environment`.

see `Environemnt`
"""
outputform(environment::E) where {E <: Environment} = outputform(E)

"""
Returns the fitness value associated to the current state of the `Environment`.

see `Environement`, `Evaluation`
"""
function fitness(environment::Env, evaluation::Eval) where {Eval <: Evaluation, Env <: Environment}
    throw(MissingException("fitness(::$Env, ::$Eval) not implemented"))
end

"""
Returns a list of all supported properties for a `Environment`.

see `Environment`
"""
properties(environment::Type{E}) where {E <: Environment} = [:Default]

"""
Trains a `Phenotype` in the `Environment` using a `Trainer`.

see `Trainer`
"""
function train!(trainer::T, network::P, environment::E) where {T <: Trainer, P <: Phenotype, E <: Environment}
    throw(MissingException("train!(::$T, ::$P, ::$E) not implemented"))
end

"""
Evaluates the performance of the `Phenotype` in a given `Environment` by simulating multiple `steps`.
"""
function evaluate!(evaluation::Eval, network::P, environment::Env, steps::Int = 1) where {Eval <: Evaluation, P <: Phenotype, Env <: Environment}
    val = process!(environment, nothing)
    for _ in 1:steps
        val = process!(network, val)
        val = process!(environment, val)
        isnothing(val) && break # failstate
    end
    return fitness(environment, evaluation)
end

"""
Adds a `Genotype` to a `Selector` using a score of `fitness`.

see `Selector`
"""
function Base.push!(selector::S, genome::G, fitness::T) where {S <: Selector, G <: Genotype, T}
    throw("No implementation for the type $S found!")
end

function Base.iterate(selector::S, state=nothing) where {S <: Selector}
    throw("The type $S needs to support iteration over the final selection!")
end
function Base.length(selector::S) where {S <: Selector}
    throw("The type $S needs to support explicit length of the final selection!")
end
Base.size(selector::S) where {S <: Selector} = (length(selector),)
Base.keys(selector::S) where {S <: Selector} = 1:length(selector)

function Base.getindex(selector::S, index::Int) where {S <: Selector}
    throw("The type $S needs to support indexing into the final selection!")
end

function Base.empty!(selector::S) where {S <: Selector}
    throw("The type $S needs to be able to be reset!")
end
Base.eltype(::Type{S}) where {S <: Selector} = Genotype

"""
Mutates a `Genotype` using a `Mutator`.

see `Mutator`
"""
function mutate(mutator::M, genome::G) where {M <: Mutator, G <: Genotype}
    throw("No implementation for the type $M found!")
end

"""
Mutates a `Genotype` using a `Mutator` multiple times.

see `Mutator`
"""
function NeuroEvolution.mutate(mutator::M, genome::G, count::Int) where {M <: Mutator, G <: Genotype}
    count < 0 && throw("count can't be negative")
    count == 0 && return genome
    for _ in 1:count
        genome = mutate(mutator, genome)
    end
    genome
end

"""
Creates `nchildren(::M)` children based on `nparents(::M)` parents using a `Mutator`.

see `Mutator`
"""
function inherit(mutator::M, selector::S)::Vector{G} where {M <: Mutator, S <: Selector, G <: Genotype}
    throw("No implementation for the type $M found!")
end

"""
The number of parents needed by a `Mutator` for inheritance.
"""
nparents(mutator::Mutator{nP, nC}) where {nP, nC} = nP

"""
The number of children created by a `Mutator` in inheritance.
"""
nchildren(mutator::Mutator{nP, nC}) where {nP, nC} = nC


"""
This class holds all the data needed to train and evolve a population of genomes.
"""
mutable struct Unit{
        Gene <: Genotype, 
        Net <: Phenotype, 
        Env <: Environment,
        Train <: Trainer,
        Eval <: Evaluation,
        Sel <: Selector,
        Mut <: Mutator
    }
    "Population of `Genotype`."
    Genes::DArray{Gene, 1}
    "Population of `Phenotype`."
    Population::DArray{Net, 1}
    "Number of times an `Environment` gets queried per generation."
    StepsPerGeneration::Int
    "Environment to train in."
    Environment::DArray{Env, 1}
    "Trainer"
    Trainer::Train
    "Evaluation to use, `Property\"Default\"` should always work."
    Evaluation::Eval
    "Number of evaluations performed per generation, the fitness is the average of all of them."
    EvaluationCount::Int
    "Whether eligable parents from each generation should be copied to the new generation (without mutations)"
    CopyParents::Bool
    "`Selector` of parents."
    Selector::Sel
    "Number of mutations after each inheritance."
    MutationsPerGeneration::Int
    "`Mutator` to handle mutations and inheritance."
    Mutator::Mut

    function Unit(
        ::Type{Net},
        population_size::Int,
        default_genome::Gene, 
        environment::Env, 
        trainer::Train, 
        evaluation::Eval,
        selector::Sel,
        mutator::Mut;
        copy_parents::Bool = false,
        initial_mutations::Int = 0,
        mutations_per_generation::Int = 1,
        steps_per_generation::Int = 1,
        evaluation_count::Int  = 1
        ) where {
            Gene <: Genotype, 
            Net <: Phenotype, 
            Env <: Environment,
            Train <: Trainer,
            Eval <: Evaluation,
            Sel <: Selector,
            Mut <: Mutator
        }
        @assert population_size > 0 "population_size needs to be positive!"
        @assert steps_per_generation > 0 "steps_per_generation needs to be positive!"
        @assert initial_mutations >= 0 "initial_mutations can't be negative!"
        @assert mutations_per_generation >= 0 "mutations_per_generation can't be negative!"
        @assert evaluation_count > 0 "Need to perform at least one evaluation!"
    
        genes = DArray((population_size,)) do I
            [mutate(mutator, default_genome, initial_mutations) for _ in I[1]]
        end
        population = express.(genes)
        environments = dfill(environment, length(procs(population)))
        new{Gene, Net, Env, Train, Eval, Sel, Mut}(
            genes,
            population,
            steps_per_generation,
            environments,
            trainer,
            evaluation,
            evaluation_count,
            copy_parents,
            selector,
            mutations_per_generation,
            mutator
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", u::Unit)
    println(io, "Unit with ", length(u.Population), " entries: \n ",
        "StepsPerGeneration = ", u.StepsPerGeneration, ", ",
        "EvaluationCount = ", u.EvaluationCount, ", ",
        "CopyParents = ", u.CopyParents, ", ",
        "MutationsPerGeneration = ", u.MutationsPerGeneration, "\n",

        " Genes ", u.Genes, "\n",
        " Population ", u.Population, "\n",
        " Environment ", u.Environment, "\n",
        " Trainer ", u.Trainer, "\n",
        " Evaluation ", u.Evaluation, "\n",
        " Selector ", u.Selector, "\n",
        " Mutator ", u.Mutator)
end

"""
Runs a single generation using the data in a `Unit` and returns the best network and its fitness.
"""
function process!(u::Unit)
    fitness = dzeros(Float64, size(u.Population))
    @sync for p in procs(u.Population)
        @spawnat p begin
            I = localindices(u.Population)
            i0 = first(I[1])
            env = u.Environment[:L][1]
            fit = fitness[:L]
            for i in I[1]
                train!(u.Trainer, u.Population[i], env)
                f = 0.0
                for _ in 1:u.EvaluationCount
                    f += evaluate!(u.Evaluation, u.Population[i], env, u.StepsPerGeneration)
                    reset!(u.Population[i])
                end
                fit[i - i0 + 1] = f / u.EvaluationCount
            end
        end
    end
    
    (max_fitness, index) = findmax(fitness)
    fittest = u.Population[index]

    empty!(u.Selector)
    for i in 1:length(u.Population)
        push!(u.Selector, u.Genes[i], fitness[i])
    end

    new_population = Queue{eltype(u.Genes)}()
    todo = length(u.Population)
    if u.CopyParents
        for parent in u.Selector
            ismissing(parent) && continue
            enqueue!(new_population, parent)
            todo -= 1
        end
    end

    batch = nchildren(u.Mutator)
    mod(todo, batch) == 0 || throw("Amount of needed children not divisible by nchildren(::$(typeof(u.Mutator)))!")

    counts = div(todo, batch)
    for _ in 1:counts
        for c in inherit(u.Mutator, u.Selector)
            enqueue!(new_population, mutate(u.Mutator, c, u.MutationsPerGeneration))
        end
    end

    u.Genes = distribute(shuffle(collect(new_population)))
    u.Population = map(express, u.Genes)

    return (fittest, max_fitness)
end

"""
`Trainer` implementation that performs no training.
"""
struct NoTrainer <: Trainer
end

train!(trainer::NoTrainer, network::P, environment::E) where {P <: Phenotype, E <: Environment} = nothing

"""
`Evaluation` implementation that represents a property stored in an `Environment`.
"""
struct Property{T} <: Evaluation end
macro Property_str(T)
    :(Property{Symbol($T)})
end
Property() = Property"Default"()

Base.show(io::IO, ::Type{Property{T}}) where {T} = print(io, "Property{", T, "}")
Base.show(io::IO, p::Property{T}) where {T} = print(io, "Property{", T, "}()")

"""
`Evaluation` implementation that represents the (weighted) average over multiple other evaluations.

for example
`Average(Property"Score"(), 2, Property"Time"())`
weights the "Score" property double that of the "Time" property.
"""
struct Average
    "`Evaluation`s used as inputs."
    Evals::Tuple{Vararg{Evaluation}}
    "Weights in the same order as `Evals`."
    Weights::Vector{Float64}
    function Average(args...)
        if all(x -> x isa Evaluation, args)
            return new(args, ones(Float64, length(args)))
        end
        evals = Queue{Evaluation}()
        weights = Queue{Float64}()
        hasweight = false
        for a in args
            if a isa Evaluation
                !hasweight && !isempty(evals) && enqueue!(weights, 1)
                enqueue!(evals, a)
                hasweight = false
            elseif hasweight
                error("Can't pass multiple weights to a single Evaluation!")
            elseif isempty(evals)
                error("No Evaluation found to assign a weight to!")
            else
                enqueue!(weights, convert(Float64, a))
                hasweight = true
            end
        end
        !hasweight && enqueue!(weights, 1)
        new((evals...,), [weights...])
    end
    function Average(args::Dict{Evaluation, Float64})
        new((keys(args)...,), [values(args)...])
    end
end
Average(args::Dict{Evaluation, <: Real}) =
    Average(convert(Dict{Evaluation, Float64}, args))

Base.show(io::IO, eval::Average) = print(io, "Average(", Dict(zip(eval.Evals, eval.Weights)), ")")
function Base.show(io::IO, ::MIME"text/plain", eval::Average)
    println(io, "Average:")
    for i in keys(eval.Evals)
        println(io, " ", eval.Evals[i], " => ", eval.Weights[i])
    end
end

fitness(env::E, eval::Average) where {E <: Environment} =
    sum([fitness(env, eval.Evals[i]) * eval.Weights[i] for i in 1:length(eval.Evals)]) / sum(eval.Weights)

"""
`Evaluation` implementation that represents a function of one or more other evaluations.

for example
```
Transform(Property"Score"(), Property"Time"()) do s, t
    s * t
end
```
represents the product of the "Score" property and the "Time" property.
"""
struct Transform <: Evaluation
    "`Evaluation`s used as input."
    Evals::Tuple{Vararg{Evaluation}}
    "Function that is performed."
    Trafo::Function
    function Transform(f::Function, evals::Evaluation...)
        hasmethod(f, Tuple{Vararg{Float64, length(evals)}}) || error("The given function is not valid!")
        new(evals, f)
    end
end

function Base.show(io::IO, ::MIME"text/plain", eval::Transform)
    println(io, "Transform ", eval.Trafo)
    for child in eval.Evals
        println(io, " ", child)
    end
end

fitness(env::E, eval::Transform) where {E <: Environment} =
    eval.Trafo([fitness(env, e) for e in eval.Evals]...)