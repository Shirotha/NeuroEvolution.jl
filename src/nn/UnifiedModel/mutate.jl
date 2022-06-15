using Random

using ...NeuroEvolution

export Mutate

"""
`Mutator` implementation for the unified model.
"""
struct Mutate <: Mutator{1, 1}
    "Source for all random numbers."
    RNG::AbstractRNG
    "Absolute value of weights get clamped to this number"
    MaximumWeight::Float64
    "Chance for weights to be mutated."
    WeightMutationChance::Float64
    "Maximum change of weights relative to a percentage of the current weight."
    WeightMutationChangePercentage::Float64
    "Chance to add a new neuron as a mutation."
    AddNeuronChance::Float64
    "Chance to add a new connection as a mutation."
    RemoveNeuronChance::Float64
    "Chance to remove a neuron as a mutation."
    AddConnectionChance::Float64
    "Chance to remove a connection as a mutation."
    RemoveConnectionChance::Float64
    "Chance for a new connection to be modulated."
    ModulationChance::Float64
    "Weighted list of possible values for `Neuron.Speed`."
    Speeds::Dict{Float64, Float64}
    "Weighted list of possible values for `Neuron.Type`."
    NeuronTypes::Dict{NeuronType, Float64}
    function Mutate(;
        rng::AbstractRNG = RandomDevice(),
        max_weight::Float64 = 1e9,
        weight_mutation_change::Float64 = 0.5,
        weight_change_percent::Float64 = 1.0,
        add_neuron_chance::Float64 = 0.01,
        remove_neuron_chance::Float64 = 0.01,
        add_connection_chance::Float64 = 0.49,
        remove_connection_chance::Float64 = 0.49,
        modulation_chance::Float64 = 0.1,
        neuron_speeds = Dict(1/1 => 1, 1/7 => 1, 1/49 => 1),
        neuron_types = Dict(IDENTITY => 1, SIGMOID => 1, THRESHOLD => 1, RANDOM => 1, CONTROL => 1)
    )
        max_weight <= 0 && error("MaximumWeight has to be positive!")
        0 <= weight_mutation_change <= 1 || error("WeightMutationChance has to be a valid pertentage value!")
        weight_change_percent <= 0 && error("WeightMutationChangePercentage has to be positive!")
        0 <= add_neuron_chance <= 1 || error("AddNeuronChance has to be a valid pertentage value!")
        0 <= remove_neuron_chance <= 1 || error("RemoveNeuronChance has to be a valid pertentage value!")
        0 <= add_connection_chance <= 1 || error("AddConnectionChance has to be a valid pertentage value!")
        0 <= remove_connection_chance <= 1 || error("RemoveConnectionChance has to be a valid pertentage value!")
        add_neuron_chance + remove_neuron_chance + add_connection_chance + remove_connection_chance <= 1 || 
            error("Combined structual change mutations chance can't be greater than 1!")
        0 <= modulation_chance <= 1 || error("ModulationChance has to be a valid pertentage value!")
        isempty(neuron_speeds) && error("Speeds can't be empty!")
        isempty(neuron_types) && error("NeuronTypes can't be empty!")
        new(
            rng, 
            max_weight, 
            weight_mutation_change, 
            weight_change_percent, 
            add_neuron_chance, 
            remove_neuron_chance, 
            add_connection_chance, 
            remove_connection_chance, 
            modulation_chance, 
            neuron_speeds, 
            neuron_types
        )
    end
end

function WeightedChoice(rng::AbstractRNG, choices::AbstractDict{T, W}; total = missing) where {T, W <: Real}
    ismissing(total) && (total = (sum ∘ values)(choices))
    r = total * rand(rng)
    s = zero(W)
    for (choice, weight) in choices
        s += weight
        r < s && return choice
    end
    missing
end

# TODO: maybe find a better way for large genomes (is sorting an ID array faster?)
function SmallestFreeID(g::Genome)
    id = 1
    while true
        id_found = false
        for n in g.Neurons
            if n.ID == id
                id_found = true
                break
            end
        end
        if !id_found
            return id
        end
        id += 1
    end
end

function RandomConnection(mut::Mutator, neurons::Vector{NeuronGene}; from::Int = 0, to::Int = 0)
    if from == 0
        from = neurons[(Int ∘ floor)(rand(mut.RNG) * length(neurons)) + 1].ID
    end
    if to == 0
        to = neurons[(Int ∘ floor)(rand(mut.RNG) * length(neurons)) + 1].ID
    end
    weight = 2rand(mut.RNG) - 1
    mod = 0
    if rand(mut.RNG) < mut.ModulationChance
        mod = neurons[(Int ∘ floor)(rand(mut.RNG) * length(neurons)) + 1].ID
        weight = 1.0
    end
    ConnectionGene(from, to, weight, mod)
end

function MutateWeights(mut::Mutate, g::Genome)
    rand(mut.RNG) < mut.WeightMutationChance || return g
    Genome(g.Neurons, Array{ConnectionGene}([ConnectionGene(c.From, c.To, clamp(
            (2rand(mut.RNG) - 1) * (mut.WeightMutationChangePercentage * c.Weight), -mut.MaximumWeight, mut.MaximumWeight
        ), c.Modulator) for c in g.Connections]))
end


function MutateStructure(mut::Mutate, g::Genome)
    mutation = WeightedChoice(mut.RNG, Dict(
            :add_neuron => mut.AddNeuronChance,
            :remove_neuron => mut.RemoveNeuronChance,
            :add_connection => mut.AddConnectionChance,
            :remove_connection => mut.RemoveConnectionChance
        );
        total = 1
    )
    ismissing(mutation) && return g
    
    index::Int = 0
    if mutation == :add_neuron
        ns = copy(g.Neurons)
        push!(ns, NeuronGene(SmallestFreeID(g), WeightedChoice(mut.RNG, mut.NeuronTypes), WeightedChoice(mut.RNG, mut.Speeds), 0))
        cs = copy(g.Connections)
        push!(cs, RandomConnection(mut, ns, from = ns[end].ID))
        push!(cs, RandomConnection(mut, ns, to = ns[end].ID))
        return Genome(ns, cs)
    elseif mutation == :remove_neuron
        index = floor(rand(mut.RNG) * length(g.Neurons)) + 1
        isinterface(g.Neurons[index]) && return g
        ns = copy(g.Neurons)
        id = ns[index].ID
        deleteat!(ns, index)
        cs = copy(g.Connections)
        for i in reverse(1:length(cs))
            if cs[i].From == id || cs[i].To == id
                deleteat!(cs, i)
            end
        end
        return Genome(ns, cs)
    elseif mutation == :add_connection
        cs = copy(g.Connections)
        push!(cs, RandomConnection(mut, g.Neurons))
        return Genome(g.Neurons, cs)
    elseif mutation == :remove_connection
        isempty(g.Connections) && return g
        cs = copy(g.Connections)
        index = floor(rand(mut.RNG) * length(g.Connections)) + 1
        deleteat!(cs, index)
        return Genome(g.Neurons, cs)
    else
        println("Unknown Mutation $(mutation)!")
        return g
    end
end

function NeuroEvolution.mutate(mut::Mutate, genome::Genome)
    MutateStructure(mut, MutateWeights(mut, genome))
end

function NeuroEvolution.inherit(mut::Mutate, sel::Sel) where {Sel <: Selector}
    parent = sel[(Int ∘ floor)(rand(mut.RNG) * length(sel)) + 1]
    [Genome(parent)]
end