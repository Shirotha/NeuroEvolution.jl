module SUNA

    include("doc-templates.jl")

    using Random

    using ...NeuroEvolution
    using ..UnifiedModel
    using ..NoveltyMap

    export Suna

    NoveltyMap.spectrumtype(::Type{Genome}) = Int

    function NoveltyMap.spectrum(g::Genome)
        # TODO: find a way to not hardcode this (meta programming -> get constants with name FUNC_*)
        identity = 0
        sigmoid = 0
        threshold = 0
        random = 0
        control = 0
        slow = 0
        for n in g.Neurons
            isinterface(n) && continue
            isslow(n) && (slow += 1)
            iscontrol(n) && (control += 1; continue)
            name = n.Type.ActivationFunction.Name
            name == :Identity && (identity += 1)
            name == :Sigmoid && (sigmoid += 1)
            name == :Threshold && (threshold += 1)
            name == :Random && (random += 1)
        end
        Spectrum(
            :Slow => slow,
            :Control => control,
            :Identity => identity,
            :Sigmoid => sigmoid,
            :Threshold => threshold,
            :Random => random
        )
    end
    
    """
    An alias for `Unit` that sets up the SUNA algorithm.

    see `UnifiedModel`, `NoveltyMap`
    """
    Suna{Env <: Environment, Eval <: Evaluation} = 
        Unit{Genome, Network, Env, NoTrainer, Eval, Map, Mutate}
    function Suna(
        environment::Env, 
        evaluation::Eval=Property"Default"();
        initial_mutations::Int = 200,
        mutations_per_generation::Int = 5,
        steps_per_generation::Int = 1000,
        evaluation_count::Int  = 1,
        population_size::Int = 100,
        novelty_map_size::Int = 20,
        # mutate forwards
        rng::AbstractRNG = RandomDevice(),
        max_weight::Float64 = 1e9,
        weight_mutation_chance::Float64 = 0.5,
        weight_change_percent::Float64 = 1.0,
        add_neuron_chance::Float64 = 0.01,
        remove_neuron_chance::Float64 = 0.01,
        add_connection_chance::Float64 = 0.49,
        remove_connection_chance::Float64 = 0.49,
        modulation_chance::Float64 = 0.1,
        neuron_speeds = Dict(1/1 => 1, 1/7 => 1, 1/49 => 1),
        neuron_types = Dict(IDENTITY => 1, SIGMOID => 1, THRESHOLD => 1, RANDOM => 1, CONTROL => 1)
    ) where {Env <: Environment, Eval <: Evaluation}
        default_genome = Genome(length(inputform(environment)), length(outputform(environment)))
        novelty_map = Map(Genome, novelty_map_size)
        mutate = Mutate(;
            rng, 
            max_weight, 
            weight_mutation_chance, 
            weight_change_percent, 
            add_neuron_chance, 
            remove_neuron_chance, 
            add_connection_chance, 
            remove_connection_chance, 
            modulation_chance, 
            neuron_speeds, 
            neuron_types
        )
        Unit(Network,
            population_size, 
            default_genome, 
            environment, 
            NoTrainer(),
            evaluation,
            novelty_map,
            mutate;
            copy_parents = true,
            initial_mutations,
            mutations_per_generation,
            steps_per_generation,
            evaluation_count
        )
    end

    using ..Speciation

    export SunaSpeciate

    function safe_iterate(iter)
        result = iterate(iter)
        if isnothing(result)
            return nothing, nothing
        else
            return result
        end
    end
    function safe_iterate(iter, state)
        result = iterate(iter, state)
        if isnothing(result)
            return nothing, state
        else
            return result
        end
    end

    function zip_unaligned(callback, aryA, aryB; by=identity)
        iterA = sort(aryA; by)
        iterB = sort(aryB; by)
        a, stateA = safe_iterate(iterA)
        b, stateB = safe_iterate(iterB)
        while !(isnothing(a) || isnothing(b))
            while by(a) < by(b)
                a, stateA = safe_iterate(iterA, stateA)
                isnothing(a) && return
            end
            while by(b) < by(a)
                b, stateB = safe_iterate(iterB, stateB)
                isnothing(b) && return
            end
            callback(a, b)
            a, stateA = safe_iterate(iterA, stateA)
            b, stateB = safe_iterate(iterB, stateB)
        end
        nothing
    end

    const DISTANCE_ACTIVATION_FUNCTION = 0.5
    const DISTANCE_SPEED = 0.3
    const DISTANCE_WEIGHT = 0.5
    const DISTANCE_MODULATOR = 0.7
    const DISTANCE_DISJOINT = 1.0
    function Speciation.distance(a::Genome, b::Genome)
        length(a.Connections) + length(b.Connections) == 0 && return 0.0
        Dact = 0.0
        Dspeed = 0.0
        Nneuron = 0
        zip_unaligned(a.Neurons, b.Neurons, by = n -> n.ID) do nA, nB
            Dact += nA.Type.ActivationFunction == nB.Type.ActivationFunction
            Dspeed += (abs ∘ log)(nA.Speed / nB.Speed)
            Nneuron += 1
        end
        if Nneuron > 0
            Dact /= Nneuron
            Dspeed /= Nneuron
        end

        Dweight = 0.0
        Dmod = 0.0
        Nconn = 0
        offset = div(typemax(Int), 2)
        zip_unaligned(a.Connections, b.Connections, by = c -> c.From + offset * c.To) do cA, cB
            Dweight += abs(cA.Weight - cB.Weight)
            Dmod += cA.Modulator == cB.Modulator
            Nconn += 1
        end
        if Nconn > 0
            Dweight /= Nconn
            Dmod /= Nconn
        end
        Ddisjoint = (length(a.Connections) + length(b.Connections) - 2Nconn) / max(length(a.Connections), length(b.Connections))

        return Dact * DISTANCE_ACTIVATION_FUNCTION +
               Dspeed * DISTANCE_SPEED +
               Dweight * DISTANCE_WEIGHT +
               Dmod * DISTANCE_MODULATOR +
               Ddisjoint * DISTANCE_DISJOINT
        end

    """
    An alias for `Unit` that sets up the SUNA algorithm but replaces the Novelty Map by a speciation approch.

    see `UnifiedModel`, `Speciate`
    """
    SunaSpeciate{Env <: Environment, Eval <: Evaluation} = 
        Unit{Genome, Network, Env, NoTrainer, Eval, Speciate, MutateSpecies}
    function SunaSpeciate(
        environment::Env, 
        evaluation::Eval=Property"Default"();
        initial_mutations::Int = 200,
        mutations_per_generation::Int = 5,
        steps_per_generation::Int = 1000,
        evaluation_count::Int  = 1,
        population_size::Int = 100,
        speciation_distance::Float64 = 1.0,
        cutoff_percent::Float64 = 0.5,
        # mutate forwards
        rng::AbstractRNG = RandomDevice(),
        max_weight::Float64 = 1e9,
        weight_mutation_chance::Float64 = 0.5,
        weight_change_percent::Float64 = 1.0,
        add_neuron_chance::Float64 = 0.01,
        remove_neuron_chance::Float64 = 0.01,
        add_connection_chance::Float64 = 0.49,
        remove_connection_chance::Float64 = 0.49,
        modulation_chance::Float64 = 0.1,
        neuron_speeds = Dict(1/1 => 1, 1/7 => 1, 1/49 => 1),
        neuron_types = Dict(IDENTITY => 1, SIGMOID => 1, THRESHOLD => 1, RANDOM => 1, CONTROL => 1)
    ) where {Env <: Environment, Eval <: Evaluation}
        default_genome = Genome(length(inputform(environment)), length(outputform(environment)))
        speciate = Speciate{Genome}(speciation_distance, cutoff_percent)
        mutate = MutateSpecies(population_size, 
            Mutate(;
                rng, 
                max_weight, 
                weight_mutation_chance, 
                weight_change_percent, 
                add_neuron_chance, 
                remove_neuron_chance, 
                add_connection_chance, 
                remove_connection_chance, 
                modulation_chance, 
                neuron_speeds, 
                neuron_types
        ))
        Unit(Network,
            population_size, 
            default_genome, 
            environment, 
            NoTrainer(),
            evaluation,
            speciate,
            mutate;
            copy_parents = false,
            initial_mutations,
            mutations_per_generation,
            steps_per_generation,
            evaluation_count
        )
    end
end