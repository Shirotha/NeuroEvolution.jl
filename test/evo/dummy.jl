@everywhere begin
    using .NeuroEvolution

    struct DummyGenome <: Genotype
        Fitness::Int
    end
    DummyGenome() = DummyGenome((Int ∘ floor)(10rand()))
    DummyGenome(g::DummyGenome) = DummyGenome(g.Fitness)
    Base.show(io::IO, g::DummyGenome) = print(io, "g($(g.Fitness))")

    mutable struct DummyNetwork <: Phenotype
        Fitness::Int
    end
    DummyNetwork(genome::DummyGenome) = DummyNetwork(genome.Fitness)
    
    NeuroEvolution.express(genome::DummyGenome) = DummyNetwork(genome)
    NeuroEvolution.process!(net::DummyNetwork, input::Int) = net.Fitness
    NeuroEvolution.reset!(net::DummyNetwork) = nothing

    mutable struct CopyEnv <: Environment
        Fitness::Int
        CopyEnv() = new(0)
    end
    
    function NeuroEvolution.process!(env::CopyEnv, output::Union{Nothing, Int} = nothing)
        isnothing(output) && return 0
        env.Fitness = output
    end
    NeuroEvolution.fitness(env::CopyEnv, ::Property"Default") = env.Fitness
    
    struct DummyMutate <: Mutator{1, 1}
        MutationSpeed::Int
    end
    
    NeuroEvolution.mutate(mut::DummyMutate, g::DummyGenome, count::Int = 1) = 
        DummyGenome(g.Fitness + (Int ∘ floor)((2rand() - 1) * mut.MutationSpeed))
    function NeuroEvolution.inherit(mut::DummyMutate, sel::Selector)
        parent = sel[(Int ∘ floor)(rand() * length(sel)) + 1]
        [DummyGenome(parent)]
    end
end