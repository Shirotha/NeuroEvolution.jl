@everywhere using .NeuroEvolution # Unit
@everywhere using .NeuroEvolution.Speciation

using Printf, ProgressBars

@testset verbose=true "Speciation" begin
    Primes = [2, 3, 5, 7, 11]
    function Speciation.distance(a::DummyGenome, b::DummyGenome)
        sum((mod.(a.Fitness, Primes) .== 0) .|| (mod.(b.Fitness, Primes) .== 0))
    end

    unit = Unit(
            DummyNetwork,
            10,
            DummyGenome(5),
            CopyEnv(),
            NoTrainer(),
            Property"Default"(),
            Speciate{DummyGenome}(),
            MutateSpecies(10, DummyMutate(2));
            copy_parents = false,
            initial_mutations = 1,
            mutations_per_generation = 1,
            steps_per_generation = 1
        )

        iter = ProgressBar(1:100)
        fitness = Vector{Int}(undef, length(iter))
        for i in iter
            fittest, value = process!(unit)
            fitness[i] = value
            set_description(iter, @sprintf "Fitness: %i" value)
        end

        @test sum(fitness[1:end÷2]) < sum(fitness[end÷2+1:end])
end