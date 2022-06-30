@everywhere using .NeuroEvolution.UnifiedModel, .NeuroEvolution.NoveltyMap
@everywhere using .NeuroEvolution.SUNA # Spectrum example
@everywhere using .NeuroEvolution # Unit

using Printf, ProgressBars

@testset verbose=true "Novelty Map" begin
    @testset "Spectrum" begin
        @test spectrumtype(Genome) == Int

        genomeA = Genome(
                [
                    NeuronGene(1, INPUT_IDENTITY, io=1),
                    NeuronGene(2, INPUT_IDENTITY, io=2),
                    NeuronGene(3, OUTPUT_IDENTITY, io=1),
                    NeuronGene(4, IDENTITY),
                    NeuronGene(5, CONTROL),
                    NeuronGene(6, CONTROL)
                ],[
                    ConnectionGene(1, 4, 1.0),
                    ConnectionGene(2, 4, 1.0),
                    ConnectionGene(1, 5, 1.0),
                    ConnectionGene(2, 5, -1.0),
                    ConnectionGene(1, 6, -1.0),
                    ConnectionGene(2, 6, 1.0),
                    ConnectionGene(5, 4, 1.0),
                    ConnectionGene(6, 4, 1.0),
                    ConnectionGene(4, 3, 1.0)
                ]
            )
        
        specA = spectrum(genomeA)
        @test specA.Names ==  [:Slow, :Control, :Identity, :Sigmoid, :Threshold, :Random]
        @test specA.Values == [ 0,     2,        1,         0,        0,          0     ]
        @test NoveltyMap.distance(specA, specA) == 0
        normA = normalize(specA)
        @test sum(normA.Values) ≈ 3
        normA = normalize(convert(Spectrum{Float64}, specA))
        @test sum(normA.Values) ≈ 1

        genomeB = Genome(
                [
                    NeuronGene(1, INPUT_IDENTITY, io=1),
                    NeuronGene(2, INPUT_IDENTITY, io=2),
                    NeuronGene(3, IDENTITY),
                    NeuronGene(4, IDENTITY),
                    NeuronGene(5, OUTPUT_IDENTITY, io=1)
                ],[
                    ConnectionGene(1, 3, 1.0),
                    ConnectionGene(1, 4, 1.0),
                    ConnectionGene(4, 3, -1.0),
                    ConnectionGene(3, 5, 1.0, 2)
                ]
            )

        specB = spectrum(genomeB)
        @test specB.Values == [0, 0, 2, 0, 0, 0]
        @test NoveltyMap.distance(specA, specB) ≈ sqrt(5)
    end
    @testset "Map" begin
        
        NoveltyMap.spectrumtype(::Type{DummyGenome}) = Bool
        Primes = [2, 3, 5, 7, 11]
        function NoveltyMap.spectrum(g::DummyGenome)
            Spectrum(Symbol.(Primes), mod.(g.Fitness, Primes) .== 0)
        end
        
        unit = Unit(
            DummyNetwork,
            10,
            DummyGenome(5),
            CopyEnv(),
            NoTrainer(),
            Property"Default"(),
            Map(DummyGenome, 4),
            DummyMutate(2);
            copy_parents = true,
            initial_mutations = 5,
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
end