@everywhere using Random
@everywhere using .NeuroEvolution.UnifiedModel

@testset verbose=true "Unified Model" begin
    @testset "Weights" begin
        genome = Genome(
            [
                NeuronGene(10, INPUT_IDENTITY, io=2),
                NeuronGene(11, INPUT_IDENTITY, io=1),
                NeuronGene(12, IDENTITY),
                NeuronGene(13, IDENTITY),
                NeuronGene(14, OUTPUT_IDENTITY, io=1),
                NeuronGene(42, RANDOM)
            ],[
                ConnectionGene(10, 12, 0.5),
                ConnectionGene(11, 12, 0.5),
                ConnectionGene(12, 14, 1.0),
                ConnectionGene(10, 13, 1.0),
                ConnectionGene(11, 13, -1.0),
                ConnectionGene(13, 14, 1.0)
            ]
        )
        
        network = express(genome)
        @test network.NumberOfInputs == 2
        @test network.NumberOfOutputs == 1
        @test network.NumberOfControls == 0
        @test length(network.Neurons) == 5
        @test length(network.Connections) == 6

        @test network(0.2, 0.4) ≈ 0.5
        @test network(0.1, 0.4) ≈ 0.55
        @test network(0.1, 0.3) ≈ 0.4
        @test network(0.2, 0.3) ≈ 0.35
        @test network(0.2, 0.5) ≈ 0.65
    end
    @testset "Recurrency" begin
        genome = Genome(
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

        network = express(genome)

        function test_derivative(f, ∂f; Δx = 1e-2, x0 = 0, x1 = π, rtol = 1e-2)
            xs = collect(x0:Δx:x1)
            ys = f.(xs)
            ∂ys = [network(y, 1/Δx) for y in ys]
            ∂fs = ∂f.(xs)
            isapprox(∂ys[2:end], ∂fs[2:end]; rtol)
        end
        
        @test test_derivative(x -> 3x^2, x -> 6x)
        @test test_derivative(sin, cos)
        @test test_derivative(exp, exp)
    end
    @testset "Control" begin
        genome = Genome(
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

        network = express(genome)
        @test network(0, 0) ≈ 0
        @test network(0, 1) ≈ 1
        @test network(1, 0) ≈ 1
        @test network(1, 1) ≈ 0
    end
    @testset "Mutation" begin
        struct DummySampler end
        mutable struct DummyRNG <: AbstractRNG
            Sequence::Vector{Float64}
            Index::Int
        end
        DummyRNG(seq::Vector{T}) where {T <: Real} = DummyRNG(convert(Vector{Float64}, seq), 1)
        Random.Sampler(::Type{DummyRNG}, ::Type{Float64}, ::Random.Repetition) where {S} = DummySampler()
        function Base.rand(rng::DummyRNG, ::DummySampler)
            val = rng.Sequence[rng.Index]
            rng.Index += 1
            rng.Index > length(rng.Sequence) && (rng.Index = 1)
            return val
        end

        Base.isapprox(a::NeuronGene, b::NeuronGene) = a.ID == b.ID && a.Type == b.Type && a.Speed ≈ b.Speed && a.Interface == b.Interface
        Base.isapprox(a::ConnectionGene, b::ConnectionGene) = a.From == b.From && a.To == b.To && a.Weight ≈ b.Weight && a.Modulator == b.Modulator

        mut = Mutate(;
            rng = DummyRNG([
                1.0, # mutate weights?
                0.505, # mutate structure (remove_neuron: 0.005, remove_connection: 0.1, add_neuron: 0.505, add_connection: 0.6)
                    0.0, # neuron type
                    0.0, # neuron speed
                    0.5, # out connection target
                    0.7, # weight
                    1.0, # modulation?
                    0.0, # in connection source
                    0.4, # weight
                    1.0, # modulation?
                1.0,
                0.1,
                    0.2, # connection index
                1.0,
                0.005,
                    0.2, # neuron index
                1.0,
                0.005,
                    0.9,
                1.0,
                0.6,
                    0.2, # from
                    0.8, # to
                    0.9, # weight
                    0.0, # modulation?
                    0.8, # modulator
            ])
        )
        genome = Genome(1, 1)
        # === Add Neuron ===
        genome = mutate(mut, genome)
        @assert length(genome.Neurons) == 3
        @test genome.Neurons[3].ID == 3
        @assert length(genome.Connections) == 2
        @test genome.Connections[1] ≈ ConnectionGene(3, 2, 0.4, 0)
        @test genome.Connections[2] ≈ ConnectionGene(1, 3, -0.2, 0)
        # === Remove IN Connection ===
        genome = mutate(mut, genome)
        @assert length(genome.Connections) == 1
        @test genome.Connections[1].From == 1
        # === Try Remove Interface Neuron ===
        genome = mutate(mut, genome)
        @assert length(genome.Neurons) == 3
        # === Remove Neuron ===
        genome = mutate(mut, genome)
        @assert length(genome.Neurons) == 2
        @assert isempty(genome.Connections)
        @test all(isinterface, genome.Neurons)
        # === Add Modulated Connection ===
        genome = mutate(mut, genome)
        @assert length(genome.Connections) == 1
        @test genome.Connections[1] ≈ ConnectionGene(1, 2, 1.0, 2)
    end
end