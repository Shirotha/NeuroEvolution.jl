@everywhere using .NeuroEvolution, .NeuroEvolution.Environments, .NeuroEvolution.SUNA

using ProgressBars, Printf

@testset "SUNA" begin
    @testset "Xor" begin
        env = Xor()
        suna = Suna(env;
            population_size = 100,
            steps_per_generation = 100
        )
        iter = ProgressBar(1:100)
        fitness = Vector{Float64}(undef, length(iter))
        for i in iter
            # FIXME: fitness is NaN sometimes (propagated from network output)
            fittest, value = process!(suna)
            fitness[i] = value
            # TODO: extract average fitness
            set_description(iter, @sprintf "Fitness: %.2e" value)
        end
        @test sum(fitness[1:end÷2]) < sum(fitness[end÷2+1:end])
    end
    @testset "Poles" begin
        env = Poles{1}(; max_initial_angle = 0.2)
        suna = Suna(env;
            population_size = 50,
            steps_per_generation = 500,
            evaluation_count = 3
        )
        iter = ProgressBar(1:50)
        fitness = Vector{Float64}(undef, length(iter))
        fittest = nothing
        for i in iter
            fittest, value = process!(suna)
            fitness[i] = value
            set_description(iter, @sprintf "Fitness: %.2e" value)
        end
        @test sum(fitness[1:end÷2]) < sum(fitness[end÷2+1:end])
    end
end