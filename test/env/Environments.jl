@everywhere using .NeuroEvolution, .NeuroEvolution.Environments

@testset verbose=true "Environments" begin
    @testset "Xor" begin
        @test inputform(Xor) == [0, 0]
        @test outputform(Xor) == [0, 0]
        @test properties(Xor) == [:Default, :Score]
        
        env = Xor()

        input = process!(env)
        @test all(x -> x ≈ 0 || x ≈ 1, input)
        @test xor(convert(Vector{Bool}, input)...) == env.Target

        process!(env, [~env.Target, env.Target])
        @test fitness(env, Property"Default"()) ≈ 0

        process!(env, [1, 1])
        @test fitness(env, Property"Score"()) ≈ -1
    end
    @testset "Poles" begin
        @test inputform(Poles{2}) == [0, 0, 0, 0]
        @test outputform(Poles{2}) == [0]
        @test properties(Poles{2}) == [:Default, :Time, :Score]

        env = Poles{1}()

        input = process!(env)
        steps = 0
        while !isnothing(input) && steps < 1000
            input = process!(env, [1000input[1]])
            steps += 1
        end
        @test !isnothing(input)
        @test fitness(env, Property"Score"()) < 0
        @test fitness(env, Property"Time"()) > 0
    end
end