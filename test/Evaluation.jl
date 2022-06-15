using .NeuroEvolution

@testset verbose=true "Evaluation" begin

    struct DummyEnv <: Environment
        Default::Float64
        Score::Float64
        Time::Float64
    end
    NeuroEvolution.fitness(env::DummyEnv, ::Property"Default") = env.Default
    NeuroEvolution.fitness(env::DummyEnv, ::Property"Score") = env.Score
    NeuroEvolution.fitness(env::DummyEnv, ::Property"Time") = env.Time

    env = DummyEnv(1.0, 2.0, 3.0)

    @testset "Property" begin
        @test fitness(env, Property"Default"()) == 1.0
        @test fitness(env, Property"Score"()) == 2.0
        @test fitness(env, Property"Time"()) == 3.0
    end
    @testset "Average" begin
        @test fitness(env, Average(Property"Score"(), Property"Time"())) ≈ 2.5
        @test fitness(env, Average(Property"Score"(), 2, Property"Time"())) ≈ 7/3
        @test fitness(env, Average(Property"Score"(), 1.5, Property"Time"(), 2)) ≈ 9/3.5
    end
    @testset "Transform" begin
        @test fitness(env, Transform((s, t) -> s * t, Property"Score"(), Property"Time"())) ≈ 6.0
        @test fitness(env, Average(
            Transform(Property"Score"(), Property"Time"()) do s, t
                s * t
            end,
            Transform(Property"Score"(), Property"Time"()) do s, t
                s + t
            end, 2
        )) ≈ 16 / 3
    end
end