using Distributed
using Test

#using NeuroEvolution
@everywhere include("../src/NeuroEvolution.jl")

@testset verbose=true "Neuro Evolution" begin
    include("Evaluation.jl")

    include("nn/UnifiedModel.jl")

    include("evo/dummy.jl")
    include("evo/NoveltyMap.jl")
    include("evo/Speciation.jl")

    include("env/Environments.jl")
    
    include("suna.jl")
end