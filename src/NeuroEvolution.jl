module NeuroEvolution

    using Distributed

    include("doc-templates.jl")

    include("core.jl")

    # NEAT
    include("evo/Speciation.jl")

    # SUNA
    include("nn/UnifiedModel.jl")
    include("evo/NoveltyMap.jl")
    include("suna.jl")

    # Environments
    include("env/Environments.jl")
end