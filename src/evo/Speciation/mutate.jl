export MutateSpecies

using ...NeuroEvolution

using Random
using DataStructures

struct MutateSpecies{N, M <: Mutator} <: Mutator{N, N}
    Mutator::M

    function MutateSpecies(population_size::Int, mutator::M) where {M <: Mutator}
        @assert population_size > 0 "population size has to be positive"
        @assert nchildren(mutator) == 1 "only mutators with single children are supported"
        new{population_size, M}(mutator)
    end
end

NeuroEvolution.mutate(mut::MutateSpecies, g::G) where {G <: Genotype} = mutate(mut.Mutator, g)
function NeuroEvolution.inherit(mut::MutateSpecies, sel::Speciate{G}) where {G <: Genotype}
    N = nchildren(mut)
    P = nparents(mut.Mutator)
    avg = average_fitness(sel)
    avg < 0.0 && error("only positive fitness supported: $avg")
    avg < 1.0 && (avg = 1.0)
    projections = map(s -> projected_children(s, avg), sel)
    i = 1
    while sum(projections) < N
        projections[i] += 1
        i += 1
        i > length(projections) && (i = 1)
    end
    while sum(projections) > N
        projections[i] > 0 && (projections[i] -= 1)
        i += 1
        i > length(projections) && (i = 1)
    end
    result = Queue{G}()
    for (n, species) in zip(projections, sel)
        parents = SpeciesSelector(species, sel.CutoffPercent)
        for _ in 1:n
            enqueue!(result, first(inherit(mut.Mutator, parents)))
        end
    end
    collect(result)
end