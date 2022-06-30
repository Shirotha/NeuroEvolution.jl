export Speciate
#export distance

using Random

using ...NeuroEvolution

"""
Member of a `Species`.
"""
struct Individual{G <: Genotype}
    "`Genotype` of the member."
    Genome::G
    "Fitness of the member."
    Fitness::Float64
end
Individual(i::Individual) = Individual(i.Genome, i.Fitness)
Individual{G}(i::Individual) where {G} = Individual{G}(convert(G, i.Genome), i.Fitness)
Base.convert(::Type{Individual{G}}, i::Individual) where {G} = Individual{G}(i)
Base.promote_rule(::Type{Individual{Ga}}, ::Type{Individual{Gb}}) where {Ga, Gb} = Individual{promote_type(Ga, Gb)}

"""
A single species.
"""
mutable struct Species{G <: Genotype}
    Members::Vector{Individual{G}}
    Species(rep::G, fitness::Float64) where {G <: Genotype} = new{G}(Vector{Individual{G}}([Individual(rep, fitness)]))
end
Base.iterate(s::Species) = iterate(s.Members)
Base.iterate(s::Species, state) = iterate(s.Members, state)
Base.eltype(::Type{Species{G}}) where {G <: Genotype} = Individual{G}
Base.length(s::Species) = length(s.Members)
Base.getindex(s::Species, i::Int) = s.Members[i]

function Base.push!(s::Species{G}, g::G, fitness::Float64) where {G <: Genotype}
    push!(s.Members, Individual(g, fitness))
    s
end

representative(s::Species) = first(s.Members)
# TODO: this only works for positive fitness values, is there a better way?
projected_children(s::Species, average_fitness::Float64) = (Int ∘ ceil)(length(s) * sum(i -> i.Fitness, s) / average_fitness)

"""
`Selector` implementation representing a speciation approch.
This will sort all `Genotype` into species and use the best per species as parents.
"""
mutable struct Speciate{G <: Genotype} <: Selector
    "List of all species"
    Species::Vector{Species{G}}
    "Only the top `CutoffPercent` members of each species will be used as parents"
    CutoffPercent::Float64
    "Individuals father away then this will be speciated."
    SpeciationDistance::Float64

    function Speciate{G}(distance::Float64 = 1.0, cutoff::Float64 = 0.5) where {G <: Genotype}
        @assert 0 < cutoff < 1 "cutoff out of range"

        new{G}(Vector{Species{G}}(), cutoff, distance)
    end
end

Base.iterate(s::Speciate) = iterate(s.Species)
Base.iterate(s::Speciate, state) = iterate(s.Species, state)
Base.eltype(::Type{Speciate{G}}) where {G <: Genotype} = Species{G}
Base.length(s::Speciate) = length(s.Species)
Base.getindex(s::Speciate, i::Int) = s.Species[i]

function push_nocheck!(s::Speciate{G}, g::G, fitness::Float64) where {G <: Genotype}
    species = Species(g, fitness)
    push!(s.Species, species)
    s
end

function Base.push!(s::Speciate{G}, g::G, fitness::Float64) where {G <: Genotype}
    for species in s
        distance(species, g) >= s.SpeciationDistance && continue
        push!(species, g, fitness)
        return s
    end
    push_nocheck!(s, g, fitness)
    s
end

function Base.empty!(s::Speciate)
    for species in s
        empty!(species.Members)
    end
    empty!(s.Species)
end

function average_fitness(s::Speciate)
    n = 0
    f = 0
    for species in s
        for i in species
            n += 1
            f += i.Fitness
        end
    end
    n == 0 && error("empty population")
    f / n
end

struct SpeciesSelector{G <: Genotype} <: Selector
    Selection::Vector{G}

    function SpeciesSelector(species::Species{G}, cutoff::Float64) where {G <: Genotype}
        n = (Int ∘ ceil)(length(species) * cutoff)
        sorted = sort(collect(species), by = i -> -i.Fitness)
        new{G}(map(i -> i.Genome, sorted[1:n]))
    end
end

Base.iterate(s::SpeciesSelector) = iterate(s.Selection)
Base.iterate(s::SpeciesSelector, state) = iterate(s.Selection, state)
Base.eltype(::SpeciesSelector{G}) where {G <: Genotype} = G
Base.length(s::SpeciesSelector) = length(s.Selection)
Base.getindex(s::SpeciesSelector, i::Int) = s.Selection[i] 

function distance(a::G, b::G) where {G <: Genotype}
    error("distance(::$G, ::$G) not implemented")
end
distance(s::Species{G}, g::G) where {G <: Genotype} = distance(representative(s).Genome, g)
