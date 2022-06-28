using ...NeuroEvolution

export Map

"""
A single entry in a `Map`.
"""
struct Cell{S <: Spectrum, G <: Genotype}
    "Cached `Spectrum` for `Genome`."
    Spectrum::S
    "Fitness for `Genome`."
    Fitness::Float64
    "Stored `Genotype`."
    Genome::G
end
Cell(obj::Cell) = Cell(obj.Spectrum, obj.Fitness, obj.Genome)
Cell{S, G}(obj::Cell) where {S, G} = Cell{S, G}(convert(S, obj.Spectrum), obj.Fitness, convert(G, obj.Genome))
Base.convert(::Type{Cell{S, G}}, obj::Cell) where {S <: Spectrum, G <: Genotype} = Cell{S, G}(obj)
Base.promote_rule(::Type{Cell{Sa, Ga}}, ::Type{Cell{Sb, Gb}}) where {Sa, Ga, Sb, Gb} = 
    Cell{promote_type(Sa, Sb), promote_type(Ga, Gb)}

Base.show(io::IO, cell::Cell{S, G}) where {S, G} = print(io, "Cell{$C, $G}(", cell.Spectrum, ", ", cell.Fitness, ", ", cell.Genome)

"""
`Selector` implementation representing a Novelty Map designed for use with the SUNA algorithm.
The used `Genotype` must support the calculation of its spectrum.

see `Spectrum`
"""
mutable struct Map{C <: Cell} <: Selector
    "Collection of all cells."
    Cells::Vector{Union{Missing, C}}
    "Current number of added genomes."
    Population::Int
    "Minimum spectrum distance between any combination in `Population`."
    MinDistance::Float64
    "The index to the genome with `MinDistance`."
    WorstIndividual::Int
end
function Map(G::Type{<:Genotype}, size::Int)
    cellT = Cell{Spectrum{spectrumtype(G)}, G}
    Map{cellT}(Vector{Union{Missing, cellT}}(missing, size), 0, 0.0, 0)
end

function Base.show(io::IO, ::MIME"text/plain", map::Map{C}) where {C}
    println(io, "Map with ", map.Population, " entries")
    println(io, " WorstIndividual at ", map.WorstIndividual, " (MinDistance = ", eval.MinDistance, ")")
    println(io, " Cells ", map.Cells)
end

Base.iterate(map::M) where {M <: Map} = iterate(map, 1)
Base.iterate(map::M, state::Int) where {M <: Map} = 
    state > map.Population ? 
        nothing : 
        (map.Cells[state].Genome, state + 1)

Base.eltype(::Type{<:Map{Cell{S, G}}}) where {S <: Spectrum, G <: Genotype} = G
Base.length(map::M) where {M <: Map} = map.Population
Base.getindex(map::M, i::Int) where {M <: Map} = map.Cells[i].Genome

function Base.empty!(map::M) where {M <: Map}
    map.Cells .= missing
    map.Population = 0
    map.MinDistance = 0.0
    map.WorstIndividual = 0
end

Base.push!(map::M, genome::G, fitness::T) where {M <: Map, G <: Genotype, T} =
    push!(map, genome, convert(Float64, fitness))
function Base.push!(map::M, genome::G, fitness::Float64) where {M <: Map, G <: Genotype}
    (index, spec) = findtarget(map, genome)
    entry = Cell(spec, fitness, genome)
    if isbetter(entry, map.Cells[index])
        map.Cells[index] = entry
    elseif !ismissing(map.Cells[index])
        entry =  map.Cells[index]
        map.Cells[index] = Cell(spec, entry.Fitness, entry.Genome)
    end
    map.Population == length(map.Cells) && updateMinDistance!(map)
    map
end

function findtarget(map::M, genome::G) where {M <: Map, G <: Genotype}
    spec = spectrum(genome)
    if map.Population < length(map.Cells)
        index = (map.Population += 1)
    else
        (index, min) = MinDistance(map, spec)
        map.MinDistance < min && (index = map.WorstIndividual)
    end
    (index, spec)
end

function updateMinDistance!(map::M) where {M <: Map}
    min = Inf; index = 0
    for i in 2:map.Population
        _, dist = MinDistance(map, map.Cells[i].Spectrum; exclude=i)
        dist < min && (min = dist; index = i)
    end
    map.MinDistance = min
    map.WorstIndividual = index
    nothing
end

function MinDistance(map::Map{Cell{S, G}}, spectrum::S; exclude::Int=0) where {S <: Spectrum, G <: Genotype}
    map.Population == 0 && return (0, 0.0)
    min = Inf; index = 0
    for i in 1:map.Population
        i == exclude && continue
        dist = distance(map.Cells[i].Spectrum, spectrum)
        dist < min && (min = dist; index = i)
    end
    (index, min)
end

function isbetter(a::T, b::Union{T, Missing}) where {T <: Cell}
    ismissing(b) ||
        a.Fitness > b.Fitness ||
        a.Fitness == b.Fitness && norm(a.Spectrum.Values) < norm(b.Spectrum.Values)
end