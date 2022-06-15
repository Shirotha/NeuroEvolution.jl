using LinearAlgebra

using ...NeuroEvolution

export Spectrum
export spectrum, distance, normalize, spectrumtype

"""
Represents the spectrum of a `Genotype` using values of type `T`.

# Interface
`spectrum(genome::G)::Spectrum{spectrumtype(G)}`

should return the `Spectrum` for a `G <: Genotype`.

- `genome`: the genome to get the spectrum from.

`spectrumtype(::Type{G})::Type`

should return the value type that a `Spectrum` should have for a `G <: Genotype`
"""
struct Spectrum{T}
    "Names of the values."
    Names::Vector{Symbol}
    "Values of the spectrum."
    Values::Vector{T}
end
function Spectrum(names::AbstractVector{Symbol}, values::AbstractVector)
    Spectrum{eltype(values)}(names, values)
end
function Spectrum(pairs::Pair{Symbol, Any}...)
    Spectrum([map(first, pairs)...], [map(last, pairs)...])
end
function Spectrum(pairs::Pair{Symbol, T}...) where {T}
    Spectrum{T}([map(first, pairs)...], [map(last, pairs)...])
end
function Spectrum(spec::Spectrum{T}) where {T}
    Spectrum{T}(spec.Names, spec.Values)
end
function Spectrum{T}(spec::Spectrum{S}) where {T, S}
    Spectrum{T}(spec.Names, convert(Vector{T}, spec.Values))
end
Base.convert(::Type{Spectrum{T}}, obj::Spectrum) where {T} = Spectrum{T}(obj)
Base.promote_rule(::Type{Spectrum{A}}, ::Type{Spectrum{B}}) where {A, B} = Spectrum{promote_type(A, B)}

function Base.show(io::IO, spec::Spectrum{T}) where {T} 
    print(io, "Spectrum{$T}(", )
    for i in keys(spec.Names)
        print(io, spec.Names[i], " => ", spec.Values[i])
        lastindex(spec.Names) != i && print(io, ", ")
    end
    print(io, ")")
end
function Base.show(io::IO, ::MIME"text/plain", spec::Spectrum{T}) where {T}
    println(io, "Spectrum{$T}:")
    for i in keys(spec.Names)
        println(io, " ", spec.Names[i], " => ", spec.Values[i])
    end
end

"""
Calculates the `Spectrum` of a `Genotype`.

see `Spectrum`
"""
function spectrum(genome::G) where {G <: Genotype}
    throw(MissingException("spectrum(::$G) not implemented"))
end
"""
Returns the type all values of a `Spectrum` for a `Genotype` should have.

see `Spectrum`
"""
function spectrumtype(::Type{G}) where {G <: Genotype}
    throw(MissingException("spectrumtype(::Type{$G}) not implemented"))
end

"""
Calculate the distance between two `Spectrum`s.
"""
distance(a::Spectrum, b::Spectrum; Fnorm = norm) = distance(promote(a, b)...; Fnorm)
function distance(a::Spectrum{T}, b::Spectrum{T}; Fnorm::Function = norm) where {T}
    Fnorm(a.Values - b.Values)
end

"""
Normalize a `Spectrum` so that `sum(spectrum.Values) == 1`.
This does nothing for a `Spectrum` with `Integer` value type.
"""
function normalize(spectrum::Spectrum)
    s = sum(spectrum.Values)
    Spectrum(spectrum.Names, s ≈ 0 ? spectrum.Values : spectrum.Values ./ s)
end
normalize(spectrum::Spectrum{<:Integer}) = spectrum