using ...NeuroEvolution

export NeuronGene, ConnectionGene, Genome, NeuronType, NamedFunction
export IDENTITY, SIGMOID, THRESHOLD, RANDOM, CONTROL, INPUT_IDENTITY, INPUT_SIGMOID, OUTPUT_IDENTITY, OUTPUT_SIGMOID
export FUNC_IDENTITY, FUNC_SIGMOID, FUNC_THRESHOLD, FUNC_RANDOM
export TYPE_CONTROL, TYPE_INPUT, TYPE_OUTPUT
export iscontrol, isinput, isoutput, isinterface, isslow, ismodulated

"""
Represents a function object with a name assigned to it.
"""
struct NamedFunction
    "Name of the `Function`."
    Name::Symbol
    "The `Function` itself."
    Function::Function
end
(f::NamedFunction)(x) = f.Function(x)

Base.show(io::IO, f::NamedFunction) = print(io, "FUNC_$(uppercase(string(f.Name)))")

const FUNC_IDENTITY = NamedFunction(:Identity, identity)
const FUNC_SIGMOID = NamedFunction(:Sigmoid, tanh)
const FUNC_THRESHOLD = NamedFunction(:Threshold, x -> x > 0.0 ? 1.0 : -1.0)
const FUNC_RANDOM = NamedFunction(:Random, x -> 2.0rand() - 1.0)

"""
Represents the type of neuron by the kind of activation function used and the usage of the neuron (control neuron, input neuron, output neuron).
"""
struct NeuronType
    "Kind of activation function used for this `Neuron`"
    ActivationFunction::NamedFunction
    "Type of `Neuron`"
    Type::UInt8
end
NeuronType(f::NamedFunction) = NeuronType(f, 0)

Base.show(io::IO, t::NeuronType) = print(io, "NeuronType($(t.ActivationFunction), $(
    t.Type == TYPE_CONTROL ? "TYPE_CONTROL" :
    t.Type == TYPE_INPUT ? "TYPE_INPUT" :
    t.Type == TYPE_OUTPUT ? "TYPE_OUTPUT" : t.Type
))")

const TYPE_CONTROL = 0x01
const TYPE_INPUT = 0x02
const TYPE_OUTPUT = 0x04

const IDENTITY = NeuronType(FUNC_IDENTITY)
const SIGMOID = NeuronType(FUNC_SIGMOID)
const THRESHOLD = NeuronType(FUNC_THRESHOLD)
const RANDOM = NeuronType(FUNC_RANDOM)

const CONTROL = NeuronType(FUNC_THRESHOLD, TYPE_CONTROL)

const INPUT_IDENTITY = NeuronType(FUNC_IDENTITY, TYPE_INPUT)
const INPUT_SIGMOID = NeuronType(FUNC_SIGMOID, TYPE_INPUT)

const OUTPUT_IDENTITY = NeuronType(FUNC_IDENTITY, TYPE_OUTPUT)
const OUTPUT_SIGMOID = NeuronType(FUNC_SIGMOID, TYPE_OUTPUT)

iscontrol(t::NeuronType)::Bool = t.Type & TYPE_CONTROL != 0
isinput(t::NeuronType)::Bool = t.Type & TYPE_INPUT != 0
isoutput(t::NeuronType)::Bool = t.Type & TYPE_OUTPUT != 0
isinterface(t::NeuronType)::Bool = t.Type & (TYPE_INPUT | TYPE_OUTPUT) != 0

"""
Represents a neuron in the unified gene model.
"""
struct NeuronGene
    "ID used in `ConnectionGene`."
    ID::Int
    "Type of `Neuron`."
    Type::NeuronType
    """
    Speed at which the `Neuron` updates itself.
    A speed of `s` means that a neuron needs `1/s` updates to reach stead state after the input was changed.
    """
    Speed::Float64
    "Index of the input/output that this `Neuron` represents, zero when it is not part of the interface."
    Interface::Int

    NeuronGene(id::Int, type::NeuronType, speed::Float64, interface::Int) = speed > 1.0 ? error("neuron can not be faster than 1.0!") : new(id, type, speed, interface)
end

NeuronGene(id::Int, type::NeuronType, speed::Float64 = 1.0; io::Int = 0) = NeuronGene(id, type, speed, io)

iscontrol(n::NeuronGene)::Bool = iscontrol(n.Type)
isinput(n::NeuronGene)::Bool = isinput(n.Type)
isoutput(n::NeuronGene)::Bool = isoutput(n.Type)
isinterface(n::NeuronGene)::Bool = n.Interface != 0
isslow(n::NeuronGene)::Bool = n.Speed < 1.0

"""
Represents a connection in the unified gene model.
"""
struct ConnectionGene
    "`NeuronGene.ID` for source."
    From::Int
    "`NeuronGene.ID` for target."
    To::Int
    "Weight of the connection, ignored when `Modulator` is set"
    Weight::Float64
    "`NeuronGene.ID` of the modulator, or zero."
    Modulator::Int
end

ConnectionGene(from::Int, to::Int, weight::Float64 = 1.0) = ConnectionGene(from, to, weight, 0)

ismodulated(c::ConnectionGene) = c.Modulator != 0

"""
`Genotype` implementation that represents a genome in the unified gene model.
"""
struct Genome <: Genotype
    "List of neuron genes."
    Neurons::Vector{NeuronGene}
    "List of connection genes."
    Connections::Vector{ConnectionGene}
end

function Genome(i::Int, o::Int; normalize_input::Bool = false, normalize_output::Bool = false)
    @assert i >= 0 "number of inputs can't be negative"
    @assert o >= 0 "number of outputs can't be negative"
    neurons = Vector{NeuronGene}(undef, i + o)
    for k in 1:i
        neurons[k] = NeuronGene(k, normalize_input ? INPUT_SIGMOID : INPUT_IDENTITY, io = k)
    end
    for k in 1:o
        neurons[i + k] = NeuronGene(i + k, normalize_output ? OUTPUT_SIGMOID : OUTPUT_IDENTITY, io = k)
    end
    Genome(neurons, Vector{ConnectionGene}())
end
Genome(g::Genome) = Genome(Vector(g.Neurons), Vector(g.Connections))