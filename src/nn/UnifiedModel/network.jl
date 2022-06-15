using Distributed, DistributedArrays
using DataStructures

using ...NeuroEvolution

export Network
export express, train!, process!

const MAX_STATE_VALUE = 1e9

"""
Represents a neuron in the `Network` class.

see `NeuronGene`
"""
struct Neuron
    ActivationFunction::NamedFunction
    Speed::Float64

    Neuron(func::NamedFunction, speed::Float64) = speed > 1.0 ? error("Neuron can not be faster than 1.0!") : new(func, speed)
end
Neuron(n::NeuronGene) = Neuron(n.Type.ActivationFunction, n.Speed)

Base.show(io::IO, n::Neuron) = print(io, "Neuron(", n.ActivationFunction, ", ", n.Speed, ")")

"""
Represents a connection in the `Network` class.
IDs are indices into a `Neuron` array.

see `ConnectionGene`
"""
struct Connection
    From::Int
    To::Int
    Weight::Float64
    Modulator::Int
end

Base.show(io::IO, c::Connection) = print(io, "Connection(", c.From, ", ", c.To, ", ", c.Weight, ", ", c.Modulator, ")")

"""
`Phenotype` implementation that uses the unified neuron model, designed to work with SUNA.
"""
mutable struct Network <: Phenotype
    "Number of input neurons."
    NumberOfInputs::Int
    "Number of output neurons."
    NumberOfOutputs::Int
    "Number of control neurons."
    NumberOfControls::Int
    "Sorted list of neurons, should be processed in that exact order."
    Neurons::Vector{Neuron}
    "Sorted list of connections, should be processed in that exact order."
    Connections::Vector{Connection}
    "Internal state of each neuron in the network."
    State::Vector{Float64}
end

function Network(g::Genome)
    neurons = Vector{Neuron}(undef, length(g.Neurons))
    mapping = Dict{Int, Int}()
    ninputs = 0
    noutputs = 0
    ncontrols = 0

    openlist = Queue{NeuronGene}()
    closedlist = Queue{NeuronGene}()

    for n in g.Neurons
        iscontrol(n) || continue
        ncontrols += 1

        isprimer = true
        for c in g.Connections
            c.To == n.ID || continue
            i = findfirst(m -> m.ID == c.From, g.Neurons)
            if iscontrol(g.Neurons[i])
                isprimer = false
                break
            end
        end
        isprimer && enqueue!(openlist, n)
    end

    while !isempty(openlist)
        n = dequeue!(openlist)
        enqueue!(closedlist, n)
        for c in g.Connections
            c.From == n.ID || continue
            any(m -> m.ID == c.To, closedlist) && continue
            any(m -> m.ID == c.To, openlist) && continue

            i = findfirst(m -> m.ID == c.To, g.Neurons)
            iscontrol(g.Neurons[i]) || continue

            enqueue!(openlist, g.Neurons[i])
        end
    end

    for n in g.Neurons
        if isinput(n)
            enqueue!(openlist, n)
            neurons[n.Interface] = Neuron(n)
            mapping[n.ID] = n.Interface
            n.Interface > ninputs && (ninputs = n.Interface)
        end
        if isoutput(n)
            n.Interface > noutputs && (noutputs = n.Interface)
        end
    end

    j = ninputs
    while !isempty(closedlist)
        n = dequeue!(closedlist)
        #haskey(mapping, n.ID) && throw("trying to add same neuron twice! ($n)")
        neurons[j += 1] = Neuron(n)
        mapping[n.ID] = j
    end
    while j < ninputs + ncontrols
        deleteat!(neurons, j + 1)
        ncontrols -= 1
    end
    
    while !isempty(openlist)
        n = dequeue!(openlist)
        isinterface(n) || enqueue!(closedlist, n)
        for c in g.Connections
            c.From == n.ID || continue
            any(m -> m.ID == c.To, closedlist) && continue
            any(m -> m.ID == c.To, openlist) && continue

            i = findfirst(m -> m.ID == c.To, g.Neurons)
            iscontrol(g.Neurons[i]) && continue
            isinterface(g.Neurons[i]) && continue

            enqueue!(openlist, g.Neurons[i])
        end
    end
    
    while !isempty(closedlist)
        n = dequeue!(closedlist)
        #haskey(mapping, n.ID) && throw("trying to add same neuron twice! ($n)")
        neurons[j += 1] = Neuron(n)
        mapping[n.ID] = j
    end
    while j < length(neurons) - noutputs
        deleteat!(neurons, j + 1)
    end

    for n in g.Neurons
        isoutput(n) || continue

        i = length(neurons) - noutputs + n.Interface
        neurons[i] = Neuron(n)
        mapping[n.ID] = i
    end
    
    connections = Queue{Connection}()
    for c in g.Connections
        haskey(mapping, c.From) || continue
        haskey(mapping, c.To) || continue
        mod = c.Modulator != 0 && haskey(mapping, c.Modulator) ? mapping[c.Modulator] : 0
        enqueue!(connections, Connection(mapping[c.From], mapping[c.To], c.Weight, mod))
    end
    connections = sort(collect(connections), by = c -> c.To)

    Network(ninputs, noutputs, ncontrols, neurons, connections, zeros(Float64, length(neurons)))
end
NeuroEvolution.express(genome::Genome) = Network(genome)

NeuroEvolution.process!(net::Network, input) = process!(net, convert(Vector{Float64}, input))
function NeuroEvolution.process!(net::Network, input::Vector{Float64})
    @assert !any(isnan, input) "input is NaN"
    output = zeros(Float64, net.NumberOfOutputs)
    first_output = length(net.Neurons) - net.NumberOfOutputs + 1
    last_control = net.NumberOfInputs + net.NumberOfControls
    c = 1
    connections = length(net.Connections)
    for i in 1:length(net.Neurons)
        sum = 0.0
        excite = 0.0
        while c <= connections && net.Connections[c].To == i
            conn = net.Connections[c]
            mod = conn.Modulator
            weight = mod != 0 ? net.State[mod] : conn.Weight
            @assert !isnan(weight) "Weight is NaN"
            if net.NumberOfInputs < conn.From <= last_control
                excite += weight * net.State[conn.From]
            else
                sum += weight * net.State[conn.From]
            end
            c += 1
        end
        if i <= net.NumberOfInputs
            sum += input[i]
        end
        @assert !isnan(sum) "sum is NaN $(net)"
        sum = excite < 0.0 ? 0.0 : net.Neurons[i].ActivationFunction(sum)
        @assert !isnan(sum) "activation function returned NaN"
        if i >= first_output
            output[i - first_output + 1] = sum
        end
        state = net.State[i]
        net.State[i] += (sum - state)net.Neurons[i].Speed
        @assert !isnan(net.State[i]) "State is NaN (sum = $sum, speed = $(net.Neurons[i].Speed), state = $state)"
        abs(net.State[i]) > MAX_STATE_VALUE && (net.State[i] = sign(net.State[i])MAX_STATE_VALUE)
    end
    return output
end

NeuroEvolution.reset!(net::Network) = (net.State .= 0.0; nothing)