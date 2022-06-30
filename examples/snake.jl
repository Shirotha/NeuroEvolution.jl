#! julia

# FIXME: current setup plataus at ~25 size

include("header.jl")
using Printf
using Serialization
using ArgParse

function keypress()
    ret = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, true)
    ret == 0 || error("unable to switch to raw mode")
    c = read(stdin, Char)
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, false)
    c
end

@everywhere begin

    using .NeuroEvolution, .NeuroEvolution.UnifiedModel, .NeuroEvolution.SUNA

    struct Position
        x::Int
        y::Int
    end

    mutable struct Entity
        Name::String
        Tags::Vector{Symbol}
        Tiles::Vector{Position}
    end

    abstract type AbstractRenderer end
    render(r::T, i::Int, n::Int) where {T <: AbstractRenderer} = 
        throw(MissingException("render(::$T, ::Int, ::Int) not implemented"))

    struct SimpleRenderer <: AbstractRenderer
        Glyph::String
    end
    render(r::SimpleRenderer, i::Int, n::Int) = r.Glyph

    struct HeadTailRenderer{H <: AbstractRenderer, T <: AbstractRenderer} <: AbstractRenderer
        Head::H
        Tail::T
    end
    render(r::HeadTailRenderer, i::Int, n::Int) = i == 1 ? render(r.Head, 1, 1) : render(r.Tail, i - 1, n - 1)

    struct RendererWrapper{T <: AbstractRenderer} <: AbstractRenderer
        Prefix::String
        Renderer::T
        Poxtfix::String
    end
    render(r::RendererWrapper, i::Int, n::Int) = 
        i == 1 ? r.Prefix * render(r.Renderer, 1, n) :
        1 == n ? render(r.Renderer, n, n) * r.Postfix :
                render(r.Renderer, i, n)

    mutable struct Snake
        Left::Int
        Top::Int
        Width::Int
        Height::Int

        UIWidth::Int
        UIHandlers::Dict{Symbol, Tuple{Position, Int, Function}}

        InitalSnakeLength::Int

        Heading::Symbol
        Entities::Vector{Entity}
        Renderers::Vector{AbstractRenderer}

        function Snake(w, h;
            left = 2,
            top = 2,
            initial_snake_length = 3,
            ui_width = 10
        )
            @assert left > 0 "playfield off screen"
            @assert top > 0 "playfield off screen"
            @assert initial_snake_length > 1 "snake too short"
            @assert w > 2initial_snake_length "playfield too small"
            @assert h > 2initial_snake_length "playfield too small"
            @assert ui_width >= 0 "UI width negative"
            new(left, top, w, h, ui_width, Dict(), initial_snake_length, :north, [
                Entity("Snake", [:Snake], [Position(w ÷ 2, h ÷ 2 + i - 1) for i in 1:initial_snake_length]),
                Entity("Apple", [:Apple], [Position(rand(1:w), rand(1:(h ÷ 2 - 1)))])
            ],[
                RendererWrapper("", HeadTailRenderer(SimpleRenderer("\e[93mO\e[92m"), SimpleRenderer("O")), "\e[0m"),
                SimpleRenderer("\e[91m*\e[0m")
            ])
        end
    end
    function NeuroEvolution.reset!(g::Snake)
        g["Snake"].Tiles = [Position(g.Width ÷ 2, g.Height ÷ 2 + i - 1) for i in 1:g.InitalSnakeLength]
        g["Apple"].Tiles[1] = Position(rand(1:g.Width), rand(1:(g.Height ÷ 2 - 1)))
        g.Heading = :north
    end
    function cycle(p::Position, g::Snake)
        x = p.x
        while x < 1
            x += g.Width
        end
        while x > g.Width
            x -= g.Width
        end
        y = p.y
        while y < 1
            y += g.Height
        end
        while y > g.Height
            y -= g.Height
        end
        Position(x, y)
    end
    Base.length(g::Snake) = length(g.Entities)
    Base.size(g::Snake) = size(g.Entities)
    Base.keys(g::Snake) = getproperty.(g.Entities, :Name)
    function Base.getindex(g::Snake, name::String)
        index = findfirst(e -> e.Name == name, g.Entities)
        isnothing(index) ? nothing : g.Entities[index]
    end
    function Base.getindex(g::Snake, tag::Symbol)
        callback(e) = any(==(tag), e.Tags)
        index = 0
        result = Entity[]
        while !(isnothing)
            index = findnext(callback, e.Entities, index + 1)
            isnothing(index) && return result
            push!(result, g.Entities[index])
        end
        result
    end
    function Base.getindex(g::Snake, p::Position, tag=nothing)
        for e in g
            any(==(p), e.Tiles) && (isnothing(tag) || any(==(tag), e.Tags)) && return e
        end
        nothing
    end
    Base.iterate(g::Snake) = iterate(g.Entities)
    Base.iterate(g::Snake, state) = iterate(g.Entities, state)

end

function render(g::Snake)
    buffer = ""
    for i in keys(g.Entities)
        e = g.Entities[i]
        r = g.Renderers[i]
        n = length(e.Tiles)
        for (k, t) in enumerate(e.Tiles)
            x = t.x + g.Left - 1
            y = t.y + g.Top - 1
            buffer *= "\e[$(y);$(x)H" * render(r, k, n)
        end
    end
    println(buffer)
end

function render_ui(g::Snake, x::Int, y::Int, text::String; width::Int = g.UIWidth)
    @assert 1 <= x <= g.UIWidth "UI outside of box"
    @assert 1 <= y <= g.Height "UI outside of box"
    @assert 1 <= width <= g.UIWidth - x + 1 "UI overflow"
    # TODO: exclude control sequences
    # text = string(Iterators.take(text, g.UIWidth - x + 1)...)
    print("\e[$(y + g.Top - 1);$(x + g.Left + g.Width)H", repeat(" ", width), "\e[$(width)D", text)
end
function is_ui_free(g::Snake, x::Int, y::Int, width::Int)
    for (_, h) in g.UIHandlers
        y == h[1].y || continue
        (x + width - 1 < h[1].x || h[1].x + h[2] - 1 < x) || return false
    end
    return true
end
function register_ui(callback::Function, g::Snake, name::Symbol, x::Int, y::Int; width = g.UIWidth - x + 1)
    is_ui_free(g, x, y, width) || error("UI overlap")
    f = (g, params) -> render_ui(g, x, y, callback(g, params); width)
    g.UIHandlers[name] = (Position(x, y), width, f)
end
function register_ui(callbacks::Vector{Function}, g::Snake, name::Symbol, x::Int, y::Int; width = g.UIWidth - x + 1)
    for (dy, callback) in enumerate(callbacks)
        register_ui(callback, g, Symbol(name, dy), x, y + dy - 1; width)
    end
end
function render_ui(g::Snake, name::Symbol)
    last(g.UIHandlers[name])(g)
end
function render_ui(g::Snake; kwargs...)
    params = Dict(kwargs)
    for (name, ui) in g.UIHandlers
        last(ui)(g, params)
    end
end

FRAME_NONE      = "               "
FRAME_LIGHT     = "┌─┐  │   │└─┘  "
FRAME_HEAVY     = "┏━┓  ┃   ┃┗━┛  "
FRAME_DOUBLE    = "╔═╗  ║   ║╚═╝  "

FRAME_LIGHT_UI  = "┌─┬─┐│ │ │└─┴─┘"
FRAME_HEAVY_UI  = "┏━┳━┓┃ ┃ ┃┗━┻━┛"
FRAME_DOUBLE_UI = "╔═╦═╗║ ║ ║╚═╩═╝"

function prerender_background(g::Snake;
    frame = FRAME_NONE
)
    is = collect(keys(frame))
    @assert length(is) == 15
    c(i) = frame[is[i]]
    row(r) = let di = 5(r - 1)
        c(di + 1) * repeat(c(di + 2), g.Width) * c(di + 3) * repeat(c(di + 4), g.UIWidth) * c(di + 5) * '\n'
    end
    "\e[$(g.Top - 1);$(g.Left - 1)H" * row(1) * repeat(row(2), g.Height) * row(3)
end

@everywhere begin

    function input!(g::Snake, cmd::Symbol)
        function loc(dx, dy)
            current = g["Snake"].Tiles[1]
            x = mod(current.x + dx - 1, g.Width) + 1
            y = mod(current.y + dy - 1, g.Height) + 1
            Position(x, y)
        end
        function move(dx, dy)
            snake = g["Snake"]
            p = loc(dx, dy)
            res = :nothing
            e = g[p]
            if isnothing(e)
                deleteat!(snake.Tiles, lastindex(snake.Tiles))
            elseif any(==(:Apple), e.Tags)
                a = p
                while !isnothing(g[a])
                    a = Position(rand(1:g.Width), rand(1:g.Height))
                end
                e.Tiles[1] = a
                res = :apple
            elseif any(==(:Snake), e.Tags)
                return :gameover
            else
                error("collided with unknown entity $e")
            end
            pushfirst!(g.Entities[1].Tiles, p)
            res
        end
        function behind(heading)
            if heading == :north
                return :south
            elseif heading == :south
                return :north
            elseif heading == :west
                return :east
            elseif heading == :east
                return :west
            else
                error("unknown heading")
            end
        end
        function rightof(heading)
            if heading == :north
                return :east
            elseif heading == :south
                return :west
            elseif heading == :west
                return :north
            elseif heading == :east
                return :south
            else
                error("unknown heading")
            end
        end
        function leftof(heading)
            if heading == :north
                return :west
            elseif heading == :south
                return :east
            elseif heading == :west
                return :south
            elseif heading == :east
                return :north
            else
                error("unknown heading")
            end
        end
        function move(heading)
            if heading == :north
                return move(0, -1)
            elseif heading == :south
                return move(0, 1)
            elseif heading == :west
                return move(-1, 0)
            elseif heading == :east
                return move(1, 0)
            else
                error("unknown heading")
            end
        end
        function isheading(heading)
            heading == :north || heading == :south || heading == :west || heading == :east
        end

        cmd == behind(g.Heading) && return :skip

        if cmd == :forward
            return move(g.Heading)
        elseif cmd == :right
            g.Heading = rightof(g.Heading)
            return move(g.Heading)
        elseif cmd == :left
            g.Heading = leftof(g.Heading)
            return move(g.Heading)
        end

        if isheading(cmd)
            g.Heading = cmd
            return move(g.Heading)
        end

        error("unknown command")
    end

    function get_vision(g::Snake; range = 6, tag = nothing)
        function raycast(p, dx, dy)::Int
            dx == 0 && dy == 0 && return 0.0
            for i in 1:range
                p = cycle(Position(p.x + dx, p.y + dy), g)
                e = g[p, tag]
                isnothing(e) && continue
                !isnothing(tag) && return i
                if any(==(:Apple), e.Tags)
                    return i
                elseif any(==(:Snake), e.Tags)
                    return -i
                else
                    error("unknown entity")
                end
            end
            0
        end
        head = g["Snake"].Tiles[1]
        if g.Heading == :north
            return [raycast(head, dx, dy) for dy in -1:1, dx in -1:1]
        elseif g.Heading == :south
            return [raycast(head, -dx, -dy) for dy in -1:1, dx in -1:1]
        elseif g.Heading == :west
            return [raycast(head, dy, -dx) for dy in -1:1, dx in -1:1]
        elseif g.Heading == :east
            return [raycast(head, -dy, dx) for dy in -1:1, dx in -1:1]
        else
            error("unknown heading") 
        end
    end

end

function create_playfield(w::Int, h::Int)
    snake = Snake(w, h)
    y = 1
    next(n = 1) = (res = y; y += n; res)
    register_ui(snake, :position, 1, next()) do g, params
        p = g["Snake"].Tiles[1]
        "($(p.x), $(p.y))"
    end
    register_ui(snake, :score, 1, next()) do g, params
        string("Score: ", params[:score])
    end
    register_ui(snake, :high_score, 1, next()) do g, params
        string("Best: ", params[:high_score])
    end
    next()
    row = r -> begin
        result = ""
        for d in r
            result *= d < 0 ? "\e[92m" : d > 0 ? "\e[91m" : "\e[0m"
            result *= @sprintf "%2i" abs(d)
            result *= "\e[0m"
        end
        result
    end
    register_ui([
        (g, params) -> row(params[:vision][1,:]),
        (g, params) -> row(params[:vision][2,:]),
        (g, params) -> row(params[:vision][3,:])
    ], snake, :vision, 3, next(3))
    snake
end

function main_loop(snake::Snake, data=nothing; vision_range=6, frame = FRAME_DOUBLE_UI)
    try
        Base.exit_on_sigint(false)
        print("\e[2J", "\e[?25l")
        bg = prerender_background(snake; frame)
        res = nothing
        score = 0
        high_score = 0
        while true
            print(bg)
            render(snake)
            render_ui(snake;
                score,
                high_score,
                vision = get_vision(snake; range = vision_range)
            )

            res = nothing
            if isnothing(data)
                c = keypress()
                (c == 'q' || c == '\e' || c == '\x03') && break

                c == 'w' && (res = input!(snake, :north))
                c == 'a' && (res = input!(snake, :west))
                c == 's' && (res = input!(snake, :south))
                c == 'd' && (res = input!(snake, :east))
                c == ' ' && (res = input!(snake, :forward))
                c == 'f' && (res = input!(snake, :left))
                c == 'j' && (res = input!(snake, :right))
            else
                sleep(0.1)
                res = input!(snake, data.Brain(get_ai_vision(snake; range = data.Vision, tags = data.Tags)))
            end

            if res == :gameover
                reset!(snake)
                isnothing(data) || reset!(data.Brain)
                score > high_score && (high_score = score)
                score = 0
            elseif res == :apple
                score += 1
            end
        end
    catch e
        e isa InterruptException || throw(e)
    finally
        Base.exit_on_sigint(true)
        print("\e[$(snake.Top + snake.Height + 1)H\e[?25h")
    end
end

@everywhere begin

    function get_ai_vision(g::Snake; range=6, tags=nothing)
        if isnothing(tags)
            vision = reshape(get_vision(g; range), 9)
            deleteat!(vision, 5)
            return map(vision) do dist
                dist == 0 ? 0.0 : 1.0 / dist
            end
        else
            visions = map(tags) do tag
                vision = reshape(get_vision(g; range, tag), 9)
                deleteat!(vision, 5)
                map(vision) do dist
                    dist == 0 ? 0.0 : 1.0 / dist
                end
            end
            return collect(Iterators.flatten(visions))
        end
    end
    function input!(g::Snake, output::Vector{Float64})
        (_, i) = findmax(output)
        if i == 1
            return input!(g, :left)
        elseif i == 2
            return input!(g, :forward)
        elseif i == 3
            return input!(g, :right)
        else
            error("bad output")
        end
    end

    mutable struct SnakeEnv <: Environment
        VisionRange::Int
        VisionTags::Union{Nothing, Vector{Symbol}}

        Snake::Snake
        Score::Int
        Time::Int

        Checkpoint::Int
        Bonus::Int

        SnakeEnv(w, h; vision_range=6, initial_snake_length=3, vision_tags=nothing) = 
            new(vision_range, vision_tags, Snake(10, 10; initial_snake_length), 0, 0, 0, 0)
    end

    NeuroEvolution.inputform(::Type{SnakeEnv}) = zeros(Float64, 8)
    NeuroEvolution.inputform(env::SnakeEnv) = zeros(Float64, isnothing(env.VisionTags) ? 8 : 8length(env.VisionTags))
    NeuroEvolution.outputform(::Type{SnakeEnv}) = zeros(Float64, 3)

    NeuroEvolution.process!(env::SnakeEnv, output) = isnothing(output) ? process!(env) : process!(env, convert(Vector{Float64}, output))
    function NeuroEvolution.process!(env::SnakeEnv, output::Union{Nothing, Vector{Float64}} = nothing)
        if isnothing(output)
            reset!(env.Snake)
            env.Score = 0
            env.Time = 0
            env.Checkpoint = env.Snake.Width + env.Snake.Height
            env.Bonus = 0
        else
            res = input!(env.Snake, output)
            res == :gameover && return nothing

            env.Time += 1
            env.Time > env.Checkpoint && return nothing

            if res == :apple
                env.Score += 1
                env.Bonus += env.Checkpoint - env.Time
                env.Checkpoint = env.Time + max(env.Snake.Width + env.Snake.Height, env.Score)
            end
        end
        get_ai_vision(env.Snake; range = env.VisionRange, tags = env.VisionTags)
    end

    NeuroEvolution.properties(::Type{<:SnakeEnv}) = [:Default, :Score, :Time]
    function NeuroEvolution.fitness(env::SnakeEnv, ::Property"Default")
        function normalized_bonus()
            env.Score == 0 && return 0.0
            x = env.Snake.Width + env.Snake.Height
            env.Score <= x && return env.Bonus / (x * env.Score)
            extra = env.Score - x
            env.Bonus / (x ^ 2 + (env.Score + 1)env.Score / 2 - (extra + 1)extra / 2)
        end
        env.Score + normalized_bonus()
    end
    NeuroEvolution.fitness(env::SnakeEnv, ::Property"Score") = env.Score
    NeuroEvolution.fitness(env::SnakeEnv, ::Property"Time") = env.Time
    NeuroEvolution.fitness(env::SnakeEnv, ::Property"Bonus") = env.Bonus

end

struct Data
    Brain::Network

    Fitness::Float64
    Generation::Int
    Time::Float64

    Width::Int
    Height::Int
    Vision::Int
    Tags::Union{Nothing, Vector{Symbol}}

    SunaArgs::Dict{Symbol, Any}
end

function train(goal::Function, w, h; 
        vision_range = 6, 
        initial_snake_length = 3,
        vision_tags = nothing,
        suna_args...
)
    suna = SunaSpeciate(SnakeEnv(w, h; 
            vision_range, 
            initial_snake_length,
            vision_tags
        ); 
        steps_per_generation = typemax(Int), 
        suna_args...)

    gen = 0
    fitness = -Inf
    fittest = nothing
    tim = time()
    dt = 0.0
    try
        Base.exit_on_sigint(false)
        while !goal(gen, fitness, dt)
            candidate, new_fitness = process!(suna)
            size = 0
            for nn in suna.Population
                size += length(nn.Neurons)
            end
            size /= length(suna.Population)
            species = length(suna.Selector)

            new_fitness >= fitness && (fittest = candidate; fitness = new_fitness)
            gen += 1
            dt = time() - tim
            @printf "\e[J[%i:%02i] gen %i: Fitness = %.2f (%.2f) (avg. size: %i, species: %i)\r" (dt ÷ 60) mod(dt, 60) gen new_fitness fitness size species
        end
    catch e
        # TODO: confirm these are the only possoblilities
        if !(e isa InterruptException || 
            (e isa RemoteException && e.captured.ex isa InterruptException) || 
            (e isa CompositeException && any(x -> 
                x isa InterruptException || 
                (x isa RemoteException && x.captured.ex isa InterruptException), e))
            )
            println(repeat("=",40))
            if e isa CompositeException
                println("COMPOSITE")
                for r in e
                    @show r
                end
            else
                @show e
            end
            println(repeat("=",40))
            throw(e)
        end
    finally
        Base.exit_on_sigint(true)
    end

    isnothing(fittest) && return nothing
    Data(fittest, fitness, gen, dt, w, h, vision_range, vision_tags, Dict(suna_args...))
end

function options()
    s = ArgParseSettings(autofix_names = true)
    @add_arg_table! s begin
        "play"
            help = "play snake"
            action = :command
        "train"
            help = "train a new brain"
            action = :command
        "identify"
            help = "show information for a previously saved brain"
            action = :command
        "--width", "-x"
            help = "width of the play field"
            arg_type = Int
            default = 10
        "--height", "-y"
            help = "height of the play field"
            arg_type = Int
            default = 10
        "--vision", "-v"
            help = "amount of vision"
            arg_type = Int
            default = 6
    end
    @add_arg_table! s["play"] begin
        "--brain", "-b"
            help = "filename to a previously trained brain (overrides size and vision to match training)"
    end
    @add_arg_table! s["train"] begin
        "--population", "-n"
            help = "size of a generation"
            arg_type = Int
            default = 100
        "--evaluations", "-e"
            help = "number of evaluations per generation"
            arg_type = Int
            default = 1
        "--speciation-distance", "-d"
            help = "size of the novelty map"
            arg_type = Float64
            default = 1.5
        "--parent-cutoff", "-p"
            help = "percent of population used as parents"
            arg_type = Float64
            default = 0.2
        "--add-neuron-chance"
            help = "percent chance to add a new neuron during mutation"
            arg_type = Float64
            default = 0.1
        "--add-connection-chance"
            help = "percent chance to add a new connection during mutation"
            arg_type = Float64
            default = 0.4
        "--remove-neuron-chance"
            help = "percent chance to remove a neuron during mutation"
            arg_type = Float64
            default = 0.0
        "--remove-connection-chance"
            help = "percent chance to remove a connection during mutation"
            arg_type = Float64
            default = 0.0
        "--weight-change-chance"
            help = "percent cahnce to change weights during mutation"
            arg_type = Float64
            default = 0.5
        "--max-generation", "-g"
            help = "generation cutoff (0 = unlimited)"
            arg_type = Int
            default = 0
        "--target-fitness", "-f"
            help = "training stops at this fitness (0 = unlimited)"
            arg_type = Float64
            default = 0.0
        "--max-time", "-t"
            help ="maximum training duration in minutes (0 = unlimited)"
            arg_type = Float64
            default = 0.0
        "--file", "-o"
            help = "filename to store the brain"
    end
    @add_arg_table! s["identify"] begin
        "file"
            help = "the file to identify"
            required = true
    end

    parse_args(s, as_symbols = true)
end


function main()
    opts = options()

    width = opts[:width]
    height = opts[:height]
    vision = opts[:vision] 

    cmd = isnothing(opts[:_COMMAND_]) ? Dict() : opts[opts[:_COMMAND_]]

    if opts[:_COMMAND_] == :play
        brain = cmd[:brain]
        if !isnothing(brain)
            data = deserialize(brain)
            width = data.Width
            height = data.Height
        end

        main_loop(create_playfield(width, height), data)
    elseif opts[:_COMMAND_] == :train
        max_n = cmd[:max_generation]
        max_n <= 0 && (max_n = typemax(Int))
        max_f = cmd[:target_fitness]
        max_f <= 0.0 && (max_f = Inf)
        max_t = 60cmd[:max_time]
        max_t <= 0.0 && (max_t = Inf)

        dist = cmd[:speciation_distance]
        if dist <= 0
            println("speciation distance has to be positive")
            exit(1)
        end

        par = cmd[:parent_cutoff]
        if !(0 < par < 1)
            println("parent cutoff needs to be between 0 and 1")
            exit(1)
        end

        add_n = cmd[:add_neuron_chance]
        add_n < 0 && (add_n = 0.0)
        add_c = cmd[:add_connection_chance]
        add_c < 0 && (add_c = 0.0)
        remove_n = cmd[:remove_neuron_chance]
        remove_n < 0 && (remove_n = 0.0)
        remove_c = cmd[:remove_connection_chance]
        remove_c < 0 && (remove_c = 0.0)

        if add_n + add_c + remove_n + remove_c > 1
            println("total mutations add to over 100%")
            exit(1)
        end

        weight = cmd[:weight_change_chance]
        if !(0 <= weight <= 1)
            println("weight mutation chance needs to be between 0 and 1")
        end

        data = train(width, height; 
            vision_range = vision,
            vision_tags = [:Snake, :Apple],
            population_size = cmd[:population],
            evaluation_count = cmd[:evaluations],
            initial_mutations = 1,
            mutations_per_generation = 1,
            speciation_distance = dist,
            cutoff_percent = par,
            add_neuron_chance = add_n,
            add_connection_chance = add_c,
            remove_neuron_chance = remove_n,
            remove_connection_chance = remove_c,
            weight_mutation_chance = weight
        ) do n, f, t
            n >= max_n || f >= max_f || t >= max_t
        end

        isnothing(data) && exit()

        file = cmd[:file]
        !isnothing(file) && serialize(file, data)

        main_loop(create_playfield(data.Width, data.Height), data)
    elseif opts[:_COMMAND_] == :identify
        file = cmd[:file]
        data = deserialize(file)
        print("Brain ", data.Width, "x", data.Height, "@", data.Vision)
        !isnothing(data.Tags) && print(" (", join(data.Tags, ", "), ")")
        print('\n')
        @printf "Generation %i (trained for %i:%02i minutes)\n" data.Generation (data.Time ÷ 60) mod(data.Time, 60)
        @printf "Fitness %.2f\n" data.Fitness
        println("SUNA Parameters")
        for (name, value) in data.SunaArgs
            println("  ", name, " => ", value)
        end
        println("Network Spectrum")
        println("  Neurons ", length(data.Brain.Neurons))
        println("  Connections ", length(data.Brain.Connections))
        println("  Inputs ", data.Brain.NumberOfInputs, " (", data.Brain.Neurons[begin].ActivationFunction.Name, ")")
        println("  Outputs ", data.Brain.NumberOfOutputs, " (", data.Brain.Neurons[end].ActivationFunction.Name, ")")
        println("  Controls ", data.Brain.NumberOfControls)
        types = Dict{Symbol, Int}()
        for neuron in data.Brain.Neurons[(begin + data.Brain.NumberOfInputs + data.Brain.NumberOfControls):(end - data.Brain.NumberOfOutputs)]
            type = neuron.ActivationFunction.Name
            if haskey(types, type)
                types[type] += 1
            else
                types[type] = 1
            end
        end
        for (type, amount) in types
            println("  ", type, " ", amount)
        end
    end

end

main()