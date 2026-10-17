# ==============================================================================
# GeoMotion Comprehensive Test Suite Orchestrator
# ==============================================================================

import Pkg

const RUNNING_UNDER_PKG_TEST = basename(dirname(Base.active_project())) == "test"
if !RUNNING_UNDER_PKG_TEST
    Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
    Pkg.instantiate(; io = devnull)
end

# Usage:
#   julia --project=. test/runtests.jl                # Run every segment
#   julia --project=. test/runtests.jl config         # Run parameters & config
#   julia --project=. test/runtests.jl spatial        # Run spatial pruning & mesh
#   julia --project=. test/runtests.jl kernel         # Run transition kernel & simulation
#   julia --project=. test/runtests.jl diagnostics    # Run routing, circuits, vis & store
#
# Under `Pkg.test()` the whole suite runs; pass segments as test_args:
#   Pkg.test(test_args = ["config", "spatial"])
# ==============================================================================

using Test
include(joinpath(@__DIR__, "test_fixtures.jl"))

const AVAILABLE_SEGMENTS = Dict{Symbol, @NamedTuple{file::String, desc::String}}(
    :config => (
        file = "test_config.jl",
        desc = "Movement Parameters, Layered Configuration & CLI Parser"
    ),
    :spatial => (
        file = "test_spatial.jl",
        desc = "Spatial Pruning, Mesh Bounding Boxes & Geometric Topologies"
    ),
    :kernel => (
        file = "test_kernel_simulation.jl",
        desc = "Transition Kernel, Bayesian MCMC Inference & Trajectory Simulation"
    ),
    :diagnostics => (
        file = "test_diagnostics_vis.jl",
        desc = "A* Routing, Circuit Theory, Interactive Visualisations & Lakehouse"
    )
)

"""
    parse_requested_segments(args::Vector{String}) -> Vector{Symbol}

Resolves CLI segment arguments to segment keys.
"""
function parse_requested_segments(args::Vector{String})::Vector{Symbol}
    if isempty(args)
        return collect(keys(AVAILABLE_SEGMENTS))
    end

    selected = Symbol[]
    for a in args
        key = Symbol(lowercase(strip(a)))
        if haskey(AVAILABLE_SEGMENTS, key)
            push!(selected, key)
        else
            avail = sort(string.(collect(keys(AVAILABLE_SEGMENTS))))
            @warn "Unrecognized test segment: '$a'. Available segments: $(join(avail, ", "))"
        end
    end

    return isempty(selected) ? sort!(collect(keys(AVAILABLE_SEGMENTS))) : unique(selected)
end

requested_segments = parse_requested_segments(ARGS)

@testset "GeoMotion Comprehensive Suite" begin
    for seg_key in requested_segments
        info = AVAILABLE_SEGMENTS[seg_key]
        println("\n" * "="^80)
        println(">>> Running Segment: $(info.desc) ($(info.file))")
        println("="^80)
        include(joinpath(@__DIR__, info.file))
    end
end
