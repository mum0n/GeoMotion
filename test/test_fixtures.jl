using Test
using GeoMotion
using LinearAlgebra
using SparseArrays
using Random
using DataFrames
using Statistics
using Turing
using TOML
using JLD2
using ForwardDiff
using DynamicPPL
using ArgParse
using Graphs
using JSON
using UUIDs
using LibGEOS

const TEST_TMP = joinpath(pkgdir(GeoMotion), "test_tmp")
mkpath(TEST_TMP)

struct ParName
    s::String
end
Base.string(p::ParName) = "Parameter($(p.s))"

struct FakeChain
    data::Dict{Any,Any}
    n::Int
end
Base.keys(c::FakeChain) = keys(c.data)
Base.getindex(c::FakeChain, k) = c.data[k]
Base.Array(c::FakeChain) = zeros(Float64, c.n, 2)
Base.Array(x::Vector) = x
Base.Array(x::Matrix) = x

const POPULATION_PARAMS = ("mu_velocity", "mu_diffusion", "mu_gamma")
const REMOVED_PARAMS = ("sigma_velocity", "sigma_diffusion", "sigma_gamma",
                        "z_velocity", "z_diffusion", "z_gamma")

function population_level_parameter_names(chain)
    names = Set(string(k) for k in keys(chain))
    return all("Parameter($p)" in names for p in POPULATION_PARAMS) &&
           !any("Parameter($p)" in names for p in REMOVED_PARAMS)
end
