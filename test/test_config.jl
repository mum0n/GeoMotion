include("test_fixtures.jl")

@testset "Parameters, Configuration & CLI" begin
    p_def = movement_parameters_default()
    @test p_def isa GeoMotionConfig
    @test p_def.tagging_file === nothing
    @test p_def.path_methods == [:astar]
    @test p_def.model_modes isa Vector{Symbol}
    @test p_def.diagnostics isa Vector{Symbol}
    @test p_def.land_source === :landmask

    c = GeoMotionConfig(; tagging_file = "x.jld2", n_samples = 7)
    @test c.tagging_file == "x.jld2"
    @test c.n_samples == 7
    @test c.path_methods == [:astar]

    c2 = load_config(cli_args = [
        "--samples=9", "--warmup=4", "--depth-range=10,50",
        "--model-modes=telemetry", "--land-source=polygons",
        "--cmap=plasma", "--region-labels=A,B", "--render-html=false",
    ])
    @test c2.n_samples == 9
    @test c2.n_warmup == 4
    @test c2.depth_range == [10.0, 50.0]
    @test c2.model_modes == [:telemetry]
    @test c2.land_source === :polygons
    @test c2.cmap === :plasma
    @test c2.region_labels == ["A", "B"]
    @test c2.render_html === false

    @test load_config(cli_args = ["--tessellation-only"]).tessellation_only === true
    @test load_config().tessellation_only === false

    c_all = load_config(cli_args = ["--all"])
    @test c_all.all === true
    @test :agent in c_all.model_modes
    @test :viterbi in c_all.path_methods
    @test :bayesian_ensemble in c_all.diagnostics

    @test load_config().bathymetry_source === nothing
    @test load_config(cli_args = ["--bathymetry-source=data/bathy.csv"]).bathymetry_source == "data/bathy.csv"

    mktempdir(TEST_TMP) do dir
        p = joinpath(dir, "c.toml")
        write(p, "bathymetry_source = \"data/real_bathy.jld2\"\n")
        @test load_config(config_path = p).bathymetry_source == "data/real_bathy.jld2"

        p_tess = joinpath(dir, "tess.toml")
        write(p_tess, "tessellation_only = true\n")
        @test load_config(config_path = p_tess).tessellation_only === true

        p_cfg = joinpath(dir, "cfg.toml")
        write(p_cfg, """
            species_name = "Snow Crab"
            n_samples = 500
            model_modes = ["telemetry", "telemetry_and_survey", "agent"]
            depth_range = [25.0, 400.0]
            region_labels = []
            """)
        from_file = load_config(config_path = p_cfg, cli_args = ["--max-paths=2"])
        @test from_file.species_name == "Snow Crab"
        @test from_file.n_samples == 500
        @test from_file.model_modes == [:telemetry, :telemetry_and_survey, :agent]
        @test from_file.depth_range == [25.0, 400.0]
        @test from_file.max_paths == 2

        out = joinpath(dir, "round.toml")
        save_config(from_file, out)
        back = GeoMotionConfig(out)
        bad = [f for f in fieldnames(GeoMotionConfig)
               if f !== :show_help && !isequal(getfield(from_file, f), getfield(back, f))]
        @test isempty(bad)
    end

    @test effective_model_modes(c) == setdiff(c.model_modes, GeoMotion.survey_dependent_modes())
    @test isempty(intersect(effective_model_modes(c), GeoMotion.survey_dependent_modes()))

    @test_throws ArgumentError GeoMotion.apply_overrides(
        GeoMotionConfig(), Dict{Symbol,Any}(:not_a_field => 1))

    for bad_flag in ("--diagnostics=valdiation", "--model-modes=telmetry",
                "--path-methods=astar,dijkstra")
        @test_throws ArgumentError load_config(cli_args = [bad_flag])
    end
    @test load_config(cli_args = ["--diagnostics=validation"]).diagnostics == [:validation]
    @test load_config(cli_args = ["--path-methods=astar,viterbi"]).path_methods == [:astar, :viterbi]

    s = create_argparse_settings()
    @test s isa ArgParse.ArgParseSettings
    known = Set(fieldnames(GeoMotionConfig))
    unbound = [f.dest_name for f in s.args_table.fields
               if f.dest_name != "config" && !(Symbol(f.dest_name) in known)]
    @test isempty(unbound)
end
