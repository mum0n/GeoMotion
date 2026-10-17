include("test_fixtures.jl")

@testset "Diagnostics, Circuits, Visualisation & Lakehouse" begin
    # A* Least-Cost Routing
    W_astar = spzeros(4, 4)
    W_astar[1, 2] = 1.0; W_astar[2, 1] = 1.0
    W_astar[1, 3] = 1.0; W_astar[3, 1] = 1.0
    W_astar[2, 4] = 1.0; W_astar[4, 2] = 1.0
    W_astar[3, 4] = 1.0; W_astar[4, 3] = 1.0
    coords_astar = [(0.0, 0.0), (1.0, 1.0), (1.0, -1.0), (2.0, 0.0)]
    path_astar = astar_least_cost_path(coords_astar, W_astar, 1, 4)
    @test length(path_astar) >= 2
    @test first(path_astar) == 1 && last(path_astar) == 4

    # Circuit Theory Diagnostics
    W_circ = spzeros(3, 3)
    W_circ[1, 2] = 1.0; W_circ[2, 1] = 1.0
    W_circ[2, 3] = 1.0; W_circ[3, 2] = 1.0
    L, C = build_circuit_laplacian(W_circ)
    @test size(L) == (3, 3)
    @test isapprox(sum(L, dims = 2), zeros(3, 1); atol = 1e-10)
    R_eff = effective_resistance_matrix(L)
    @test R_eff[1, 3] > R_eff[1, 2]

    # Posterior Path Ensemble Panel
    S_ens = 5
    cents_ens = [(-64.0 + 0.5 * i, 45.0 + 0.2 * i) for i in 1:S_ens]
    polys_ens = [[(c[1] - 0.2, c[2] - 0.1), (c[1] + 0.2, c[2] - 0.1),
              (c[1] + 0.2, c[2] + 0.1), (c[1] - 0.2, c[2] + 0.1)] for c in cents_ens]
    au_ens = (centroids = cents_ens, centroids_lonlat = cents_ens,
          polygons = polys_ens, polygons_lonlat = polys_ens, n_units = S_ens)
    hsi_ens = collect(range(0.1, 0.9; length = S_ens))
    agreed = (ensemble_paths = Dict("A" => [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
              ensemble_corridors = Dict("A" => zeros(5, 5)))
    m_ens = plot_posterior_path_ensemble(agreed, au_ens; hsi = hsi_ens)
    @test m_ens isa InteractiveMap
    @test occursin("\"modal_share\": 1.0", m_ens.html)

    # Style settings
    light = plot_hsi_map(hsi_ens[1:3], (centroids = cents_ens[1:3], centroids_lonlat = cents_ens[1:3], polygons = polys_ens[1:3], polygons_lonlat = polys_ens[1:3], n_units = 3); cmap = :viridis, dark_mode = false)
    dark  = plot_hsi_map(hsi_ens[1:3], (centroids = cents_ens[1:3], centroids_lonlat = cents_ens[1:3], polygons = polys_ens[1:3], polygons_lonlat = polys_ens[1:3], n_units = 3); cmap = :viridis, dark_mode = true)
    @test light.html != dark.html

    # Trait-Movement Associations OLS
    obs_df = DataFrame(tag_id = 1:10, carapace_width = [95.0, 102.0, 108.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0])
    mov_stats = (net_displacement_km = [10.0, 12.0, 15.0, 18.0, 20.0, 24.0, 28.0, 30.0, 33.0, 36.0],
                 path_efficiency = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72],
                 tortuosity = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1])
    assoc = model_trait_movement_associations(obs_df, mov_stats, (au = nothing,))
    @test haskey(assoc.models, "displacement")
    @test assoc.models["displacement"].r2 > 0.9

    # Failed panels recording
    sk = GeoMotion.PANEL_SKIPS
    empty!(sk)
    GeoMotion._record_panel_skip("Widget", ErrorException("boom"))
    @test length(sk) == 1
    @test GeoMotion.report_panel_skips() == 1
    empty!(sk)

    # Input assimilation
    tmp_assim = joinpath(TEST_TMP, "assimilation_$(UUIDs.uuid4())")
    input_dir = joinpath(tmp_assim, "data")
    store_dir = joinpath(tmp_assim, "store")
    mkpath(input_dir)
    tagging = DataFrame(aid = [1, 2, 3], lon = [-1.0, 0.0, 1.0], lat = [44.0, 45.0, 46.0])
    hsi_assim = (sims = reshape(collect(0.1:0.1:(0.1 * 6 * 2 * 3)), 6, 2, 3), years = [1999, 2000], auids = collect(1:6))
    sppoly = DataFrame(AUID = collect(1:4), centroid_lon = 1.0:4.0)
    JLD2.save(joinpath(input_dir, "tagging.jld2"), "tagging", tagging)
    JLD2.save(joinpath(input_dir, "hsi.jld2"), "sims", hsi_assim.sims, "years", hsi_assim.years, "auids", hsi_assim.auids)
    JLD2.save(joinpath(input_dir, "sppoly.jld2"), "graph", Graphs.SimpleGraph(4), "sppoly", sppoly)

    r_assim = GeoMotion.assimilate_movement_inputs(input_dir = input_dir, store_dir = store_dir, verbose = false)
    @test r_assim.written == ["tagging", "hsi", "sppoly"]
    back_assim = GeoMotion.read_assimilated_movement_inputs(store_dir = store_dir)
    @test size(back_assim.tagging, 1) == 3
    rm(tmp_assim; recursive = true, force = true)
end
