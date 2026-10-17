include("test_fixtures.jl")

@testset "Spatial Pruning, Mesh & Geometry" begin
    # resolve_bbox delimitation
    lons = [-66.5, -60.0, -58.0]
    lats = [42.0, 45.0, 47.0]
    bbox_raw = resolve_bbox(nothing, 0.5, lons, lats)
    @test bbox_raw[1] == -67.0
    @test bbox_raw[2] == 41.5

    sp_b = (-65.48, 43.04, -57.32, 47.27)
    bbox_delim = resolve_bbox(nothing, 0.5, lons, lats; sppoly_bounds = sp_b)
    @test bbox_delim[1] == -67.0
    @test bbox_delim[2] == 41.5
    @test bbox_delim[3] == -57.32
    @test bbox_delim[4] == 47.5

    @test bbox_delim[1] <= sp_b[1]
    @test bbox_delim[2] <= sp_b[2]
    @test bbox_delim[3] >= sp_b[3]
    @test bbox_delim[4] >= sp_b[4]

    bbox_explicit = resolve_bbox([-62.0, 44.0, -60.0, 46.0], 0.5, lons, lats;
                                 sppoly_bounds = sp_b)
    @test bbox_explicit == (-62.0, 44.0, -60.0, 46.0)

    # prune_mesh
    m_test = build_hex_mesh_planar([-64.0, -62.0], [44.0, 46.0]; radius_km = 30.0)
    S_orig = m_test.n_units
    keep_mask = falses(S_orig)
    keep_mask[1:div(S_orig, 2)] .= true
    m_pruned = prune_mesh(m_test, keep_mask)

    @test m_pruned.n_units == div(S_orig, 2)
    @test length(m_pruned.centroids_lonlat) == div(S_orig, 2)
    @test length(m_pruned.polygons_lonlat) == div(S_orig, 2)
    @test size(m_pruned.W) == (div(S_orig, 2), div(S_orig, 2))
    @test_throws ArgumentError prune_mesh(m_test, falses(S_orig))

    # Domain footprint from spatial-unit file
    mktempdir(TEST_TMP) do dir
        units = DataFrame(
            AUID     = [1, 2],
            wkt_geo  = [
                "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                "POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))",
            ],
            wkt_planar = [
                "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
                "POLYGON ((2 0, 4 0, 4 2, 2 2, 2 0))",
            ],
        )
        path = joinpath(dir, "sppoly.jld2")
        JLD2.save(path, Dict("graph" => [1, 2], "sppoly" => units))

        geom = read_domain_polygon(path)
        @test geom !== nothing
        @test geometry_contains(geom, 1.0, 1.0)
        @test geometry_contains(geom, 3.0, 1.0)
        @test !geometry_contains(geom, 5.0, 1.0)
        @test !geometry_contains(geom, 1.0, 5.0)
        @test geometry_contains(geom, 2.0, 1.0)

        @test geometries_in_domain(geom, [(1.0, 1.0), (3.0, 1.0), (5.0, 1.0)]) ==
              BitVector([true, true, false])
        @test geometries_in_domain(nothing, [(1.0, 1.0), (5.0, 1.0)]) ==
              BitVector([true, true])
        @test read_domain_polygon(nothing) === nothing

        @test_throws ArgumentError read_domain_polygon(joinpath(dir, "absent.jld2"))
    end

    # GeoJSON and shapefile polygon readers
    mktempdir(TEST_TMP) do dir
        p = joinpath(dir, "poly.geojson")
        open(p, "w") do io
            JSON.print(io, Dict(
                "type" => "FeatureCollection",
                "features" => [Dict(
                    "type" => "Feature",
                    "properties" => Dict("name" => "square"),
                    "geometry" => Dict(
                        "type" => "Polygon",
                        "coordinates" => [[[0.0, 0.0], [1.0, 0.0],
                                           [1.0, 1.0], [0.0, 0.0]]],
                    ),
                )],
            ))
        end
        rings = read_polygon_file(p)
        @test length(rings) == 1
        @test length(rings[1]) == 4
        @test rings[1][1] == (0.0, 0.0)
        @test rings[1][3] == (1.0, 1.0)

        cents = [(0.5, 0.5), (5.0, 5.0)]
        @test land_mask_from_polygon_files([p], cents) == BitVector([true, false])
    end

    # Reconnect severed components
    nx, ny = 6, 5
    n = nx * ny
    idx(i, j) = (j - 1) * nx + i
    W = spzeros(n, n)
    nbr = [Int[] for _ in 1:n]
    for j in 1:ny, i in 1:nx
        i < nx && (W[idx(i, j), idx(i + 1, j)] = 1.0; W[idx(i + 1, j), idx(i, j)] = 1.0)
        j < ny && (W[idx(i, j), idx(i, j + 1)] = 1.0; W[idx(i, j + 1), idx(i, j)] = 1.0)
    end
    rows, cols, _ = findnz(W)
    for (r, c) in zip(rows, cols)
        push!(nbr[r], Int(c))
    end
    cents_grid = [(-60.0 + 0.1 * (i - 1), 44.0 + 0.1 * (j - 1)) for j in 1:ny for i in 1:nx]

    keep = falses(n)
    for j in (1, 2), i in 1:nx
        keep[idx(i, j)] = true
    end
    for i in 1:nx
        keep[idx(i, 4)] = true
    end

    out, n_link, n_unlinked = GeoMotion.reconnect_severed_components(keep, W, cents_grid)
    @test n_link == 1
    @test n_unlinked == 0
    @test count(out) == 19
    @test length(GeoMotion._connected_components(out, nbr)) == 1

    # Global land/sea mask
    @test GeoMotion._basemap_grid_token(5.0) == "5"
    @test GeoMotion._basemap_grid_token(10.0) == "10"
    m_mask = land_mask_from_global_mask([(-63.0, 45.5), (-60.0, 45.5)]; resolution = "l", grid_minutes = 5.0)
    @test m_mask[1]
    @test !m_mask[2]
end
