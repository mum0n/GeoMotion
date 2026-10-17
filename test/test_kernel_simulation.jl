include("test_fixtures.jl")

@testset "Kernel, Bayesian Inference & Trajectory Simulation" begin
    # Transition Kernel Construction
    W = spzeros(5, 5)
    for i in 1:4
        W[i, i+1] = 1.0
        W[i+1, i] = 1.0
    end
    hsi = [0.2, 0.4, 0.6, 0.8, 1.0]

    P = construct_stochastic_transition_kernel(
        W, hsi; gamma = 1.0, residence = 0.2, advection = 0.5
    )
    @test size(P) == (5, 5)
    @test all(isapprox.(vec(sum(P; dims = 2)), 1.0; atol = 1e-6))

    for (gamma, rho, alpha) in ((0.0, 0.1, 0.0), (0.0, 0.9, 1.0),
                                (10.0, 0.5, 0.5), (-10.0, 0.5, 0.5),
                                (1.0, 0.5, 0.5))
        P_sp = build_sparse_transition_kernel(W, hsi, gamma, rho, alpha, nothing)
        @test all(P_sp .>= 0.0)
        @test all(isapprox.(vec(sum(P_sp; dims = 2)), 1.0; atol = 1e-6))
    end

    # Explicit Turing Telemetry Model
    W_t = spzeros(4, 4)
    W_t[1, 2] = 1.0; W_t[2, 1] = 1.0
    W_t[2, 3] = 1.0; W_t[3, 2] = 1.0
    W_t[3, 4] = 1.0; W_t[4, 3] = 1.0
    hsi_t = [0.2, 0.5, 0.8, 0.3]
    releases   = [1, 2, 2]
    recaptures = [2, 3, 4]
    ks         = [1, 1, 2]

    m_tel = pure_telemetry_turing_model(
        releases, recaptures, ks, W_t, hsi_t, zeros(Bool, 4)
    )
    @test m_tel isa DynamicPPL.Model

    chn = sample(MersenneTwister(42), m_tel, MH(), 40; progress = false)
    @test size(chn, 1) == 40
    @test population_level_parameter_names(chn)

    # movement_alpha_rho
    for v in (0.05, 0.2, 0.5, 0.9)
        for d in (0.01, 0.1, 0.4, 1.0)
            a, r = GeoMotion.movement_alpha_rho(v, d)
            tot = v + d + 1e-6
            @test a ≈ clamp(v / tot, 0.0, 1.0)
            @test r ≈ clamp(1.0 / (1.0 + tot), 0.01, 0.95)
        end
    end

    # Posterior Extraction from a Population-Level Chain
    n_draws = 4
    c = FakeChain(Dict(
        ParName("mu_velocity")  => collect(range(0.2, 0.5; length = n_draws)),
        ParName("mu_diffusion") => fill(0.05, n_draws),
        ParName("mu_gamma")     => fill(1.0, n_draws),
    ), n_draws)

    v, d, g = GeoMotion.posterior_kernel_draws(c)
    @test v isa Vector{Float64} && length(v) == n_draws
    @test d isa Vector{Float64} && length(d) == n_draws
    @test g isa Vector{Float64} && length(g) == n_draws

    # kstep_transition_cache
    S = 12
    W_k = spzeros(S, S)
    for i in 1:S-1
        W_k[i, i+1] = 1.0; W_k[i+1, i] = 1.0
    end
    hsi_k = collect(range(0.1, 0.9; length = S))
    P_k = build_sparse_transition_kernel(W_k, hsi_k, 0.8, 0.6, 0.4, falses(S))
    P_T = sparse(P_k')

    releases_k = [1, 1, 1, 4, 4, 9, 9, 9, 9]
    ks_k       = [0, 1, 3, 2, 5, 1, 2, 4, 6]

    naive = Dict{Tuple{Int,Int},Vector{Float64}}()
    for (r, k) in unique(zip(releases_k, ks_k))
        vec_k = zeros(Float64, S); vec_k[r] = 1.0
        for _ in 1:max(0, k)
            vec_k = P_T * vec_k
        end
        naive[(r, k)] = GeoMotion._rel_probs(vec_k)
    end

    grouped = kstep_transition_cache(P_T, releases_k, ks_k)
    @test Set(keys(grouped)) == Set(keys(naive))
    for key in keys(naive)
        @test grouped[key] == naive[key]
    end

    # Snap pruned endpoints
    cents = [(0.0, 0.0), (1.0, 0.0), (9.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    keep  = [true, false, false, true, true, false]
    obs = DataFrame(tagid = ["a", "b", "c"], release = [2, 3, 1], k = [4, 9, 729], recapture = [6, 5, 4])
    n_rel, n_rec = GeoMotion._snap_pruned_endpoints!(obs, keep, cents)
    @test (n_rel, n_rec) == (2, 1)
    @test obs.release == [1, 4, 1]
    @test obs.recapture == [5, 5, 4]

    # Validate mark-recapture indices
    @test validate_mark_recapture_indices([1, 2, 3], [3, 2, 1], 3) === nothing
    @test_throws ArgumentError validate_mark_recapture_indices([0, 2], [2, 0], 3)

    # Exact-k Router
    W_ring = sparse([
        0 1 0 1;
        1 0 1 0;
        0 1 0 1;
        1 0 1 0;
    ])
    P_ring = sparse([
        0.0 0.5 0.0 0.5;
        0.5 0.0 0.5 0.0;
        0.0 0.5 0.0 0.5;
        0.5 0.0 0.5 0.0;
    ])
    p_exact = GeoMotion._exact_k_max_prob_path(P_ring, 1, 3, 2)
    @test p_exact == [1, 2, 3] || p_exact == [1, 4, 3]

    # Agent-based simulation
    W_ag = sparse([
        0 1 0 1 0 0;
        1 0 1 0 0 0;
        0 1 0 0 0 1;
        1 0 0 0 1 0;
        0 0 0 1 0 1;
        0 0 1 0 1 0;
    ])
    hsi_ag = [0.1, 0.2, 0.3, 0.1, 0.5, 1.0]
    T_ag = build_sparse_transition_kernel(W_ag, hsi_ag, 2.0, 0.1, 1.0, nothing)
    df_ag = simulate_agent_trajectories(5, fill(1, 5), 10, sparse(T_ag); seed = 42)
    @test names(df_ag) == ["tagid", "step", "mesh_unit"]
    @test size(df_ag, 1) == 55

    # Forward project agents and space use
    fp_ag = forward_project_agents([1, 2, 3], [2, 4, 6]; n_agents = 10, transition_kernel = sparse(T_ag), seed = 5)
    @test length(unique(fp_ag.tagid)) == 10
    su_ag = forward_space_use(fp_ag, 6)
    @test su_ag.unit_id == collect(1:6)
    @test all(0.0 .<= su_ag.visit_probability .<= 1.0)
end
