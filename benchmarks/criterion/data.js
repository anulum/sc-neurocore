window.BENCHMARK_DATA = {
  "lastUpdate": 1784706368286,
  "repoUrl": "https://github.com/anulum/sc-neurocore",
  "entries": {
    "Rust Criterion Benchmark": [
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "d3d81032609cb9964f81f0c77ab6c69af8cbcfe0",
          "message": "fix(studio): align readiness test fixtures\n\nSeat: 3314012\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-19T01:04:10+02:00",
          "tree_id": "e2882d65dcc0f4ad353334cdc0c78bd1dac7f7ca",
          "url": "https://github.com/anulum/sc-neurocore/commit/d3d81032609cb9964f81f0c77ab6c69af8cbcfe0"
        },
        "date": 1784419679044,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 829450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4004,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 4000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 838820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19886,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4007,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 8743,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 87232,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3581,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4646,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3167,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1169,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 286,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 539600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 69909,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 69956,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 278520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 254,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1458700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 261,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 268,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20801,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2722,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10316,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 67605000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 26900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 28079,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24705,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 247180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93352,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 931170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1378,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13614,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14006000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1888300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 989570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 4000899,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4373200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1561500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5010500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 582530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5030600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93507,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 56006,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1031600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95679,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33815,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6550900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2773300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5582800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2969800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3742200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4206300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4553600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4106700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4181000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3203800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3438700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2341300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2402900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2596000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1735300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2209600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2228300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4485300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1104700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1110900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 742710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11575,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5783600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2555500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3049500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60568000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 385880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2300900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1271900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1429200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1465700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 348170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1615500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 562050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 194710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 290000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 228610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 372480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41251,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35344,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2844200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93368,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46673,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 325270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1293500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 60780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 592120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1480900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 30197,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30438,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 320750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12663,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7368,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 135760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4226700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10640000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70664,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93102,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78166,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 636780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73351,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 141990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73115,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 90791,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 131250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86422,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4231200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3135,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1857300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 219800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2617200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 286560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79333,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 346010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16328000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 4771400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11692000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2789300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6617700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 187990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31885,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83824,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 117020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 543280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32810000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20431000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 350720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 343350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62245,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46107,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 37026,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 95654,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4225700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 62410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 140910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62474,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4459100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14364,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1110100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 345950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1472900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 778680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 183990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 333510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 18003,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20667,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1481900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 735260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 431960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2739700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 455020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 515370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88075,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81779,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "a8518fe1eddfec3e852a31232e7284ace046221b",
          "message": "fix(benchmarks): refresh resonate evidence\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-19T03:26:23+02:00",
          "tree_id": "5261586863ed59496f0e9e3b5a1ce2a0a1e7242d",
          "url": "https://github.com/anulum/sc-neurocore/commit/a8518fe1eddfec3e852a31232e7284ace046221b"
        },
        "date": 1784430324116,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 836320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4006,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3992,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 839210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19809,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4004,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 8737,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 87269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3584,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4676,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3163,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1167,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 283,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 294,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 537880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 69801,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 70071,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 278160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 255,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1452500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 264,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 281,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20746,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2729,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10326,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 66937000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 28600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 26879,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24705,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 246980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 91595,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 931750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1379,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13629,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14031000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1892500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 989680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3995300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4372500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1561900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5012400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 583090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5036900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93514,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 56027,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1032200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 96327,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33964,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 342070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6548500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2770900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5609200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2970300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3695700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4207200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4554600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4092100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4187400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3201000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3442500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2339200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2403200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2591700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1447600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1736300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2209500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 548160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2229100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4470300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1101700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1112400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 742900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11356,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5807700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2556700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3049700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60808000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 389290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39284,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2299900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1270000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1427400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1465100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 348250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1616300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 528980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 562210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 194420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 289630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 228800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 372660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41219,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2843000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93397,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46688,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 325030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1292200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 61008,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 591140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1480600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 30221,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30499,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 320720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12661,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7361,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 135410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4226700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10508000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70842,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93096,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 636750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73557,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 141980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73491,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 90822,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 130449,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4228900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3125,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1859100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2587700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 288980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 78955,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63238,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16331000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 4770400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11684000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2788400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6623300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 187910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 32564,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83872,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 149400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 208810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 544170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32759000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20381000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 350270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 342170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46115,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 37006,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 95376,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4225400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 62381,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61998,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 140220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62449,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 121070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27985,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4429800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14363,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1113400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 345540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1444000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 778330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 181260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 333510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 18008,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20666,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1481700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 735140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 431930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2739400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 455120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 515580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88108,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81760,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Miroslav Šotek",
            "username": "anulum",
            "email": "protoscience@anulum.li"
          },
          "committer": {
            "name": "Miroslav Šotek",
            "username": "anulum",
            "email": "protoscience@anulum.li"
          },
          "id": "a8518fe1eddfec3e852a31232e7284ace046221b",
          "message": "fix(benchmarks): refresh resonate evidence\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-19T01:26:23Z",
          "url": "https://github.com/anulum/sc-neurocore/commit/a8518fe1eddfec3e852a31232e7284ace046221b"
        },
        "date": 1784445215016,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 732090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4419,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 4126,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 733340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 247670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 21361,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4343,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 4127,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 161220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 9513,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 94980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3264,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4544,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3266,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 936,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 313,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 271060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 74720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 74676,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 309150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 286,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1538400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 304,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 19543,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2777,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 11343,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 71093000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 28599,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 27410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 27370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 272940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 105730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 1056300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1372,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13675,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14819000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1885100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 1000400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 4076300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4468400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1616200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5238200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 612670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 382890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5385300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 105340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 63189,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 513390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1165300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 117940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 184780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 37664,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 393590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6685200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2894400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 6096300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 126010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 130610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2973200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3785400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4460700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4805700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4422300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4166699,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3189900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3459700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2600200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2666500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2892900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 1074000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1534300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1940900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2551800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 691120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2533000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4524700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1228400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1230500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 756150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 224340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 196040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11761,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 6078600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2682200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3082600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 64119000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 414800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 43385,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2315900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1362200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1430700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1558100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 362880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1765500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 567030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 608830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 181970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 306540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 218190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 397940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 45393,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 39273,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 579530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 3143300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 500230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 83063,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 52722,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 722320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 357670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1430400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 248310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 68125,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 638250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1647300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 492360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 767540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 34251,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 34258,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 330550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 14036,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 154940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 719050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4750600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 11117000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 77225,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 274590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 140510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 103520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 90775,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 676880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 80117,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 145720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 83206,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 316190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 98285,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 137070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 196600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 91145,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 281140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4672600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3525,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 2104000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 246320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2592500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 225420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79326,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63911,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 394130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16565000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 4434600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 12099000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 702520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2653000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 7092400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 214310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 35205,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 190710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 82832,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 132280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 237490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 131760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 613080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 31180000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 22765000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 435170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 386300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 386300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 77366,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 50825,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 42969,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 101450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4780000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 70414,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 137030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 68992,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 133420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 70261,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 332970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 124640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 31587,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 5062700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 15999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1221100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 339250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1442500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 801440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 189960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 352460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 19944,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 23551,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1746700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 850050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 509940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 3152700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 567390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 564070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 98354,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 109020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 99832,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "d889647141fac0b0853f5d8d31b3d73dc28e38f5",
          "message": "fix(typing): isolate untyped SymPy calls",
          "timestamp": "2026-07-19T15:21:53+02:00",
          "tree_id": "f779b885f35044366a26478a41775838b2241dd5",
          "url": "https://github.com/anulum/sc-neurocore/commit/d889647141fac0b0853f5d8d31b3d73dc28e38f5"
        },
        "date": 1784470898453,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 732420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4416,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 4134,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 735020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 247960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 21359,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4434,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 4152,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 161180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 9873,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 98573,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4541,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3284,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 911,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 322,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 323,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 275540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 74772,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 74601,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 310350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1551300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 303,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 308,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20859,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2783,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 11354,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 71802000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 29231,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 28047,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 27383,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 273060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 105790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 1056200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1374,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13661,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14810000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1882600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 1004799,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 4062299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4481600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1615500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5244400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 611590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 378890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5386800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 105410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 63208,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 513450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1166200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 117840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 184900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 37490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 393430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6677000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2896700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 6091600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 126050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 131190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2985100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3790100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4446100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4805600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4430400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4175900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3189400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3449700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2601800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2665700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2896100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 1074800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1533300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1940100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2552300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 690150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2537500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4533400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1239900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1233800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 762160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 224520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 196430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 6073000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2684200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3079200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 64480000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 413030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 43384,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2322500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1359800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1444500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1534700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 361630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1773100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 567290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 621710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 181010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 307480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 218160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 398530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 45400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 39213,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 579340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 3139600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 499870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 83101,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 52713,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 722330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 357460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1432500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 248320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 68079,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 638390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1641200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 491130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 769100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 34245,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 34286,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 331940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 14039,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7931,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 155930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 718950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4742600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 11061000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 77231,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 274600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 140450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 101720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 90608,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 692940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 79844,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 146030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 86542,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 316140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 98448,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 33262,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 196710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 91235,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 281500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4676500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 2110600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 246430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2622400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 224930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79239,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 67640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 395340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16535000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 4397800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 12035000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 702860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2652800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 7091800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 214370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 35211,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 190620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 82855,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 131930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 237670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 131760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 612180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 31304000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 22732000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 434180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 386400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 385470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 77085,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 50841,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 42978,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 73888,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4824900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 35298,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 137020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 69017,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 132360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 70368,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 333020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 124050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 31587,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 5086100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 16018,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1295000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 339360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1441600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 802120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 190490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 353660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 19934,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 23539,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1750200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 849860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 509420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 3153200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 567870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 564300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 98321,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 108870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 99763,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Anulum Fortis",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Anulum Fortis",
            "username": "anulum"
          },
          "distinct": true,
          "id": "394892dbe05293e667d49da178a61b5a1082eb4c",
          "message": "chore(studio): sync format and generated docs for Phase0 pre-commit\n\nruff-format the analysis jobs test; regenerate capability_manifest surfaces\nand API_REFERENCE so complete pre-commit on the successor path set is clean.\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-19T16:34:35+02:00",
          "tree_id": "9c32c369f2cec27808c3087cec0bf1e38621a11d",
          "url": "https://github.com/anulum/sc-neurocore/commit/394892dbe05293e667d49da178a61b5a1082eb4c"
        },
        "date": 1784475513764,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 831850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4016,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3992,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 837930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19956,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4009,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10485,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 104560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1214,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 283,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 286,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 541890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 68309,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 67788,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 277590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 252,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1459300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 261,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 19350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2366,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10358,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 67037000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 26640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 27921,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24794,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 248080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93711,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 937150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1348,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13622,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 13997000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1882900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 998430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3974500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4545100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1563600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5122600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 583490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5070400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93484,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 55975,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1032000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95654,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33355,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 342300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6549500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2767300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5810200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2966100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3700100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4291700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4651100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4263000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4245200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3217200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3595700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2339400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2408900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2598600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1448200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1735200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2208200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2232400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4736500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1100500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1110300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 744570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 203000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5797900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2588500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3071900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60657000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39306,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2314700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1273200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1507700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1473100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 348880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1621700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 200720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 290150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 233200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 373640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41544,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35388,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520929,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2843000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93364,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46704,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1291100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 60672,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 597360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1487800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30893,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12657,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7383,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 140860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4225900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10602000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70211,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93054,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78351,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 637670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73645,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 141920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73732,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91155,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29613,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86562,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4228900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1855800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2620700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 285670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 80124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 64691,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16989000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 4747200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11657000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 618390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2937600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6587500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31167,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83659,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 33052999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20286000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 358430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 313860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62223,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46164,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 36982,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74488,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4170499,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 32091,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 62097,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 138860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27995,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4436900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14418,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1135400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 346310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1440200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 776620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 346120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 18004,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20668,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1489200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 734760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 438080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2735900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 455360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81795,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Anulum Fortis",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "3571a6e9c89f7dbbe4933ec877912e0847e35d50",
          "message": "feat(studio): imperative async analysis job runner\n\nW12-C runner builds a fail-closed analysis-job request, runs the W07/W08\nsession submit+poll path to a terminal phase, sinks completed results via\nW12-B, and always disposes. Invalid requests never start a session. Not\nwired into store/App yet (W12-D+).\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-19T23:51:02+02:00",
          "tree_id": "d5f3752374129d8b7c7b98dc55a725ddb87bea7c",
          "url": "https://github.com/anulum/sc-neurocore/commit/3571a6e9c89f7dbbe4933ec877912e0847e35d50"
        },
        "date": 1784501648235,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 828810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4003,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3976,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 841960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 21975,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4027,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3979,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 104430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3572,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4667,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3286,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1214,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 293,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 549650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 70087,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 70840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 278670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 258,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1458100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 274,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20663,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2358,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 66760999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 26966,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 28634,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24698,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 248230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93661,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 936780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1348,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13629,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14005000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1886400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 993540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 4231700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4418900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1561800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5016300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 583120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 395060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5000500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93479,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 56002,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1031300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95687,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 189040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33382,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6534900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2765600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5592000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 139440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2949500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3687200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4252000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4562800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4089100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4188799,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3207500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3453700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2339000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2403300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2596600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1445900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1735900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2209500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2227500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4473900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1098300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1108300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 741550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11194,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5789300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2553400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3068800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60633000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39331,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2298600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1276400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1435800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1465900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 346910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1615700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 194690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 291630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 229220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 373950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41436,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35268,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2843300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93358,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46673,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1292200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 60694,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 592820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1482800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 676520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29814,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30766,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12653,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7318,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 648190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4226300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10779000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 239740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93129,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78418,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 639730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73747,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 141970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73676,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91157,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29615,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 249530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4228900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3126,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1855600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2609800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 287370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63409,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16477000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3924700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11509000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 620320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2825500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6589000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31392,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83704,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 210460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32766000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20324000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 353150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 312250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62342,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46169,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 37116,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74345,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4171299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 57562,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61983,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 139230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62444,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4472600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14421,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1119500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 347550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1446400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 780820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 181360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 337370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20667,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1481700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 734550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 437490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2740600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 454970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 516780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88105,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 104970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81791,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "083b8ae61e5410634a3efb6a9028018b639ee80b",
          "message": "feat(neurons): close alpha fidelity",
          "timestamp": "2026-07-20T01:37:38+02:00",
          "tree_id": "e93110d0a0d4d0abf9aca20646aab9fa66d25047",
          "url": "https://github.com/anulum/sc-neurocore/commit/083b8ae61e5410634a3efb6a9028018b639ee80b"
        },
        "date": 1784508135374,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 831480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4007,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3978,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 839000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19708,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4024,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3984,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10484,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 104100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3554,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4701,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1178,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 286,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 545030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 68191,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 67890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 277630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 252,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1447700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 268,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 19390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2327,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10164,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 67734000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 27425,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 27671,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24748,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 247760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 91722,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 917100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1355,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13761,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 13989000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1887200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 994490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3981800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4383400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1560600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5014600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 582590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 399760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4993100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93778,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 56170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1040400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 197480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33041,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6528600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2765100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5583000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2950100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3755700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4279000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4590300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4130899,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4243800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3204600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3498000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2340600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2411600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2605400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1447900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1738700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2209900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 548340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2245200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4465400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1097900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1113100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 746540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11358,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5756500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2555500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3043600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60603000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39303,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2296800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1275500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1437300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1462700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 349980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1620300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 196790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 289710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 228730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 374020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41501,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 34397,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 521789,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2844000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93363,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46681,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 646880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1293300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 471920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 599130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1484600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29905,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30749,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12656,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7407,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4224200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10712000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70356,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 239000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93074,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78528,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 637720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 74189,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 142020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 281030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91188,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29775,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86613,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4256300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3123,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1856300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2588500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 286710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79815,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63619,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16760000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3979500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11548000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 621650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2834600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6599900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31141,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83855,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32868000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20315000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 352180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 341980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 313160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62262,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 36982,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74615,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4169500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 45321,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 62015,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 139850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62559,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 121450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27981,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4442100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14417,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1118700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 347310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1441100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 778150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 338550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17998,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20662,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1485300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 735100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 437400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2738800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 456260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88129,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81746,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "84262d73014fe549f66fdb3182fac40603923f6f",
          "message": "fix(ci): restore benchmark evidence and alpha throughput\n\nRegenerate affected benchmark evidence from repository-relative commands and ratchet generator provenance. Keep the Alpha Julia facade module-owned, refresh formal inventory documentation, and reuse exact-flow coefficients in the Rust hot path.\n\nSeat: user/terminal-3314012\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-20T04:33:38+02:00",
          "tree_id": "530e5da17f739dd401b1ed981adce5bfbe80e58b",
          "url": "https://github.com/anulum/sc-neurocore/commit/84262d73014fe549f66fdb3182fac40603923f6f"
        },
        "date": 1784521095520,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 830200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4003,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3988,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 842590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19047,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4004,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 150300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 8423,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 84138,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3606,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4642,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3249,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1194,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 283,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 537700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 67438,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 67687,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 276970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 254,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1438100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 262,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 262,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 19383,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10098,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 67029000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 26437,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 27437,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24711,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 247170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93073,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 916770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1364,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13432,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14036000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1891000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 989110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3986300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4372400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1559700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5031700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 579000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 395450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4983900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93494,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 55976,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 444730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1029099,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95649,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33702,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6557200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2761100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5568600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2968900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3701000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4208900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4562500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4094999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4160100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3214000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3443000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2339300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2403100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2597800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1735700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2207700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2227400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4489200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1100800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1109900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 753180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 12978,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5829500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2546300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3057200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60730000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 387610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2299100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1288100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1429700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1460500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 348120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1615700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 196260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 292620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 228540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 372540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41502,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 34971,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520679,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2836300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93371,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46674,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1295300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49806,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 589880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1482900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 674790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29866,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30524,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 322540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12649,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7344,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 135130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4247700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 9812700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 71123,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93074,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 80927,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 638280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73068,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 131720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 74227,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 90924,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29635,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86551,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4228300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1857100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2582900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 285930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 82571,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63113,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 340540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16357000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3922700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11553000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2817300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6588300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 187780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31165,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 149240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 541900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32703000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20462000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 353350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 342090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62365,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46107,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 36986,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 73637,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4164700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 50866,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61971,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 139940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62425,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27983,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4460500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14321,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1118100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 346020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1453800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 778180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 335260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 18006,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20656,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1481100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 734700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 446480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2770200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 454840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88075,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 104960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81737,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Miroslav Šotek",
            "username": "anulum",
            "email": "protoscience@anulum.li"
          },
          "committer": {
            "name": "Miroslav Šotek",
            "username": "anulum",
            "email": "protoscience@anulum.li"
          },
          "id": "84262d73014fe549f66fdb3182fac40603923f6f",
          "message": "fix(ci): restore benchmark evidence and alpha throughput\n\nRegenerate affected benchmark evidence from repository-relative commands and ratchet generator provenance. Keep the Alpha Julia facade module-owned, refresh formal inventory documentation, and reuse exact-flow coefficients in the Rust hot path.\n\nSeat: user/terminal-3314012\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-20T02:33:38Z",
          "url": "https://github.com/anulum/sc-neurocore/commit/84262d73014fe549f66fdb3182fac40603923f6f"
        },
        "date": 1784532337946,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 548940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4307,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 2064,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 547180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 217290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 13707,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4399,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 2010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 137050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 5940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 60633,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 2539,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 3183,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 2398,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 774,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 148,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 379500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 35838,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 37616,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 112150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 142,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 713220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 139,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 11728,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 1621,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 7227,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 47600000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 20970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 20404,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 20324,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 206380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 81645,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 824560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 11553,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 9737700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1290800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 663460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 2626700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 2936700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1118900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 3601800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 443650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 261130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 3709800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 65111,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 39561,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 383570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 836930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 90493,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 136600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33716,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 309790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 4509000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2041000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 4394200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 95840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 99125,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2050800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 2620300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 2932300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 3128100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 2897800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 2752300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 2170400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 2397700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2020300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2082900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2207700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 911820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1287400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1582400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 1999700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 558890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 1913300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 2953600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 859910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 879330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 514429,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 157080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 142640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 7137,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 4301700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 1880300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 2141400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 43685000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 283490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 29863,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 1578700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 957350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1048600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 990470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 236150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1148100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 372620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 402850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 132790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 196160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 153060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 267330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 29843,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 27139,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 424020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2135900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 285700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 71980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 36416,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 478680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 221420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1030999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 169510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 45288,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 422230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1126500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 367010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 558300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 24434,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 23173,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 234860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 10001,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 5500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 103510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 453010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 3909100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 8703000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 51804,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 185930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 93757,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 67830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 71165,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 426090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 54137,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 124300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 59571,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 233940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 71436,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 23156,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 69267,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 70282,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 224570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 3342800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 2512,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1832400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 127840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 1627500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 187100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 50710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 46027,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 259110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 11067000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 2448800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 6938100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 503220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 1843100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 5001500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 148190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 25404,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 140360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 61146,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 84296,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 165780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 73583,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 496670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 21018000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 16184999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 388630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 272970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 270190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 66151,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 30359,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 21015,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 47194,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 3318800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 24351,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 100770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 47001,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 89758,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 46070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 230870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 73367,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 22640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 2977400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 10034,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 578220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 223110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 937030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 501450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 179730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 243760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 13728,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 16803,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1357000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 569540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 286850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2171300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 507530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 327640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 70373,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 92183,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 82564,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "e693a2f1f15e77ddc478b1a1e593542b3459ccc3",
          "message": "fix(ci): collect native parity inventory\n\nSeat: user/terminal-3314012\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-20T12:03:26+02:00",
          "tree_id": "b88d39e82084614322fc10fbb6d4e332d5d60cdb",
          "url": "https://github.com/anulum/sc-neurocore/commit/e693a2f1f15e77ddc478b1a1e593542b3459ccc3"
        },
        "date": 1784545486228,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 730550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4417,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 4124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 729170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 250400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 21066,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4527,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 4124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 161160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 9508,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 95055,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3292,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4419,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 901,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 317,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 272210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 74005,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 74065,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 309300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 286,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1553600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 294,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 304,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 19555,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2796,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 11350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 72391000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 28657,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 28525,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 27250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 272000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 105410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 1055200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1371,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13667,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14803000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1878000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 1001200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 4057600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4467500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1620000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5254500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 609930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 377510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5379300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 105400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 63196,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 515110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1164700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 118190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 186040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 37518,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 393380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6676900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2914600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 6085000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 126000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 130669,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2988000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3795900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4439500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4776100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4398300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4155400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3184400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3445800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2602200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2651800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2893700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 1074600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1531700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1941100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2556300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 690340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2522100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4518300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1236100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1230700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 756380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 224360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 196040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 12328,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 6070500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2680800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3079200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 64504000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 412860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 43424,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2317000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1361300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1430100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1545000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 361040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1765200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 567140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 610280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 180990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 306920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 217800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 398190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 45381,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 39209,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 580480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 3134400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 499920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 83287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 52690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 722410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 357310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1434100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 248330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 48486,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 639660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1643900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 491240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 768260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 34397,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 34257,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 330020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 14034,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 8001,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 154830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 719030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4743600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 11062000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 77135,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 274650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 140530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 101710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 90680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 675810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 79399,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 136790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 84089,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 316210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 98186,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 33329,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 196730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 91187,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 281170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4674700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3529,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 2105300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 247610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2614100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 225280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79488,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 67835,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 392830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16533999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3619900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11974000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 702740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2637600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 7094400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 214220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 35191,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 190650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 82865,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 132390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 237470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 131860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 611910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 31267000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 22686000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 444500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 386480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 386420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 70810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 50817,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 42988,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74541,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4758800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 35219,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 136210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 69025,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 131670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 70233,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 333180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 123990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 31588,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4991600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 15986,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1221400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 339220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1441000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 803400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 190760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 352400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 19937,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 23587,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1742100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 849750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 510050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 3162100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 567780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 564410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 98352,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 108890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 99785,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "7a5b207ad2b99f3323ec692eeea11ade7bcbd5be",
          "message": "refactor(engine): extract Phi-star binding\n\nSeat: user/terminal-3314012\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-20T16:15:55+02:00",
          "tree_id": "6854d710a56eaa670d3b6af7320e641f19370e81",
          "url": "https://github.com/anulum/sc-neurocore/commit/7a5b207ad2b99f3323ec692eeea11ade7bcbd5be"
        },
        "date": 1784560521847,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 827300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4013,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3992,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 841550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 21293,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4011,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10401,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 104020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4673,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3272,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1242,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 292,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 288,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 547680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 69860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 69951,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 278450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 252,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1444400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 262,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20819,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2366,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10362,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 66230999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 27765,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 27771,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 247730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 92065,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 921790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13633,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14058000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1895400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 990680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3975500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4375700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1577300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5022000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 584990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 395610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4968600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93537,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 56008,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 443040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1032999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95718,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33367,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 342040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6550800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2769300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5573600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2971300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3686800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4229500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4540400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4101600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4162899,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3211900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3452600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2339800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2407900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2595900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1449200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1736400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2209200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2236200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4469200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1101500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1111200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 743140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11211,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5787800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2544200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3078600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 61149000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39304,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2296300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1295700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1441300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1467000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 348690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1617100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 527060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 194750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 290780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 229410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 374760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41448,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35422,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2868900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93371,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46686,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1293400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49794,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 591190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1485000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29802,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30764,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12683,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7329,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 645910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4223400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10601000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70229,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93086,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78387,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 640740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 142060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73199,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91183,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29637,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86571,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4239300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3126,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1856000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2596100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 287030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79491,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 64034,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16527999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3943400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11513000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2804300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6589400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31154,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83666,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32904000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20429000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 353210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 315520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62265,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46158,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 36982,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74365,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4167500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 59623,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 62011,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 138910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62614,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27997,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4545900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14397,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1113100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 345670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1446100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 791450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 337630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17996,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20661,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1481300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 735190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 438130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2771800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 455080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88094,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81764,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "8fd62241ab03f3bd4ece907a0d9cdc52b99e4e70",
          "message": "fix(ci): restore package and evidence contracts\n\nSeat: user/terminal-3314012\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)",
          "timestamp": "2026-07-20T18:32:34+02:00",
          "tree_id": "62ccdf289595f1851fd7db4c6865cbe5f4065778",
          "url": "https://github.com/anulum/sc-neurocore/commit/8fd62241ab03f3bd4ece907a0d9cdc52b99e4e70"
        },
        "date": 1784568743785,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 828040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4006,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3998,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 840290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19745,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4008,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3979,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10434,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 105400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3542,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4749,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3302,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1212,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 291,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 549000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 70269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 70320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 277800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 252,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1433100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 268,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 276,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20321,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2363,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10376,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 66561000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 26701,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 27093,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24692,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 246940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93477,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 917090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1349,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13623,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14121000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1895900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 992280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3976600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4374900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1572000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5031200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 584430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5055100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93505,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 55977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1033099,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95698,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33336,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6570400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2775600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5585800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2958200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3682800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4256300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4583200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4101000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4179000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3225500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3457900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2338800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2406400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2599600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1738200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2208000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2227100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4486700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1101100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1111100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 743530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11226,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5786300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2560900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3067600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60836000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 387310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39305,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2306200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1285700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1436400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1465300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 347090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1616600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 194880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 292240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 229660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 374200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41435,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35319,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2844200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93411,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46774,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1290600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49763,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 591450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1484400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29828,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30715,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12659,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7337,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4224000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10604000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 239000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93075,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78387,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 637860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73201,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 142040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73344,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91149,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29628,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86591,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 249170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4236000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3126,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1857800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2593100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 288580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79604,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63565,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16475000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3925900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11549000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 618660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2805000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6592500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31209,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83761,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 543280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32860999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20341000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 353210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 311330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62244,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46162,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 36977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74317,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4245700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 33161,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61997,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 138830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62442,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 121070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4443300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14396,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1113300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 347630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1447300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 777930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 338100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17993,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20658,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1483300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 735630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 437450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2739100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 454920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81748,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": false,
          "id": "d49729d76210e922531b02b11e6d79c5c9ea422d",
          "message": "fix(optimizer): sort and deduplicate the Rust-path Pareto frontier\n\nThe full CI matrix (SC-NeuroCore CI, \"Test + coverage\") failed on origin/main\n8fd62241a across Python 3.10-3.14 with a single deterministic failure,\ntest_pareto_points_sorted_by_luts. SCOptimizer.optimize_annealing exposes a\nPareto frontier whose LUT-ascending ordering and (luts, power, score)\ndeduplication is guaranteed by the pure-Python fallback _extract_pareto, but the\nRust-accelerated path built the frontier straight from py_opt_extract_pareto\nwithout applying that normalisation. When the engine is present (as in CI) the\nRust path runs, so the reported frontier was unsorted and could carry duplicates.\n\nBoth backends now funnel their non-dominated points through a shared\n_sort_and_dedupe_frontier normaliser, restoring Python/Rust parity on the\nfrontier contract. A focused regression test drives the Rust dispatch with an\nunsorted, duplicate-bearing fake result and asserts the report is LUT-sorted and\ndeduplicated, so the contract cannot silently regress again.\n\nPython-only change: no Rust rebuild, no benchmark evidence, descriptor, or\ncapability-manifest impact. Verified locally: the previously-failing test plus\nthe full tests/test_optimizer suite (95 passed), ruff, ruff format, strict mypy,\nand exact-path pre-commit. This workstation cannot run the full suite; the\nfull-matrix CI is the authority.\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\nSeat: 14753",
          "timestamp": "2026-07-21T04:31:40+02:00",
          "tree_id": "cec33663d6f02d4675231b64ac1eb446498cbda7",
          "url": "https://github.com/anulum/sc-neurocore/commit/d49729d76210e922531b02b11e6d79c5c9ea422d"
        },
        "date": 1784610529081,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 830940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 3997,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3989,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 839270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 224110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 22610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3981,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10428,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 103010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4695,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1196,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 288,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 542550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 67618,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 67648,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 277320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 252,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1446300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 261,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 19565,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2364,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10359,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 66593000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 27200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 28179,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24854,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 248730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93632,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 919080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1347,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13615,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14046000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1889400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 991400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3982000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4374300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1569400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5029300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 584570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4967300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93582,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 55966,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1031600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95661,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33455,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6562700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2769800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5574200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2975000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3687100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4215600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4547600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4098900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4164900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3218800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3460300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2338800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2405600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2595400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1736200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2208900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2232100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4465700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1100000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1111400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 742890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11212,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5784600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2557900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3083600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60833000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39303,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2301300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1288700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1434700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1461100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 347530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1622200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 196580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 290540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 229190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 375810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35498,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 521610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2852200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93381,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46673,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 646960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1292200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49747,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 590380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1485500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29819,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12656,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7457,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4222200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10728000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70215,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93142,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78379,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 638110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 141990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73555,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91135,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29612,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86547,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4229800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1855900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2606500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 285820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79212,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63539,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16492000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3929300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11517000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2837000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6590300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 32153,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32912999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20397000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 352140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 341980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 314350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46165,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 37013,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74434,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4168900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 31426,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 122080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 62006,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 139240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62425,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 121880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27981,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4435400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14411,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1114900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 345990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1438700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 776560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 337950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20686,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1483200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 734990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 437460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2746400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 454880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88105,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81880,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Miroslav Šotek",
            "username": "anulum",
            "email": "protoscience@anulum.li"
          },
          "committer": {
            "name": "Miroslav Šotek",
            "username": "anulum",
            "email": "protoscience@anulum.li"
          },
          "id": "beddf040cb466a63fd516c3d07bf2d2bcca0ea05",
          "message": "test(studio): update guided-operator-run mocks for the async analysis-job flow\n\nW12-D (thin-store refactor) moved the guided run's f-I analysis off the legacy\nsync /api/fi-curve endpoint onto the async analysis-job runner\n(runStudioAnalysisJob -> POST /api/analysis/jobs -> poll status_route). The\npre-existing e2e guided-operator-run.spec.ts still mocked only /api/fi-curve, so\nthe job never resolved: fiResult stayed null, analysisComplete stayed false, and\n\"Run next guided step\" was stuck on \"Run f-I analysis\" instead of advancing to\n\"Skip training\" (the frontend CI failure on the W12 chain).\n\nMock the real async flow the same way analysis-job-host.spec.ts does -- the\nPOST /api/analysis/jobs receipt plus a poll sequence (running -> completed with\nthe fi_curve result) -- and assert one /api/analysis/jobs submission instead of\nthe retired /api/fi-curve call. Test-only fix-forward for the W12-D contract; no\nproduct change. Lander fix on grok-sol's studio chain; grok-sol second-eye\nrequested.\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\nSeat: 14753",
          "timestamp": "2026-07-21T06:39:30Z",
          "url": "https://github.com/anulum/sc-neurocore/commit/beddf040cb466a63fd516c3d07bf2d2bcca0ea05"
        },
        "date": 1784618000140,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 703790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 5516,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 2751,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 704970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 292840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19932,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 5507,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 2755,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 187650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 11431,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 113550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4223,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3257,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1079,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 352,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 207,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 544000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 46045,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 46059,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 146670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 177,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 868400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 349,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 187,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 15053,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 1866,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 9182,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 59682000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 25046,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 25605,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 28770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 287530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 117310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 1175100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1604,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 15910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 13933000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1775800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 948480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3774200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4181000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1512500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5097900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 614950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 362420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5136300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 98277,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 61079,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 546260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1134100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 134440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 46446,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 435950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6289900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2747200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 6068400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 136790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 136990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2887200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3621100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4131900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4355000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4013000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 3851000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3088700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3446000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2890600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2941800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 3109200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 1294800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1837100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 2301100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2848600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 845010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2719800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4117600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1210000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1217300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 718380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 218500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 201300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 7853,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5863100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2560000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3053700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60467000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 382320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 41717,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2177600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1265700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1450500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1358800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 328620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1569100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 532770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 570610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 178830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 274510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 214390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 375400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 42199,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 38593,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 577860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2996100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 420530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 102700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 55685,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 641710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 333410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1412400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 237550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 57740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 596810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1533500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 515679,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 770360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 33830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 32936,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 324940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 16891,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7341,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 143710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 650540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 5308100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 11767000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 74589,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 268050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 137720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 94556,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 95198,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 574980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 101700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 187250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 80588,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 373810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 100900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 32634,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 87651,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 95609,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 278870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4747800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3381,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 2409700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 171690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2250700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 262940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 69607,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 62191,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 347440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 15326000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3210100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 9598000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 721550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2500400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6685200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 234870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 33783,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 213760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 84860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 115950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 235990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 103760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 647980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 28382000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 21612000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 550720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 363250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 363070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 81274,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 41800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 29472,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 83186,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4619200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 45816,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 143530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 65105,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 126060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 63928,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 318410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 100180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 37351,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4197500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14092,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 798290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 308610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1275300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 694490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 268130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 344340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 20101,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 26607,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1881800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 705950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 419870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 3137900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 756690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 452400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 101440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 125920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 111600,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "ac7ef879518080f1463fae688629f29a4b2a0e72",
          "message": "refactor(engine): extract Cazelles map binding\n\nSeat: 3314012",
          "timestamp": "2026-07-21T20:18:02+02:00",
          "tree_id": "98fdf0b8ad5e7145f2c72fc3fecee7f23b1762d0",
          "url": "https://github.com/anulum/sc-neurocore/commit/ac7ef879518080f1463fae688629f29a4b2a0e72"
        },
        "date": 1784661774596,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 568610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 3428,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 1122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 567810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 192070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 14314,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 3464,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 1195,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 125240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 8199,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 81974,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 2577,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 3533,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 2541,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 726,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 261,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 209380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 53265,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 55259,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 234320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 231,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 976030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 248,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 247,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 13403,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2154,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 8839,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 55647000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 23002,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 22106,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 21216,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 212330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 81888,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 819100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1065,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 10601,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 11481000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1468900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 775960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3152400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 3474200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1252100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 4070200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 474250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 293740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4191799,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 81734,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 49025,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 398300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 905130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 91250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 144080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 29109,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 305190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 5198700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2260000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 4743900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 97772,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 101220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2305100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 2931500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 3444100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 3706400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 3414200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 3222900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 2476600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 2676200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2018200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2064099,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2244900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 834770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1190200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1506100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 1984000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 535580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 1963200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 3507700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 953590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 954950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 590200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 173970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 152060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 9119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 4734100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2081100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 2409600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 49878000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 320050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 33761,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 1796400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1055700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1108100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1191300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 280010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1369800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 439960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 487620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 140240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 238640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 169290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 308540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 35346,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 30309,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 449370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2431900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 388260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 64447,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 41013,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 560120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 277180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1110200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 192870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 37599,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 498410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1274400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 381100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 595440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 26573,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 26676,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 256680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 10892,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 6189,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 120260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 603770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 3676900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 8583500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 59869,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 213530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 108990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 80744,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 70347,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 527650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 62057,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 113000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 63965,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 245320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 76355,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 25795,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 152770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 70802,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 218110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 3628300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 2738,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1634300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 191970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2034100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 174850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 61513,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 52275,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 302210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 12814000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3037200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 9380100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 546210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2105300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 5512900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 166300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 27314,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 147780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 64279,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 102950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 184180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 102500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 475020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 24500000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 17626000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 336880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 299760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 295520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 60001,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 39422,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 33433,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 57345,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 3731600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 27325,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 106310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 53551,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 102140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 54647,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 258529,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 96027,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 24505,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 3970600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 12415,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 969240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 264380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1123500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 623680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 151330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 273120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 15465,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 18281,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1358100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 659480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 395230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2446300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 440260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 437770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 76295,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 84479,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 77582,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "11f20c3301c904a28ea6ee9f9f2dcf5d9b77a539",
          "message": "refactor(engine): extract Wilson-HR binding\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\n\nSeat: 3314012",
          "timestamp": "2026-07-21T23:25:33+02:00",
          "tree_id": "1439e1bb9c08f4a76d0630de6eb4e8833423c444",
          "url": "https://github.com/anulum/sc-neurocore/commit/11f20c3301c904a28ea6ee9f9f2dcf5d9b77a539"
        },
        "date": 1784673017578,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 831910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4004,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3987,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 842710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19522,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4014,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3983,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10433,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 104420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3578,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4683,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1212,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 545570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 66718,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 67348,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 277680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 262,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1434100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 273,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 275,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20081,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2356,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10367,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 66440000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 27100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 26333,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24744,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 247700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93736,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 939580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13622,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14099000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1896900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 990140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3972300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4374300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1560300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5068700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 583920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4993000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93909,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 56386,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1032000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95698,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33353,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 343250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6559500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2770800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5611700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 139170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2954800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3685900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4213400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4538700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4114499,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4164300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3215400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3455700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2339300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2409500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2596000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 965520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1739300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2213900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2241300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4470900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1102200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1118300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 747470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11214,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5788100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2546400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3078200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60843000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2313300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1279500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1436500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1462600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 347460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1625200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 527720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 569650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 195050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 290560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 230230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 378610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41799,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35399,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 521000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2843400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 448870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93426,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46683,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1294500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49793,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 590470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1484100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 456510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29874,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30766,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 322250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12666,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 137640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 674430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4231700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10684000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70454,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93093,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78598,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 642430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73297,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 142210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73415,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91167,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29626,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86846,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 249850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4231500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1856900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2595500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 287510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16492000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 4147500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11607000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 618530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2833000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6597400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 32436,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83669,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 151180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 116870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 543110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32898000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20442000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 351660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 313060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46182,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 37002,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 75170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4204600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 33806,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 62033,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 140450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27983,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4473700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14412,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1125700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 345640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1465200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 781630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 339000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20664,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1485200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 735950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 437960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2763700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 455040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81990,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "172bc5a8e2cf8867f09570553b5165ff085bb462",
          "message": "refactor(engine): extract COBA-LIF binding\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\n\nSeat: 3314012",
          "timestamp": "2026-07-22T00:37:50+02:00",
          "tree_id": "635550bdca03b8516653db13fee560b7fcd35d16",
          "url": "https://github.com/anulum/sc-neurocore/commit/172bc5a8e2cf8867f09570553b5165ff085bb462"
        },
        "date": 1784678516092,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 803990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4418,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 4127,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 728820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 247570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 21267,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4454,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 4124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 161200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10569,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 105830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3299,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4556,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3285,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 921,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 321,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 333,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 272250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 71950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 72087,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 311540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 294,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1548700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 312,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 18899,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2779,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 11425,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 69747000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 27552,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 27552,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 27194,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 272770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 105490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 1055100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1372,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13667,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14806000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1880900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 1002200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 4067400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4471600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1616400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5241300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 611680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 377640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 5393000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 105420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 63214,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 513559,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1164800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 117680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 184850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 37520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 393460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6679100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2895300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 6093300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 126030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 130650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2978300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3782600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4436400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4779700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4389500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4155100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3188600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3458000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2602100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2660100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2894100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 1074700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1533800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1941900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2551800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 691100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2532200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4524700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1227500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1231200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 756720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 224440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 196140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11758,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 6076900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2682900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3076900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 64337000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 412650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 43476,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2311300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1363200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1430300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1541500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 361390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1764000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 567390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 609210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 180830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 307440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 218140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 398710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 45429,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 39188,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 579360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 3136100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 499920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 83109,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 52693,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 722330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 357380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1431300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 248310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 48449,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 640130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1642800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 491330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 767680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 34250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 34294,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 330210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 14041,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 8003,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 154870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 724180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4741500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 11069000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 76988,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 274680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 140490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 104030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 90621,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 680820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 80245,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 145740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 83560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 316270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 98429,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 33223,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 196640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 91175,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 281250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4674700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 2105000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 246330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2524200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 225080,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79465,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 67417,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 393280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16704999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3630500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 12063000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 702770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2642300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 7094900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 214450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 35181,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 190600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 82903,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 131970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 237520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 131790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 612210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 31300000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 22743000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 433530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 386390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 378660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 77394,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 50851,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 42984,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 73949,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4770000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 35248,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 136230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 69030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 131710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 70332,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 333100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 123950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 31585,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 5312200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 16001,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1263600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 340360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1439700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 801710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 190050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 353560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 19937,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 23547,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1745400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 850250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 510020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 3155900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 567580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 564370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 98523,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 108900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 99805,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "4a7891fc66b6718eef86495284426b23e3d9fd47",
          "message": "refactor(engine): extract EscapeRate binding\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\n\nSeat: 3314012",
          "timestamp": "2026-07-22T01:34:43+02:00",
          "tree_id": "5215682527e6ab797badcc512b0ecb7200ec5fa5",
          "url": "https://github.com/anulum/sc-neurocore/commit/4a7891fc66b6718eef86495284426b23e3d9fd47"
        },
        "date": 1784682175052,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 830000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4019,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3994,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 837270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19417,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4008,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3978,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10376,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 104110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3576,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4715,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3322,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1236,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 283,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 289,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 545490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 66218,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 66081,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 276810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 252,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1432700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 263,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 19798,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2365,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 66833000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 25477,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 26244,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24739,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 247390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 937260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1347,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13625,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14030000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1892900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 989140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3991900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4373300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1562700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5020800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 584570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4964200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 55995,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1032300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95709,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33362,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6539200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2763500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5575300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2951600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3707600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4212400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4533300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4108500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4162600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3216300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3453200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2340300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2406800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2594900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1736100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2207800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2230800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4472500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1105500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1114200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 743450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11206,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5787200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2553600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3075700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60651000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 387960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39327,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2296400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1278300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1438400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1459100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 348860,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1618300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 197060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 289940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 230110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 372630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41469,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35372,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2843700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93394,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46701,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1290000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49855,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 590110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1484200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 674850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29813,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30694,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321060,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12655,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7375,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 646010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4225500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10953000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93086,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78394,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 637690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73632,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 142010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73688,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91174,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29622,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86572,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248880,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4228600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3124,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1854800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218160,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2617100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 287070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79542,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63790,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16495000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3945600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11543000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2828700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6619200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 32892,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83674,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 152120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 127630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32838000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20437000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 351610,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 314170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62259,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46161,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 36994,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74629,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4168900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 32149,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61998,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 139050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62471,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27982,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4439100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14406,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1114000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 347350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1439000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 776480,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 338500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17998,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20676,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1481400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 734960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 437470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2740600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 454950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88112,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81755,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "912e51e7e7416b4c62788627d18bf16644208c44",
          "message": "refactor(engine): extract IQIF binding\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\n\nSeat: 3314012",
          "timestamp": "2026-07-22T03:23:45+02:00",
          "tree_id": "936f56b244e02a9a7afcc2b767c3a550a133478b",
          "url": "https://github.com/anulum/sc-neurocore/commit/912e51e7e7416b4c62788627d18bf16644208c44"
        },
        "date": 1784687992944,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 829990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4007,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3991,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 838830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 222600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 20993,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4004,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3979,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10459,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 103900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3555,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4704,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3284,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1223,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 283,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 294,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 543280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 66728,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 66732,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 277200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 258,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1428700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 261,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 18578,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2368,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10363,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 65897999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 25756,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 26705,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24697,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 246900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 91752,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 936340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1347,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13622,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14037000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1915400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 989150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3968900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4370500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1560000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5023500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 582820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4976900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93768,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 56009,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1031700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95644,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 191010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33348,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6542400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2764100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5577200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 100730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2950400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3689100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4217500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4547900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4100500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4164399,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3208700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3462100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2338500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2402400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2595500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 963150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1735200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2208600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2227400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4466600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1101600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1110600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 743050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5788400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2546200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3067000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60764000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39265,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2297800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1279100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1438500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1461500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 347010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1616500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 195580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 291090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 229400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 373650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41416,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35342,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 520429,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2843500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447280,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93355,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46662,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 647040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 324650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1290800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49727,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 590130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1484000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454510,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 675750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29788,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 319920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12651,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 645930,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4224400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10645000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70191,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 125720,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93142,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78382,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 637620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73134,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 141910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73117,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 279980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91533,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29604,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86576,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4227900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1855300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2577500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 287260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 80656,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63502,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16484000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 4013800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11527000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2801600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6587500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31168,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 127620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 33098999,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20361000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 352340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 341970,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 315440,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62217,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46139,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 36979,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74301,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4167200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 43800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 121840,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61991,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 139740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62419,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 122650,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27984,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4588200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1113300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 345820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1442500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 778560,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 183210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 337750,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17994,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20657,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1480600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 735000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 437250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2740400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 454780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88088,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81734,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "35328691d97c1af31e10801672f49544cdcc8097",
          "message": "refactor(engine): extract LGSSM binding\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\n\nSeat: 3314012",
          "timestamp": "2026-07-22T06:10:13+02:00",
          "tree_id": "9589475c65a451830372553c6eb06b1915e54fd9",
          "url": "https://github.com/anulum/sc-neurocore/commit/35328691d97c1af31e10801672f49544cdcc8097"
        },
        "date": 1784699082661,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 626690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4904,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 2459,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 633950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 253210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 15973,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4772,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 2442,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 173520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 9839,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 98467,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 2857,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 3665,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 2923,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 931,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 306,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 183,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 492410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 39149,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 40803,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 128520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 158,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 756490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 324,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 173,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 13609,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 1667,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 8031,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 54655000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 22791,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 23159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 25241,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 251690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 103730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 1030500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1409,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 14320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 12177000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1566500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 849240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3329900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 3791100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1362400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 4552400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 561120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 316260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4515900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 87694,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 54138,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 498990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1000100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 118100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 164100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 41329,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 382530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 5511500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2415300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5384800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 120220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 119540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2524500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3144100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 3664200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 3876100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 3558500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 3392800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 2742200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3023300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2569600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2674000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2711900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 1135600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1619000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 2079100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2619000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 742460,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2436500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 3622800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1059600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1071700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 633390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 195830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 176730,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 7013,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5149600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2260100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 2796400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 54328000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 341420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 37143,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 1964900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1104500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1300400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1199500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 289320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1401800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 486800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 514150,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 159700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 244310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 187890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 336830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 37620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 34752,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 531450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2774600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 378430,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 92205,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 49686,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 588190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 305670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1276200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 51336,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 539870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1426500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 465450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 689630,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29949,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 29849,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 299100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 14976,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 6624,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 128330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 582130,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4746400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10392000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 74562,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 235950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 120820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 84878,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 85221,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 518159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 74548,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 170660,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 72935,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 348190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 93278,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29679,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 79369,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86654,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 255640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4208800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3081,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 2143000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 154320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2011899,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 232990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 62637,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 56140,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 309810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 13692000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 2959600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 8600600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 661920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2294400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 5960300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 210810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 29830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 193870,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 76014,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 103490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 210310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 90828,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 557780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 25027000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 19019000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 485100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 328680,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 325010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 72026,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 36905,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 26677,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 75425,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4070200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 57998,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 127830,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 58580,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 111800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 56360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 282050,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 88852,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 32591,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 3962700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 13173,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 721030,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 279450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1182000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 633210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 228180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 313740,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 18349,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 23624,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1668400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 621950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 369910,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2782600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 671100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 402270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 96032,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 117940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 101440,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "committer": {
            "email": "protoscience@anulum.li",
            "name": "Miroslav Šotek",
            "username": "anulum"
          },
          "distinct": true,
          "id": "155d66d1fef8525ac2ab043289a964c5fb46706d",
          "message": "refactor(engine): extract EI network binding\n\nAuthored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)\n\nSeat: 3314012",
          "timestamp": "2026-07-22T06:55:00+02:00",
          "tree_id": "d5eac9b6a383a1cb0ee120a2483165b5621be611",
          "url": "https://github.com/anulum/sc-neurocore/commit/155d66d1fef8525ac2ab043289a964c5fb46706d"
        },
        "date": 1784706367677,
        "tool": "cargo",
        "benches": [
          {
            "name": "pack_bitstream_1m",
            "value": 830530,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 3998,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_dispatch_1m",
            "value": 3977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_1m",
            "value": 841470,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_fast_1m",
            "value": 223490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pack_dispatch_1m",
            "value": 19821,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_portable_1m",
            "value": 4002,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "popcount_simd_1m",
            "value": 3977,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "encoder_64k_steps",
            "value": 142640,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_10k_steps",
            "value": 10429,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lif_100k_steps",
            "value": 104320,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_1024",
            "value": 3548,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_stream_pack_1024",
            "value": 4710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_1024",
            "value": 3285,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_fast_1024",
            "value": 1219,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_1024",
            "value": 291,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_packed_simd_xoshiro_1024",
            "value": 287,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_scalar_16w",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fused_and_popcount_dispatch_16w",
            "value": 7,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_64x32",
            "value": 540450,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_64x32",
            "value": 65913,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fast_flat_64x32_b",
            "value": 67078,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_fused_64x32",
            "value": 277690,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bernoulli_encode_and_popcount_1024",
            "value": 252,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_batch_64x32_x100",
            "value": 1447500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_chacha_fill_1024",
            "value": 266,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prng_xoshiro_fill_1024",
            "value": 269,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dense_forward_prepacked_64x32",
            "value": 20257,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mixed_dense_q88_q1616_64x32",
            "value": 2356,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "block_floating_dense_q16_64x32",
            "value": 10356,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kuramoto_100_osc_1000_steps",
            "value": 65691000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_10x16_20x32",
            "value": 26259,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gnn_20x8_forward",
            "value": 26527,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_1k_steps",
            "value": 24694,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "adex_10k_steps",
            "value": 246980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_1k_steps",
            "value": 93641,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "expif_10k_steps",
            "value": 916990,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_1k_steps",
            "value": 1348,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lapicque_10k_steps",
            "value": 13621,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pv_fs_1k_steps",
            "value": 14052000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sst_1k_steps",
            "value": 1895100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "vip_1k_steps",
            "value": 989370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chandelier_1k_steps",
            "value": 3977200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "basket_cerebellar_1k_steps",
            "value": 4396500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "martinotti_1k_steps",
            "value": 1565900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_motor_1k_steps",
            "value": 5019500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "upper_motor_1k_steps",
            "value": 581950,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "motor_unit_10k_steps",
            "value": 394390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "renshaw_1k_steps",
            "value": 4978000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_motor_10k_steps",
            "value": 93526,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ihc_10k_steps",
            "value": 55975,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rod_10k_steps",
            "value": 442260,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rgc_10k_steps",
            "value": 1031700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "merkel_10k_steps",
            "value": 95619,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pacinian_10k_steps",
            "value": 188330,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nociceptor_10k_steps",
            "value": 33380,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "olfactory_10k_steps",
            "value": 341940,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "granule_10k_steps",
            "value": 6586000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golgi_1k_steps",
            "value": 2776000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stellate_1k_steps",
            "value": 5577200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "lugaro_10k_steps",
            "value": 101220,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ubc_10k_steps",
            "value": 138520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dcn_1k_steps",
            "value": 2974200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "persistent_na_1k_steps",
            "value": 3681500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ih_1k_steps",
            "value": 4210700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ttype_ca_1k_steps",
            "value": 4531900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atype_k_1k_steps",
            "value": 4097200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bk_1k_steps",
            "value": 4194700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sk_1k_steps",
            "value": 3234100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nmda_1k_steps",
            "value": 3455600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "aihara_100k_steps",
            "value": 2339000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "kilinc_bhatt_100k_steps",
            "value": 2402500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ermentrout_kopell_100k_steps",
            "value": 2595500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "montbrio_100k_steps",
            "value": 962800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_100k_steps",
            "value": 1446100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tum_100k_steps",
            "value": 1734900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "elboustani_100k_steps",
            "value": 2207900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "graded_synapse_100k_steps",
            "value": 546090,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gap_junction_100k_steps",
            "value": 2227100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fh_axon_1k_steps",
            "value": 4485300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "node_of_ranvier_1k_steps",
            "value": 1100900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "myelinated_axon_1k_steps",
            "value": 1111100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cardiac_purkinje_1k_steps",
            "value": 742570,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "smooth_muscle_1k_steps",
            "value": 202520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "beta_cell_1k_steps",
            "value": 170550,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brunel_wang_10k_steps",
            "value": 11200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hh_1k_steps",
            "value": 5782300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "traub_miles_1k_steps",
            "value": 2585600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wang_buzsaki_1k_steps",
            "value": 3068500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "connor_stevens_1k_steps",
            "value": 60764000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "destexhe_1k_steps",
            "value": 386670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "huber_braun_1k_steps",
            "value": 39275,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "golomb_fs_1k_steps",
            "value": 2296300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pospischil_1k_steps",
            "value": 1283900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mainen_sejnowski_1k_steps",
            "value": 1437400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "de_schutter_purkinje_1k_steps",
            "value": 1462600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plant_r15_1k_steps",
            "value": 346980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "prescott_10k_steps",
            "value": 1615600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mihalas_niebur_10k_steps",
            "value": 526980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glif_10k_steps",
            "value": 561370,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gif_pop_10k_steps",
            "value": 196590,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "avron_cardiac_1k_steps",
            "value": 291250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "durstewitz_1k_steps",
            "value": 229070,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hill_tononi_1k_steps",
            "value": 372980,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bertram_phantom_1k_steps",
            "value": 41414,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "yamada_1k_steps",
            "value": 35360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fhn_10k_steps",
            "value": 521360,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "morris_lecar_10k_steps",
            "value": 2843600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hindmarsh_rose_10k_steps",
            "value": 447310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "resonate_and_fire_10k_steps",
            "value": 93335,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "balanced_resonate_and_fire_10k_steps",
            "value": 46669,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fitzhugh_rinzel_10k_steps",
            "value": 646770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mckean_10k_steps",
            "value": 325180,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "terman_wang_10k_steps",
            "value": 1291300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "benda_herz_10k_steps",
            "value": 215290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "alpha_10k_steps",
            "value": 49717,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "coba_lif_10k_steps",
            "value": 590240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gutkin_ermentrout_10k_steps",
            "value": 1482600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_hr_10k_steps",
            "value": 454540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_1k_steps",
            "value": 676020,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chay_keizer_1k_steps",
            "value": 29804,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sherman_rinzel_keizer_1k_steps",
            "value": 30715,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "butera_respiratory_1k_steps",
            "value": 321190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "eprop_alif_10k_steps",
            "value": 12655,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "superspike_10k_steps",
            "value": 7400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "learnable_neuron_10k_steps",
            "value": 136420,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pernarowski_10k_steps",
            "value": 645780,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "qif_100k_steps",
            "value": 4224000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "theta_100k_steps",
            "value": 10662000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "perfect_integrator_100k_steps",
            "value": 70092,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gated_lif_100k_steps",
            "value": 238820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nlif_10k_steps",
            "value": 127490,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sfa_10k_steps",
            "value": 93067,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "mat_10k_steps",
            "value": 78372,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "escape_rate_10k_steps",
            "value": 637250,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "klif_100k_steps",
            "value": 73967,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ilif_100k_steps",
            "value": 141960,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "clif_100k_steps",
            "value": 73373,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "plif_100k_steps",
            "value": 280000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "nrlif_10k_steps",
            "value": 91170,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "atif_10k_steps",
            "value": 29638,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigma_delta_100k_steps",
            "value": 149350,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "energy_lif_10k_steps",
            "value": 86897,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "iqif_100k_steps",
            "value": 248810,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cfc_100k_steps",
            "value": 4235900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_lif_10k_steps",
            "value": 3122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "poisson_100k_steps",
            "value": 1855800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "inhom_poisson_100k_steps",
            "value": 218190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "gamma_renewal_100k_steps",
            "value": 2577200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "stochastic_if_10k_steps",
            "value": 286890,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "galves_locherbach_10k_steps",
            "value": 79538,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spike_response_10k_steps",
            "value": 63629,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "glm_10k_steps",
            "value": 344210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wilson_cowan_100k_steps",
            "value": 16492000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "jansen_rit_100k_steps",
            "value": 3930100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wong_wang_100k_steps",
            "value": 11529000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ek_population_100k_steps",
            "value": 617670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "wendling_100k_steps",
            "value": 2805400,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "larter_breakspear_100k_steps",
            "value": 6590500,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "sigmoid_rate_100k_steps",
            "value": 188040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "threshold_linear_100k_steps",
            "value": 31144,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "astrocyte_10k_steps",
            "value": 165240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "tsodyks_markram_10k_steps",
            "value": 83636,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ltc_10k_steps",
            "value": 150240,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compte_wm_10k_steps",
            "value": 209200,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "parallel_spiking_10k_steps",
            "value": 127620,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "fractional_lif_10k_steps",
            "value": 542710,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "siegert_100k_steps",
            "value": 32840000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "amari_field_10k_steps",
            "value": 20419000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "leaky_compete_fire_10k_steps",
            "value": 352100,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi_cuba_100k_steps",
            "value": 342310,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "loihi2_100k_steps",
            "value": 314670,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "truenorth_100k_steps",
            "value": 62281,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "brainscales_1k_steps",
            "value": 46161,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker_lif_10k_steps",
            "value": 37133,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "spinnaker2_100k_steps",
            "value": 74354,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dpi_100k_steps",
            "value": 4393900,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "akida_100k_steps",
            "value": 56832,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "neurogrid_1k_steps",
            "value": 122190,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "multi_timescale_10k_steps",
            "value": 61978,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "attention_gated_10k_steps",
            "value": 138820,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "predictive_coding_10k_steps",
            "value": 62436,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "self_referential_10k_steps",
            "value": 312410,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "compositional_binding_10k_steps",
            "value": 120770,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "differentiable_surrogate_10k_steps",
            "value": 27985,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "continuous_attractor_10k_steps",
            "value": 4450600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "meta_plastic_10k_steps",
            "value": 14403,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "arcane_10k_steps",
            "value": 1113700,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "pinsky_rinzel_1k_steps",
            "value": 346920,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "hay_l5_1k_steps",
            "value": 1446800,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "marder_stg_1k_steps",
            "value": 778000,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rall_cable_1k_steps",
            "value": 180230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "booth_rinzel_1k_steps",
            "value": 338040,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "dendrify_1k_steps",
            "value": 17992,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "two_comp_lif_10k_steps",
            "value": 20665,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "chialvo_100k_steps",
            "value": 1481300,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "rulkov_100k_steps",
            "value": 734760,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "ibarz_tanaka_100k_steps",
            "value": 439850,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "medvedev_100k_steps",
            "value": 2741600,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cazelles_100k_steps",
            "value": 455520,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "courage_nekorkin_100k_steps",
            "value": 514210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "outer_hair_cell_10k_steps",
            "value": 88642,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "cone_photoreceptor_10k_steps",
            "value": 105010,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "taste_receptor_10k_steps",
            "value": 81709,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}