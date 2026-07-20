window.BENCHMARK_DATA = {
  "lastUpdate": 1784532338603,
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
      }
    ]
  }
}