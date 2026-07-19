window.BENCHMARK_DATA = {
  "lastUpdate": 1784430325216,
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
      }
    ]
  }
}