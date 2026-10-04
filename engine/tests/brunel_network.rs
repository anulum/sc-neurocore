// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public fixed-point Brunel network contracts

use rand::SeedableRng;
use rand_distr::{Distribution, Poisson};
use rand_xoshiro::Xoshiro256PlusPlus;
use sc_neurocore_engine::brunel::BrunelNetwork;
use sc_neurocore_engine::neuron::{mask, FixedPointLif};

struct Parameters {
    n: usize,
    rows: Vec<usize>,
    columns: Vec<usize>,
    weights: Vec<i16>,
    width: u32,
    fraction: u32,
    refractory: i32,
    lambda: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            n: 4,
            rows: vec![0, 1, 2, 3, 4],
            columns: vec![1, 2, 3, 0],
            weights: vec![128; 4],
            width: 16,
            fraction: 8,
            refractory: 2,
            lambda: 0.7,
        }
    }
}

impl Parameters {
    fn build(self) -> Result<BrunelNetwork, String> {
        BrunelNetwork::new(
            self.n,
            self.rows,
            self.columns,
            self.weights,
            self.width,
            self.fraction,
            0,
            0,
            256,
            self.refractory,
            20,
            256,
            self.lambda,
            128,
            17,
        )
    }
}

#[test]
fn invalid_csr_bounds_and_order_are_rejected() {
    for (rows, columns, weights, message) in [
        (
            vec![0],
            vec![1, 2, 3, 0],
            vec![128; 4],
            "w_row_offsets length 1 != n_neurons+1=5",
        ),
        (
            vec![1, 1, 2, 3, 4],
            vec![1, 2, 3, 0],
            vec![128; 4],
            "w_row_offsets must start at 0",
        ),
        (
            vec![0, 2, 1, 3, 4],
            vec![1, 2, 3, 0],
            vec![128; 4],
            "w_row_offsets must be nondecreasing",
        ),
        (
            vec![0, 1, 2, 3, 3],
            vec![1, 2, 3, 0],
            vec![128; 4],
            "w_row_offsets must end at the number of weights",
        ),
        (
            vec![0, 8, 8, 8, 8],
            vec![1, 2, 3, 0],
            vec![128; 4],
            "w_row_offsets must end at the number of weights",
        ),
        (
            vec![0, 1, 2, 3, 4],
            vec![1, 2, 3, 4],
            vec![128; 4],
            "w_col_indices must be less than n_neurons",
        ),
        (
            vec![0, 1, 2, 3, 4],
            vec![1, 2, usize::MAX, 0],
            vec![128; 4],
            "w_col_indices must be less than n_neurons",
        ),
        (
            vec![0, 1, 2, 3, 4],
            vec![1, 2, 3, 0],
            vec![128],
            "w_col_indices len 4 != w_values len 1",
        ),
    ] {
        let result = Parameters {
            rows,
            columns,
            weights,
            ..Default::default()
        }
        .build();
        assert_eq!(result.err().expect("invalid CSR must fail"), message);
    }
}

#[test]
fn invalid_drive_and_fixed_point_configuration_are_rejected() {
    for lambda in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        let result = Parameters {
            lambda,
            ..Default::default()
        }
        .build();
        assert_eq!(
            result.err().unwrap(),
            "ext_lambda must be finite and nonnegative"
        );
    }
    assert!(Parameters {
        lambda: 1e20,
        ..Default::default()
    }
    .build()
    .is_err());
    for width in [0, 17, 33] {
        let result = Parameters {
            width,
            ..Default::default()
        }
        .build();
        assert_eq!(result.err().unwrap(), "data_width must be in [1, 16]");
    }
    for fraction in [16, 32] {
        let result = Parameters {
            fraction,
            ..Default::default()
        }
        .build();
        assert_eq!(
            result.err().unwrap(),
            "fraction must be less than data_width"
        );
    }
    let result = Parameters {
        refractory: -1,
        ..Default::default()
    }
    .build();
    assert_eq!(
        result.err().unwrap(),
        "refractory_period must be nonnegative"
    );
    assert!(Parameters {
        n: usize::MAX,
        ..Default::default()
    }
    .build()
    .is_err());
}

#[test]
fn seeded_small_mean_trace_and_split_continuation_are_preserved() {
    let mut network = Parameters::default().build().unwrap();
    let expected = vec![
        1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 2, 0, 0, 1, 0, 2,
        2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 1, 1, 0, 1, 0, 2, 1, 0, 1, 2, 0, 1, 3, 0, 0, 0, 0, 2, 2, 0,
        1, 0, 1, 0,
    ];
    assert_eq!(network.run(64), expected);
    let mut split = Parameters::default().build().unwrap();
    let mut counts = split.run(23);
    counts.extend(split.run(41));
    assert_eq!(counts, expected);
}

#[test]
fn large_means_follow_the_public_poisson_and_lif_components() {
    for lambda in [30.0, 1000.0, 100000.0, 1e10] {
        for weight in [32767_i16, -32768] {
            let mut network = BrunelNetwork::new(
                1,
                vec![0, 0],
                vec![],
                vec![],
                16,
                0,
                0,
                0,
                900,
                0,
                0,
                1,
                lambda,
                weight,
                17,
            )
            .unwrap();
            let poisson = Poisson::<f64>::new(lambda).unwrap();
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(17);
            let mut reference = FixedPointLif::new(16, 0, 0, 0, 900, 0);
            let expected: Vec<u32> = (0..64)
                .map(|_| {
                    let count = poisson.sample(&mut rng) as u64;
                    let current = (count as i32).wrapping_mul(weight as i32);
                    reference.step(0, 1, mask(current, 16), 0).0 as u32
                })
                .collect();
            assert_eq!(network.run(64), expected);
        }
    }
}

#[test]
fn duplicate_connections_wrap_large_synaptic_sums() {
    let count = 65539;
    let mut network = BrunelNetwork::new(
        1,
        vec![0, count],
        vec![0; count],
        vec![32767; count],
        16,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0.0,
        0,
        17,
    )
    .unwrap();
    assert_eq!(network.run(2), vec![1, 1]);
}

#[test]
fn empty_csr_zero_drive_and_width_boundaries_are_valid() {
    for (n, rows, width, fraction) in [
        (0, vec![0], 16, 8),
        (4, vec![0; 5], 1, 0),
        (4, vec![0; 5], 16, 15),
    ] {
        let mut network = Parameters {
            n,
            rows,
            columns: vec![],
            weights: vec![],
            width,
            fraction,
            refractory: 0,
            lambda: 0.0,
        }
        .build()
        .unwrap();
        assert_eq!(network.run(8), vec![0; 8]);
    }
}
