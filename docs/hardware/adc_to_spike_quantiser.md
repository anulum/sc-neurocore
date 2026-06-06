# ADC-to-Spike Quantiser HDL

The ADC-to-spike quantiser is the NEU-C.5 sensor-ingress block for converting sampled analogue probe data into AER events. It is designed for the MIF B-dot probe ingress path and sits between a JESD204B receive wrapper and the strict-priority AER queue.

## Contract

The module lives at `hdl/sensors/adc_to_spike_quantiser.v` and exposes three stages:

1. ADC sample quantisation into a signed Q-format value, default Q8.8.
2. Decimation over a fixed sample window, with rounded average amplitude.
3. Deterministic rate-coded AER event generation, where `floor(abs(window_q) / threshold_q)` events are emitted after each completed window.

Negative windows use `BASE_ADDRESS + NEGATIVE_OFFSET`; non-negative windows use `BASE_ADDRESS`. `aer_polarity` is also exported so downstream bridges can preserve signed sensor direction without decoding addresses.

## Safety behaviour

`adc_ready` is asserted only when the quantiser can accept a sample. If a source asserts `adc_valid` while `adc_ready` is low, `dropped_sample` latches high. If `threshold_q == 0`, `threshold_error` latches high and `adc_ready` remains low. Pending AER events remain stable under downstream backpressure until `aer_ready` consumes them.

The formal harness proves the local safety contract under bounded backpressure:

1. Ready-obeying ADC sources do not drop samples.
2. AER addresses match the encoded polarity.
3. Completed windows transfer their computed average and spike budget into registered diagnostics.
4. Positive and negative windows are covered, and AER address polarity is asserted.

## Replication

Run the focused tests:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_adc_to_spike_quantiser.py cosim/test_adc_to_spike_cosim.py -q
```

Run the formal proof directly:

```bash
sby -f hdl/formal/sensors/adc_to_spike_quantiser.sby
```

Run the benchmark on isolated cores:

```bash
sudo systemctl set-property --runtime system.slice AllowedCPUs=0-9
sudo systemctl set-property --runtime user.slice AllowedCPUs=0-9
sudo systemctl set-property --runtime init.scope AllowedCPUs=0-9
sudo systemd-run --wait --collect --pipe --slice=benchmark.slice \
  -p AllowedCPUs=10-11 -p CPUAffinity='10 11' \
  --uid=anulum --gid=anulum --working-directory="$PWD" \
  --setenv=PYTHONPATH="src:tools" \
  /usr/bin/taskset -c 10-11 \
  .venv/bin/python benchmarks/bench_adc_to_spike_quantiser.py \
  --samples 4096 --repeats 100 \
  --json benchmarks/results/local_python_2026-06-04_adc_to_spike_quantiser.json
sudo systemctl set-property --runtime system.slice AllowedCPUs=0-31
sudo systemctl set-property --runtime user.slice AllowedCPUs=0-31
sudo systemctl set-property --runtime init.scope AllowedCPUs=0-31
```

The committed benchmark artefact is local regression evidence and records `hardware_measurement_claimed=false`. The 2026-06-04 run used cpuset `10-11`, measured `3704.696` ns/sample in the Python reference, proved the formal harness in `4.579` s, and reported `7675` Yosys cells in `11.861` s. Hardware throughput claims require an isolated hardware runner and board-level evidence.
