# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Native Mojo kernels for the separately named SC-COMPTE-WM-NETWORK.

The stable C ABI covers kernel-spectrum construction, counter-addressed
Poisson input, and one atomic complete-state transition over all 2,048
excitatory and 512 inhibitory cells. It preserves the scalar Compte model as a
separate source-bounded unit. Caller-owned state enables a Python receipt/run
facade without hiding the native recurrence or substituting a stub.
"""

from std.math import cos, exp, isfinite, sin
from std.memory import UnsafePointer, alloc

comptime F64Ptr = UnsafePointer[Float64, MutAnyOrigin]
comptime I64Ptr = UnsafePointer[Int64, MutAnyOrigin]
comptime U64Ptr = UnsafePointer[UInt64, MutAnyOrigin]

comptime N_EXC: Int = 2048
comptime N_INH: Int = 512
comptime DT_MS: Float64 = 0.02
comptime PI: Float64 = 3.141592653589793238462643383279502884
comptime GOLDEN: UInt64 = 0x9E3779B97F4A7C15
comptime STEP_MIX: UInt64 = 0xD1B54A32D192ED03
comptime STREAM_MIX: UInt64 = 0x94D049BB133111EB


@always_inline
def _f64(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


@always_inline
def _i64(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


@always_inline
def _u64(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def _alloc_f64(n: Int) -> F64Ptr:
    var raw = alloc[Float64](n)
    return F64Ptr(unsafe_from_address=Int(raw))


def _free_f64(pointer: F64Ptr):
    var raw = UnsafePointer[Float64, MutExternalOrigin](
        unsafe_from_address=Int(pointer)
    )
    raw.free()


@always_inline
def _mg_block(voltage: Float64) -> Float64:
    var exponent = -0.062 * voltage
    if exponent < -700.0:
        exponent = -700.0
    elif exponent > 700.0:
        exponent = 700.0
    return 1.0 / (1.0 + exp(exponent) / 3.57)


def _fft_in_place(real: F64Ptr, imag: F64Ptr, inverse: Bool):
    """Apply the fixed-size iterative radix-2 transform in place."""
    for index in range(N_EXC):
        var source = index
        var reversed = 0
        for _ in range(11):
            reversed = (reversed << 1) | (source & 1)
            source = source >> 1
        if reversed > index:
            real[index], real[reversed] = real[reversed], real[index]
            imag[index], imag[reversed] = imag[reversed], imag[index]
    var length = 2
    var sign = -1.0
    if inverse:
        sign = 1.0
    while length <= N_EXC:
        var angle = sign * 2.0 * PI / Float64(length)
        var root_real = cos(angle)
        var root_imag = sin(angle)
        var start = 0
        while start < N_EXC:
            var factor_real = 1.0
            var factor_imag = 0.0
            for offset in range(length // 2):
                var odd_index = start + offset + length // 2
                var odd_real = (
                    factor_real * real[odd_index]
                    - factor_imag * imag[odd_index]
                )
                var odd_imag = (
                    factor_real * imag[odd_index]
                    + factor_imag * real[odd_index]
                )
                var even_real = real[start + offset]
                var even_imag = imag[start + offset]
                real[start + offset] = even_real + odd_real
                imag[start + offset] = even_imag + odd_imag
                real[odd_index] = even_real - odd_real
                imag[odd_index] = even_imag - odd_imag
                var next_factor_real = (
                    factor_real * root_real - factor_imag * root_imag
                )
                factor_imag = factor_real * root_imag + factor_imag * root_real
                factor_real = next_factor_real
            start += length
        length = length << 1
    if inverse:
        for index in range(N_EXC):
            real[index] /= Float64(N_EXC)
            imag[index] /= Float64(N_EXC)


def _circular_sum(
    source: F64Ptr,
    spectrum_real: F64Ptr,
    spectrum_imag: F64Ptr,
    output: F64Ptr,
    scratch_real: F64Ptr,
    scratch_imag: F64Ptr,
):
    for index in range(N_EXC):
        scratch_real[index] = source[index]
        scratch_imag[index] = 0.0
    _fft_in_place(scratch_real, scratch_imag, False)
    for index in range(N_EXC):
        var value_real = scratch_real[index]
        var value_imag = scratch_imag[index]
        scratch_real[index] = (
            value_real * spectrum_real[index]
            - value_imag * spectrum_imag[index]
        )
        scratch_imag[index] = (
            value_real * spectrum_imag[index]
            + value_imag * spectrum_real[index]
        )
    _fft_in_place(scratch_real, scratch_imag, True)
    for index in range(N_EXC):
        output[index] = scratch_real[index]


@export
def sc_compte_wm_network_kernel_spectrum_c(
    j_plus: Float64,
    sigma_deg: Float64,
    output_real_addr: Int,
    output_imag_addr: Int,
) -> Int32:
    """Build one unit-mean circular footprint spectrum.

    Args:
        j_plus: Proximal footprint peak.
        sigma_deg: Circular Gaussian width in degrees.
        output_real_addr: Address of 2,048 writable binary64 real values.
        output_imag_addr: Address of 2,048 writable binary64 imaginary values.

    Returns:
        Zero on success; nonzero means no output is valid.
    """
    if (
        not isfinite(j_plus)
        or j_plus <= 0.0
        or not isfinite(sigma_deg)
        or sigma_deg <= 0.0
    ):
        return 1
    if output_real_addr == 0 or output_imag_addr == 0:
        return 1
    var output_real = _f64(output_real_addr)
    var output_imag = _f64(output_imag_addr)
    var mean = 0.0
    for index in range(N_EXC):
        var angle = Float64(index) * 360.0 / Float64(N_EXC)
        var distance = angle
        if distance >= 180.0:
            distance -= 360.0
        var gaussian = exp(
            -0.5 * (distance / sigma_deg) * (distance / sigma_deg)
        )
        output_real[index] = gaussian
        mean += gaussian
    mean /= Float64(N_EXC)
    var j_minus = (1.0 - j_plus * mean) / (1.0 - mean)
    if not isfinite(j_minus) or j_minus <= 0.0:
        return 2
    var weight_mean = 0.0
    for index in range(N_EXC):
        output_real[index] = j_minus + (j_plus - j_minus) * output_real[index]
        output_imag[index] = 0.0
        weight_mean += output_real[index]
    weight_mean /= Float64(N_EXC)
    for index in range(N_EXC):
        output_real[index] /= weight_mean
    _fft_in_place(output_real, output_imag, False)
    return 0


@always_inline
def _splitmix64(value: UInt64) -> UInt64:
    var mixed = value + GOLDEN
    mixed = (mixed ^ (mixed >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> 27)) * UInt64(0x94D049BB133111EB)
    return mixed ^ (mixed >> 31)


@export
def sc_compte_wm_network_counter_poisson_c(
    population_size: Int,
    rate_hz: Float64,
    dt_ms: Float64,
    seed: UInt64,
    stream: UInt64,
    step_index: UInt64,
    output_addr: Int,
) -> Int32:
    """Write portable per-cell Poisson counts for one counter address.

    The output is an unsigned 64-bit array. The inverse CDF is bounded to 255
    events and a residual tail below 1e-15.
    """
    if population_size <= 0 or output_addr == 0:
        return 1
    if (
        not isfinite(rate_hz)
        or rate_hz < 0.0
        or not isfinite(dt_ms)
        or dt_ms <= 0.0
    ):
        return 1
    var mean = rate_hz * dt_ms / 1000.0
    if mean > 32.0:
        return 1
    var cdf = _alloc_f64(256)
    var probability = exp(-mean)
    var cumulative = probability
    cdf[0] = cumulative
    var cdf_size = 1
    while cumulative < 1.0 - 1.0e-15:
        if cdf_size > 255:
            _free_f64(cdf)
            return 2
        probability *= mean / Float64(cdf_size)
        cumulative += probability
        if cumulative > 1.0:
            cumulative = 1.0
        cdf[cdf_size] = cumulative
        cdf_size += 1
    cdf[cdf_size - 1] = 1.0
    var output = _u64(output_addr)
    for cell in range(population_size):
        var counter = (
            seed
            + step_index * STEP_MIX
            + stream * STREAM_MIX
            + UInt64(cell) * GOLDEN
        )
        var random_bits = _splitmix64(counter)
        var uniform = (
            Float64(random_bits >> 11) + 0.5
        ) * 1.1102230246251565e-16
        var low = 0
        var high = cdf_size
        while low < high:
            var middle = low + (high - low) // 2
            if cdf[middle] < uniform:
                low = middle + 1
            else:
                high = middle
        output[cell] = UInt64(low)
    _free_f64(cdf)
    return 0


def _derivatives(
    structured_ei: Bool,
    modulated: Bool,
    allow_autapses: Bool,
    v_exc: F64Ptr,
    v_inh: F64Ptr,
    ext_exc: F64Ptr,
    ext_inh: F64Ptr,
    nmda: F64Ptr,
    nmda_rise: F64Ptr,
    gabaa: F64Ptr,
    refractory_exc: F64Ptr,
    refractory_inh: F64Ptr,
    current_pa: F64Ptr,
    ee_kernel_zero: Float64,
    ee_spectrum_real: F64Ptr,
    ee_spectrum_imag: F64Ptr,
    ei_spectrum_real: F64Ptr,
    ei_spectrum_imag: F64Ptr,
    dv_exc: F64Ptr,
    dv_inh: F64Ptr,
    dext_exc: F64Ptr,
    dext_inh: F64Ptr,
    dnmda: F64Ptr,
    dnmda_rise: F64Ptr,
    dgabaa: F64Ptr,
    aggregate_ee: F64Ptr,
    aggregate_ei: F64Ptr,
    scratch_real: F64Ptr,
    scratch_imag: F64Ptr,
):
    if structured_ei:
        _circular_sum(
            nmda,
            ei_spectrum_real,
            ei_spectrum_imag,
            scratch_real,
            aggregate_ee,
            scratch_imag,
        )
        for index in range(N_INH):
            aggregate_ei[index] = scratch_real[index * 4]
    else:
        var total_nmda = 0.0
        for index in range(N_EXC):
            total_nmda += nmda[index]
        for index in range(N_INH):
            aggregate_ei[index] = total_nmda
    _circular_sum(
        nmda,
        ee_spectrum_real,
        ee_spectrum_imag,
        aggregate_ee,
        scratch_real,
        scratch_imag,
    )
    if not allow_autapses:
        for index in range(N_EXC):
            aggregate_ee[index] -= ee_kernel_zero * nmda[index]
    var total_gabaa = 0.0
    for index in range(N_INH):
        total_gabaa += gabaa[index]
    var nmda_scale = 1.0
    var gaba_scale = 1.0
    if modulated:
        nmda_scale = 1.2
        gaba_scale = 1.4
    for index in range(N_EXC):
        var voltage = v_exc[index]
        dv_exc[index] = 0.0
        if refractory_exc[index] <= 0.0:
            dv_exc[index] = (
                -0.025 * (voltage + 70.0)
                - 0.0031 * ext_exc[index] * voltage
                - 0.000381
                * nmda_scale
                * aggregate_ee[index]
                * _mg_block(voltage)
                * voltage
                - 0.001336 * gaba_scale * total_gabaa * (voltage + 70.0)
                + current_pa[index] / 1000.0
            ) / 0.5
        dext_exc[index] = -ext_exc[index] / 2.0
        dnmda[index] = -nmda[index] / 100.0 + 0.5 * nmda_rise[index] * (
            1.0 - nmda[index]
        )
        dnmda_rise[index] = -nmda_rise[index] / 2.0
    for index in range(N_INH):
        var voltage = v_inh[index]
        var ii = total_gabaa
        if not allow_autapses:
            ii -= gabaa[index]
        dv_inh[index] = 0.0
        if refractory_inh[index] <= 0.0:
            dv_inh[index] = (
                -0.020 * (voltage + 70.0)
                - 0.00238 * ext_inh[index] * voltage
                - 0.000292
                * nmda_scale
                * aggregate_ei[index]
                * _mg_block(voltage)
                * voltage
                - 0.001024 * gaba_scale * ii * (voltage + 70.0)
            ) / 0.2
        dext_inh[index] = -ext_inh[index] / 2.0
        dgabaa[index] = -gabaa[index] / 10.0


@export
def sc_compte_wm_network_step_c(
    structured_ei_flag: Int64,
    modulated_flag: Int64,
    allow_autapses_flag: Int64,
    v_exc_addr: Int,
    v_inh_addr: Int,
    refractory_exc_addr: Int,
    refractory_inh_addr: Int,
    external_ampa_exc_addr: Int,
    external_ampa_inh_addr: Int,
    recurrent_nmda_addr: Int,
    recurrent_nmda_rise_addr: Int,
    recurrent_gabaa_addr: Int,
    direct_current_pa_addr: Int,
    external_exc_events_addr: Int,
    external_inh_events_addr: Int,
    ee_kernel_zero: Float64,
    ee_spectrum_real_addr: Int,
    ee_spectrum_imag_addr: Int,
    ei_spectrum_real_addr: Int,
    ei_spectrum_imag_addr: Int,
    excitatory_spikes_addr: Int,
    inhibitory_spikes_addr: Int,
) -> Int32:
    """Advance all 2,560 cells by one atomic midpoint-RK2 timestep.

    All arrays are caller-owned contiguous binary64 except event inputs
    (UInt64) and spike outputs (Int64). Status zero commits the complete state;
    any nonzero status leaves every state array unchanged.
    """
    if not (
        (structured_ei_flag == 0 or structured_ei_flag == 1)
        and (modulated_flag == 0 or modulated_flag == 1)
        and (allow_autapses_flag == 0 or allow_autapses_flag == 1)
    ):
        return 1
    var addresses = [
        v_exc_addr,
        v_inh_addr,
        refractory_exc_addr,
        refractory_inh_addr,
        external_ampa_exc_addr,
        external_ampa_inh_addr,
        recurrent_nmda_addr,
        recurrent_nmda_rise_addr,
        recurrent_gabaa_addr,
        direct_current_pa_addr,
        external_exc_events_addr,
        external_inh_events_addr,
        ee_spectrum_real_addr,
        ee_spectrum_imag_addr,
        excitatory_spikes_addr,
        inhibitory_spikes_addr,
    ]
    for address in addresses:
        if address == 0:
            return 1
    if structured_ei_flag == 1 and (
        ei_spectrum_real_addr == 0 or ei_spectrum_imag_addr == 0
    ):
        return 1
    if not isfinite(ee_kernel_zero) or ee_kernel_zero <= 0.0:
        return 1
    var v_exc = _f64(v_exc_addr)
    var v_inh = _f64(v_inh_addr)
    var ref_exc = _f64(refractory_exc_addr)
    var ref_inh = _f64(refractory_inh_addr)
    var ext_exc_state = _f64(external_ampa_exc_addr)
    var ext_inh_state = _f64(external_ampa_inh_addr)
    var nmda_state = _f64(recurrent_nmda_addr)
    var nmda_rise_state = _f64(recurrent_nmda_rise_addr)
    var gabaa_state = _f64(recurrent_gabaa_addr)
    var current_pa = _f64(direct_current_pa_addr)
    var exc_events = _u64(external_exc_events_addr)
    var inh_events = _u64(external_inh_events_addr)
    var exc_spikes = _i64(excitatory_spikes_addr)
    var inh_spikes = _i64(inhibitory_spikes_addr)
    for index in range(N_EXC):
        if not (
            isfinite(v_exc[index])
            and v_exc[index] >= -200.0
            and v_exc[index] <= 100.0
            and isfinite(ref_exc[index])
            and ref_exc[index] >= 0.0
            and ref_exc[index] <= 1.0e6
            and isfinite(ext_exc_state[index])
            and ext_exc_state[index] >= 0.0
            and ext_exc_state[index] <= 1.0e6
            and isfinite(nmda_state[index])
            and nmda_state[index] >= 0.0
            and nmda_state[index] <= 1.0
            and isfinite(nmda_rise_state[index])
            and nmda_rise_state[index] >= 0.0
            and nmda_rise_state[index] <= 1.0e6
            and isfinite(current_pa[index])
        ):
            return 2
        if Float64(exc_events[index]) + ext_exc_state[index] > 1.0e6:
            return 3
    for index in range(N_INH):
        if not (
            isfinite(v_inh[index])
            and v_inh[index] >= -200.0
            and v_inh[index] <= 100.0
            and isfinite(ref_inh[index])
            and ref_inh[index] >= 0.0
            and ref_inh[index] <= 1.0e6
            and isfinite(ext_inh_state[index])
            and ext_inh_state[index] >= 0.0
            and ext_inh_state[index] <= 1.0e6
            and isfinite(gabaa_state[index])
            and gabaa_state[index] >= 0.0
            and gabaa_state[index] <= 1.0e6
        ):
            return 2
        if Float64(inh_events[index]) + ext_inh_state[index] > 1.0e6:
            return 3
    var ee_spectrum_real = _f64(ee_spectrum_real_addr)
    var ee_spectrum_imag = _f64(ee_spectrum_imag_addr)
    var ei_spectrum_real = ee_spectrum_real
    var ei_spectrum_imag = ee_spectrum_imag
    if structured_ei_flag == 1:
        ei_spectrum_real = _f64(ei_spectrum_real_addr)
        ei_spectrum_imag = _f64(ei_spectrum_imag_addr)

    var ext_exc = _alloc_f64(N_EXC)
    var ext_inh = _alloc_f64(N_INH)
    var derivative_v_exc = _alloc_f64(N_EXC)
    var derivative_v_inh = _alloc_f64(N_INH)
    var derivative_ext_exc = _alloc_f64(N_EXC)
    var derivative_ext_inh = _alloc_f64(N_INH)
    var derivative_nmda = _alloc_f64(N_EXC)
    var derivative_nmda_rise = _alloc_f64(N_EXC)
    var derivative_gabaa = _alloc_f64(N_INH)
    var stage_v_exc = _alloc_f64(N_EXC)
    var stage_v_inh = _alloc_f64(N_INH)
    var stage_ext_exc = _alloc_f64(N_EXC)
    var stage_ext_inh = _alloc_f64(N_INH)
    var stage_nmda = _alloc_f64(N_EXC)
    var stage_nmda_rise = _alloc_f64(N_EXC)
    var stage_gabaa = _alloc_f64(N_INH)
    var aggregate_ee = _alloc_f64(N_EXC)
    var aggregate_ei = _alloc_f64(N_INH)
    var scratch_real = _alloc_f64(N_EXC)
    var scratch_imag = _alloc_f64(N_EXC)
    for index in range(N_EXC):
        ext_exc[index] = ext_exc_state[index] + Float64(exc_events[index])
    for index in range(N_INH):
        ext_inh[index] = ext_inh_state[index] + Float64(inh_events[index])
    _derivatives(
        structured_ei_flag == 1,
        modulated_flag == 1,
        allow_autapses_flag == 1,
        v_exc,
        v_inh,
        ext_exc,
        ext_inh,
        nmda_state,
        nmda_rise_state,
        gabaa_state,
        ref_exc,
        ref_inh,
        current_pa,
        ee_kernel_zero,
        ee_spectrum_real,
        ee_spectrum_imag,
        ei_spectrum_real,
        ei_spectrum_imag,
        derivative_v_exc,
        derivative_v_inh,
        derivative_ext_exc,
        derivative_ext_inh,
        derivative_nmda,
        derivative_nmda_rise,
        derivative_gabaa,
        aggregate_ee,
        aggregate_ei,
        scratch_real,
        scratch_imag,
    )
    for index in range(N_EXC):
        stage_v_exc[index] = (
            v_exc[index] + 0.5 * DT_MS * derivative_v_exc[index]
        )
        stage_ext_exc[index] = (
            ext_exc[index] + 0.5 * DT_MS * derivative_ext_exc[index]
        )
        stage_nmda[index] = (
            nmda_state[index] + 0.5 * DT_MS * derivative_nmda[index]
        )
        stage_nmda_rise[index] = (
            nmda_rise_state[index] + 0.5 * DT_MS * derivative_nmda_rise[index]
        )
    for index in range(N_INH):
        stage_v_inh[index] = (
            v_inh[index] + 0.5 * DT_MS * derivative_v_inh[index]
        )
        stage_ext_inh[index] = (
            ext_inh[index] + 0.5 * DT_MS * derivative_ext_inh[index]
        )
        stage_gabaa[index] = (
            gabaa_state[index] + 0.5 * DT_MS * derivative_gabaa[index]
        )
    _derivatives(
        structured_ei_flag == 1,
        modulated_flag == 1,
        allow_autapses_flag == 1,
        stage_v_exc,
        stage_v_inh,
        stage_ext_exc,
        stage_ext_inh,
        stage_nmda,
        stage_nmda_rise,
        stage_gabaa,
        ref_exc,
        ref_inh,
        current_pa,
        ee_kernel_zero,
        ee_spectrum_real,
        ee_spectrum_imag,
        ei_spectrum_real,
        ei_spectrum_imag,
        derivative_v_exc,
        derivative_v_inh,
        derivative_ext_exc,
        derivative_ext_inh,
        derivative_nmda,
        derivative_nmda_rise,
        derivative_gabaa,
        aggregate_ee,
        aggregate_ei,
        scratch_real,
        scratch_imag,
    )
    var valid = True
    for index in range(N_EXC):
        stage_v_exc[index] = v_exc[index] + DT_MS * derivative_v_exc[index]
        stage_ext_exc[index] = (
            ext_exc[index] + DT_MS * derivative_ext_exc[index]
        )
        stage_nmda[index] = nmda_state[index] + DT_MS * derivative_nmda[index]
        stage_nmda_rise[index] = (
            nmda_rise_state[index] + DT_MS * derivative_nmda_rise[index]
        )
        if not (
            isfinite(stage_v_exc[index])
            and stage_v_exc[index] >= -200.0
            and stage_v_exc[index] <= 100.0
            and isfinite(stage_ext_exc[index])
            and stage_ext_exc[index] >= 0.0
            and stage_ext_exc[index] <= 1.0e6
            and isfinite(stage_nmda[index])
            and stage_nmda[index] >= 0.0
            and stage_nmda[index] <= 1.0
            and isfinite(stage_nmda_rise[index])
            and stage_nmda_rise[index] >= 0.0
            and stage_nmda_rise[index] <= 1.0e6
        ):
            valid = False
        if (
            ref_exc[index] <= 0.0
            and stage_v_exc[index] >= -50.0
            and stage_nmda_rise[index] > 1.0e6 - 1.0
        ):
            valid = False
        exc_spikes[index] = 0
    for index in range(N_INH):
        stage_v_inh[index] = v_inh[index] + DT_MS * derivative_v_inh[index]
        stage_ext_inh[index] = (
            ext_inh[index] + DT_MS * derivative_ext_inh[index]
        )
        stage_gabaa[index] = (
            gabaa_state[index] + DT_MS * derivative_gabaa[index]
        )
        if not (
            isfinite(stage_v_inh[index])
            and stage_v_inh[index] >= -200.0
            and stage_v_inh[index] <= 100.0
            and isfinite(stage_ext_inh[index])
            and stage_ext_inh[index] >= 0.0
            and stage_ext_inh[index] <= 1.0e6
            and isfinite(stage_gabaa[index])
            and stage_gabaa[index] >= 0.0
            and stage_gabaa[index] <= 1.0e6
        ):
            valid = False
        if (
            ref_inh[index] <= 0.0
            and stage_v_inh[index] >= -50.0
            and stage_gabaa[index] > 1.0e6 - 1.0
        ):
            valid = False
        inh_spikes[index] = 0
    if valid:
        for index in range(N_EXC):
            var active = ref_exc[index] <= 0.0
            ref_exc[index] = max(0.0, ref_exc[index] - DT_MS)
            if not active:
                stage_v_exc[index] = -60.0
            elif stage_v_exc[index] >= -50.0:
                stage_v_exc[index] = -60.0
                ref_exc[index] = 2.0
                stage_nmda_rise[index] += 1.0
                exc_spikes[index] = 1
            v_exc[index] = stage_v_exc[index]
            ext_exc_state[index] = stage_ext_exc[index]
            nmda_state[index] = stage_nmda[index]
            nmda_rise_state[index] = stage_nmda_rise[index]
        for index in range(N_INH):
            var active = ref_inh[index] <= 0.0
            ref_inh[index] = max(0.0, ref_inh[index] - DT_MS)
            if not active:
                stage_v_inh[index] = -60.0
            elif stage_v_inh[index] >= -50.0:
                stage_v_inh[index] = -60.0
                ref_inh[index] = 1.0
                stage_gabaa[index] += 1.0
                inh_spikes[index] = 1
            v_inh[index] = stage_v_inh[index]
            ext_inh_state[index] = stage_ext_inh[index]
            gabaa_state[index] = stage_gabaa[index]
    _free_f64(ext_exc)
    _free_f64(ext_inh)
    _free_f64(derivative_v_exc)
    _free_f64(derivative_v_inh)
    _free_f64(derivative_ext_exc)
    _free_f64(derivative_ext_inh)
    _free_f64(derivative_nmda)
    _free_f64(derivative_nmda_rise)
    _free_f64(derivative_gabaa)
    _free_f64(stage_v_exc)
    _free_f64(stage_v_inh)
    _free_f64(stage_ext_exc)
    _free_f64(stage_ext_inh)
    _free_f64(stage_nmda)
    _free_f64(stage_nmda_rise)
    _free_f64(stage_gabaa)
    _free_f64(aggregate_ee)
    _free_f64(aggregate_ei)
    _free_f64(scratch_real)
    _free_f64(scratch_imag)
    if not valid:
        return 4
    return 0


# Build: mojo build --emit shared-lib --target-cpu x86-64-v3 -o libsc_compte_wm_network.so sc_compte_wm_network.mojo
