# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (fixed_point_reference) from former test_kuramoto_rtl.py

from __future__ import annotations

from tests.test_hdl_gen.kuramoto_rtl_support import *  # noqa: F403


def test_kuramoto_emitter_fixed_point_reference_matches_known_coupled_step() -> None:
    emitter = KuramotoEmitter(
        n_oscillators=2,
        omegas=[0.0, 0.0],
        initial_phases=[0.0, 1.5707963267948966],
        coupling=1.0,
        dt=0.01,
        data_width=24,
        fraction=16,
        lut_size=64,
    )

    assert emitter.initial_phase_state_fixed() == [0, 102944]
    assert emitter.fixed_point_step(emitter.initial_phase_state_fixed()) == [327, 102618]


@pytest.mark.parametrize(
    ("phase_state", "message"),
    [
        ([0], "phase_state length must equal n_oscillators"),
        ([True, 0], "phase_state entries must be integers"),
        ([0.25, 0], "phase_state entries must be integers"),
        ([-1, 0], "phase_state entries must satisfy 0 <= phase < phase modulus"),
        (
            [int(round(2.0 * math.pi * (1 << 16))), 0],
            "phase_state entries must satisfy 0 <= phase < phase modulus",
        ),
    ],
)
def test_kuramoto_emitter_fixed_point_step_rejects_noncanonical_phase_state(
    phase_state: list[object], message: str
) -> None:
    emitter = KuramotoEmitter(n_oscillators=2, fraction=16)

    with pytest.raises(ValueError, match=message):
        emitter.fixed_point_step(cast(list[int], phase_state))


@pytest.mark.parametrize(
    ("phase_state", "message"),
    [
        ([0], "phase_state length must equal n_oscillators"),
        ([False, 0], "phase_state entries must be integers"),
        ([1.5, 0], "phase_state entries must be integers"),
        ([-1, 0], "phase_state entries must satisfy 0 <= phase < phase modulus"),
        (
            [int(round(2.0 * math.pi * (1 << 16))), 0],
            "phase_state entries must satisfy 0 <= phase < phase modulus",
        ),
    ],
)
def test_kuramoto_emitter_fixed_state_to_float_rejects_noncanonical_phase_state(
    phase_state: list[object], message: str
) -> None:
    emitter = KuramotoEmitter(n_oscillators=2, fraction=16)

    with pytest.raises(ValueError, match=message):
        emitter.fixed_state_to_float(cast(list[int], phase_state))


def test_kuramoto_emitter_characterises_fixed_point_error_against_float_reference() -> None:
    emitter = KuramotoEmitter(
        n_oscillators=4,
        omegas=[0.88, 0.96, 1.04, 1.12],
        initial_phases=[0.05, 0.9, 1.7, 2.6],
        coupling=1.0,
        dt=2.5e-4,
        data_width=24,
        fraction=16,
        lut_size=128,
    )

    summary = emitter.fixed_point_error_summary(steps=32)

    assert summary["steps"] == 32
    assert summary["oscillator_count"] == 4
    assert cast(float, summary["max_abs_phase_error_rad"]) < 0.01
    assert cast(float, summary["rms_phase_error_rad"]) < 0.005
    assert summary["final_fixed_phases_rad"] != summary["final_float_phases_rad"]


def test_kuramoto_emitter_fixed_point_error_summary_requires_positive_steps() -> None:
    emitter = KuramotoEmitter()

    with pytest.raises(ValueError, match="steps must be >= 1"):
        emitter.fixed_point_error_summary(steps=0)


def test_kuramoto_emitter_hdl_matches_no_coupling_fixed_point_step(tmp_path: Path) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_no_coupling_step",
        n_oscillators=1,
        omegas=[1.0],
        initial_phases=[0.0],
        coupling=0.0,
        dt=0.01,
        data_width=24,
        fraction=16,
    )
    expected_next = emitter.fixed_point_step(emitter.initial_phase_state_fixed())
    source = (
        emitter.generate()
        + f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg step_en = 1'b0;
    wire update_done;
    wire [23:0] phase_bus;

    kuramoto_no_coupling_step dut (
        .clk(clk),
        .rst_n(rst_n),
        .step_en(step_en),
        .update_done(update_done),
        .phase_bus(phase_bus)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        step_en = 1'b1;
        @(posedge clk);
        @(negedge clk);

        if (phase_bus !== 24'd{expected_next[0]}) begin
            $fatal(
                1,
                "expected one fixed-point phase step of {expected_next[0]}, observed %0d",
                phase_bus
            );
        end
        if (update_done !== 1'b1) begin
            $fatal(1, "expected update_done pulse after enabled step");
        end

        step_en = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (update_done !== 1'b0) begin
            $fatal(1, "expected update_done to clear when step_en is low");
        end

        $finish(0);
    end
endmodule
"""
    )
    rtl_path = tmp_path / "kuramoto_no_coupling_step.v"
    sim_path = tmp_path / "kuramoto_no_coupling_step.out"
    rtl_path.write_text(source)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run(
        [vvp, str(sim_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr


def test_kuramoto_emitter_hdl_matches_coupled_two_oscillator_fixed_point_step(
    tmp_path: Path,
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise AssertionError("iverilog and vvp must be available for HDL simulation tests")

    emitter = KuramotoEmitter(
        module_name="kuramoto_coupled_step",
        n_oscillators=2,
        omegas=[0.0, 0.0],
        initial_phases=[0.0, 1.5707963267948966],
        coupling=1.0,
        dt=0.01,
        data_width=24,
        fraction=16,
        lut_size=64,
    )
    initial_state = emitter.initial_phase_state_fixed()
    expected_next = emitter.fixed_point_step(initial_state)
    source = (
        emitter.generate()
        + f"""
module tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg step_en = 1'b0;
    wire update_done;
    wire [47:0] phase_bus;

    kuramoto_coupled_step dut (
        .clk(clk),
        .rst_n(rst_n),
        .step_en(step_en),
        .update_done(update_done),
        .phase_bus(phase_bus)
    );

    always #5 clk = ~clk;

    initial begin
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        if (phase_bus[23:0] !== 24'd{initial_state[0]} || phase_bus[47:24] !== 24'd{initial_state[1]}) begin
            $fatal(
                1,
                "unexpected initial phases p0=%0d p1=%0d",
                phase_bus[23:0],
                phase_bus[47:24]
            );
        end

        step_en = 1'b1;
        @(posedge clk);
        @(negedge clk);

        if (phase_bus[23:0] !== 24'd{expected_next[0]} || phase_bus[47:24] !== 24'd{expected_next[1]}) begin
            $fatal(
                1,
                "unexpected coupled phase step p0=%0d p1=%0d",
                phase_bus[23:0],
                phase_bus[47:24]
            );
        end
        if (update_done !== 1'b1) begin
            $fatal(1, "expected update_done pulse after coupled step");
        end

        $finish(0);
    end
endmodule
"""
    )
    rtl_path = tmp_path / "kuramoto_coupled_step.v"
    sim_path = tmp_path / "kuramoto_coupled_step.out"
    rtl_path.write_text(source)

    compile_result = subprocess.run(
        [iverilog, "-g2012", "-o", str(sim_path), str(rtl_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    sim_result = subprocess.run(
        [vvp, str(sim_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert sim_result.returncode == 0, sim_result.stdout + sim_result.stderr
