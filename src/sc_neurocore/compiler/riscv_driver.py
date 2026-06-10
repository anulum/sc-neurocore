# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — RISC-V SoC drivers

"""RISC-V driver generation utilities for compiled neuron modules.

Generates C-based MMIO drivers for RISC-V SoC integration, with support
for bare-metal, FreeRTOS, and Zephyr RTOS templates.
"""

from __future__ import annotations

from typing import Literal


def generate_riscv_driver(
    module_name: str,
    params: dict[str, int],
    *,
    base_address: int = 0x4000_0000,
    data_width: int = 16,
    fraction: int = 8,
    rtos: Literal["freertos", "zephyr", "baremetal"] = "baremetal",
) -> str:
    """Generate a RISC-V C driver for neuron control via MMIO.

    Supports bare-metal, FreeRTOS, and Zephyr templates with timer-driven
    neuron tick tasks for real-time operating system integration.

    Parameters
    ----------
    module_name : str
        Neuron module name.
    params : dict[str, int]
        Parameter names and bit widths.
    base_address : int
        MMIO base address.
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits.
    rtos : str
        ``"baremetal"``, ``"freertos"``, or ``"zephyr"``.

    Returns
    -------
    str
        Complete RISC-V C driver with optional RTOS integration.
    """
    guard = module_name.upper() + "_RISCV_H"
    upper = module_name.upper()

    lines = [
        f"/* Auto-generated RISC-V driver for {module_name} */",
        f"/* SC-NeuroCore — RISC-V SoC integration ({rtos}) */",
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        f"#define {upper}_BASE    0x{base_address:08X}U",
        f"#define {upper}_FRAC    {fraction}",
        f"#define {upper}_CTRL    ({upper}_BASE + 0x00)",
        f"#define {upper}_I_T     ({upper}_BASE + 0x04)",
        f"#define {upper}_SPIKES  ({upper}_BASE + 0x08)",
    ]

    for i, pname in enumerate(params):
        lines.append(f"#define {upper}_{pname.upper()}  ({upper}_BASE + 0x{0x0C + i * 4:02X})")

    lines.extend(
        [
            "",
            "#define MMIO_WR(a,v) (*(volatile uint32_t*)(a) = (v))",
            "#define MMIO_RD(a)   (*(volatile uint32_t*)(a))",
            "",
            f"static inline int32_t {module_name}_encode(float v) {{",
            f"    return (int32_t)(v * (1 << {upper}_FRAC));",
            "}",
            "",
            f"static inline void {module_name}_enable(void)  {{ MMIO_WR({upper}_CTRL, 0x01); }}",
            f"static inline void {module_name}_disable(void) {{ MMIO_WR({upper}_CTRL, 0x00); }}",
            "",
            f"static inline void {module_name}_reset(void) {{",
            f"    MMIO_WR({upper}_CTRL, 0x02);",
            f"    MMIO_WR({upper}_CTRL, 0x01);",
            "}",
            "",
            f"static inline void {module_name}_set_current(float I) {{",
            f"    MMIO_WR({upper}_I_T, (uint32_t){module_name}_encode(I));",
            "}",
            "",
            f"__attribute__((weak)) float {module_name}_read_current(void) {{",
            "    return 0.0f;",
            "}",
            "",
            f"static inline uint32_t {module_name}_get_spikes(void) {{",
            f"    return MMIO_RD({upper}_SPIKES);",
            "}",
        ]
    )

    for pname in params:
        lines.extend(
            [
                "",
                f"static inline void {module_name}_set_{pname.lower()}(float v) {{",
                f"    MMIO_WR({upper}_{pname.upper()}, (uint32_t){module_name}_encode(v));",
                "}",
            ]
        )

    if rtos == "freertos":
        lines.extend(
            [
                "",
                "/* ── FreeRTOS neuron tick task ───────────────────── */",
                '#include "FreeRTOS.h"',
                '#include "task.h"',
                "",
                f"static void {module_name}_tick(void *p) {{",
                "    (void)p;",
                f"    {module_name}_reset();",
                f"    {module_name}_enable();",
                "    for (;;) {",
                f"        float I = {module_name}_read_current();",
                f"        {module_name}_set_current(I);",
                "        vTaskDelay(pdMS_TO_TICKS(1));",
                "    }",
                "}",
                "",
                f"static inline void {module_name}_start_rtos(void) {{",
                f'    xTaskCreate({module_name}_tick, "{module_name}",',
                "                configMINIMAL_STACK_SIZE, NULL,",
                "                tskIDLE_PRIORITY + 1, NULL);",
                "}",
            ]
        )
    elif rtos == "zephyr":
        lines.extend(
            [
                "",
                "/* ── Zephyr neuron tick thread ──────────────────── */",
                "#include <zephyr/kernel.h>",
                "",
                f"static void {module_name}_thread(void *a, void *b, void *c) {{",
                "    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);",
                f"    {module_name}_reset();",
                f"    {module_name}_enable();",
                "    while (1) {",
                f"        {module_name}_set_current({module_name}_read_current());",
                "        k_msleep(1);",
                "    }",
                "}",
                "",
                f"K_THREAD_DEFINE({module_name}_tid, 1024,",
                f"    {module_name}_thread, NULL, NULL, NULL, 5, 0, 0);",
            ]
        )

    lines.extend(["", f"#endif /* {guard} */", ""])
    return "\n".join(lines)
