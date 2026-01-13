"""
SC NeuroCore PYNQ Driver
========================

Python driver for the sc-neurocore FPGA IP on PYNQ-Z2.
Handles AXI-Lite register configuration and status monitoring.

Author: SCPN Research Team
Date: January 2026
"""

import time
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SCNeuroCoreDriver:
    """
    Driver for sc_neurocore IP.
    
    Register Map:
    - CTRL (0x00): [0] Start
    - STATUS (0x04): [0] Busy, [1] Done
    - X_INPUTS (0x10...): Input values (Q8.8)
    - WEIGHTS (0x20...): Weight values (Q8.8)
    - Y_RANGE (0x30, 0x34): Min/Max current
    - CONFIG (0x40...): Stream len, dt, scale
    - BIOLOGY (0x50...): Leak, Gain
    - RATES (0x80...): Output firing rates
    """
    
    # Register Offsets
    ADDR_CTRL       = 0x00
    ADDR_STATUS     = 0x04
    
    ADDR_X_BASE     = 0x10
    ADDR_W_BASE     = 0x20
    
    ADDR_Y_MIN      = 0x30
    ADDR_Y_MAX      = 0x34
    
    ADDR_STREAM_LEN = 0x40
    ADDR_DT_MS      = 0x44
    ADDR_SCALE_Q16  = 0x48
    
    ADDR_LEAK       = 0x50
    ADDR_GAIN       = 0x54
    
    ADDR_RATE_BASE  = 0x80
    
    def __init__(self, overlay=None, ip_name='sc_neurocore_0'):
        """
        Initialize driver.
        
        Args:
            overlay: Pynq Overlay object (optional if mocking)
            ip_name: Name of the IP in the overlay
        """
        self.mock_mode = overlay is None
        
        if not self.mock_mode:
            if ip_name not in overlay.ip_dict:
                raise ValueError(f"IP '{ip_name}' not found in overlay")
            self.mmio = overlay.ip_dict[ip_name]['mmio'] # Or getattr(overlay, ip_name)
            self.ip = getattr(overlay, ip_name, None)
        else:
            logger.warning("Running in MOCK MODE (No hardware)")
            self.mmio = MockMMIO()
            
    def configure(self, 
                  stream_len=1024, 
                  dt_ms=1, 
                  y_min=-1.0, 
                  y_max=1.0,
                  leak=0.1,
                  gain=1.0):
        """
        Configure global parameters.
        
        Args:
            stream_len: Number of clock cycles per run
            dt_ms: Simulation timestep (informational)
            y_min, y_max: Current range mapping
            leak: Neuron leak rate (0.0 - 1.0)
            gain: Input gain
        """
        self._write(self.ADDR_STREAM_LEN, stream_len)
        self._write(self.ADDR_DT_MS, dt_ms)
        
        self._write(self.ADDR_Y_MIN, self._float_to_q8_8(y_min))
        self._write(self.ADDR_Y_MAX, self._float_to_q8_8(y_max))
        
        # New biological parameters
        self._write(self.ADDR_LEAK, self._float_to_q8_8(leak))
        self._write(self.ADDR_GAIN, self._float_to_q8_8(gain))
        
        # Calculate scale factor for rate estimation
        # Rate = Count / Stream_Len
        # We want Rate_Q16 = (Count * Scale) >> 16 ?? 
        # No, sc_firing_rate_bank does: rate_q16 = count * SCALE_Q16
        # So SCALE_Q16 should be (1.0 / stream_len) * 2^32 ?
        # Or if output is Q16.16 (i.e. 1.0 = 65536), then
        # rate = count / len * 65536
        # SCALE_Q16 = (1/len) * 65536.
        # But wait, sc_firing_rate_bank multiplies: count * SCALE.
        # If count is 100, len is 1000, rate is 0.1.
        # Target Q16.16 is 0.1 * 65536 = 6553.
        # 100 * SCALE = 6553 -> SCALE = 65.53.
        # So SCALE = (1/len) * 2^16.
        # But the module input is 32-bit.
        # If we use 32-bit scale, we might overflow if we are not careful?
        # Let's verify sc_firing_rate_bank logic.
        # rate_q16[i] <= accumulators[i] * SCALE_Q16; 
        # accum is 16-bit. scale is 32-bit. result is 32-bit reg? 
        # No, verilog truncates to LHS size. 
        # We need to ensure we don't overflow.
        # Best approach: Scale is Q16.16 representation of (1/StreamLen).
        
        if stream_len > 0:
            scale = (1.0 / stream_len)
            scale_q16 = int(scale * 65536)
            self._write(self.ADDR_SCALE_Q16, scale_q16)
        
    def set_inputs(self, x_values, weights):
        """
        Set input values and weights.
        
        Args:
            x_values: List of floats [0, 1]
            weights: List of floats [0, 1]
        """
        for i, x in enumerate(x_values):
            if i > 2: break
            addr = self.ADDR_X_BASE + (i * 4)
            self._write(addr, self._float_to_q8_8(x))
            
        for i, w in enumerate(weights):
            if i > 2: break
            addr = self.ADDR_W_BASE + (i * 4)
            self._write(addr, self._float_to_q8_8(w))
            
    def run(self):
        """Start execution and wait for completion."""
        # Pulse Start bit
        self._write(self.ADDR_CTRL, 1)
        self._write(self.ADDR_CTRL, 0)
        
        # Wait for Done
        if self.mock_mode:
            return
            
        while True:
            status = self._read(self.ADDR_STATUS)
            done = (status >> 1) & 1
            if done:
                break
            time.sleep(0.001)
            
    def get_rates(self):
        """Read back firing rates."""
        rates = []
        for i in range(7): # 7 neurons
            addr = self.ADDR_RATE_BASE + (i * 4)
            val_q16 = self._read(addr)
            rates.append(val_q16 / 65536.0)
        return rates

    # Helpers
    def _write(self, offset, value):
        if self.mock_mode:
            self.mmio.write(offset, value)
        else:
            self.mmio.write(offset, int(value)) # Ensure int
            
    def _read(self, offset):
        if self.mock_mode:
            return self.mmio.read(offset)
        else:
            return self.mmio.read(offset)

    def _float_to_q8_8(self, val):
        # 16-bit fixed point: 8 int, 8 frac
        # Range [-128, 127.99]
        clipped = max(-128.0, min(127.99, val))
        return int(clipped * 256) & 0xFFFF


class MockMMIO:
    def __init__(self):
        self.mem = {}
    def write(self, offset, value):
        self.mem[offset] = value
    def read(self, offset):
        return self.mem.get(offset, 0)
