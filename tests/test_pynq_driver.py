# tests/test_pynq_driver.py

import pytest
from sc_neurocore.drivers.sc_neurocore_driver import SCNeuroCoreDriver

def test_driver_configuration():
    """Verify driver writes correct values to registers."""
    driver = SCNeuroCoreDriver(overlay=None) # Mock mode
    
    # Configure with specific biology
    leak = 0.5
    gain = 2.0
    driver.configure(stream_len=100, leak=leak, gain=gain)
    
    # Check registers
    # ADDR_LEAK = 0x50, ADDR_GAIN = 0x54
    # Q8.8 format: 0.5 -> 128, 2.0 -> 512
    
    assert driver._read(0x50) == 128
    assert driver._read(0x54) == 512
    
    print("Driver configuration verified.")

def test_driver_run():
    """Verify run sequence."""
    driver = SCNeuroCoreDriver(overlay=None)
    driver.run()
    
    # In mock mode, it just returns. 
    # Real test would need to mock the status register change.
    
if __name__ == "__main__":
    test_driver_configuration()
    test_driver_run()
