import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron

def generate_stimuli(num_steps=1000):
    """
    Generates random inputs for the neuron.
    """
    inputs = []
    
    # Randomly vary parameters occasionally to test dynamic config
    leak_k = 20 # Small leak
    gain_k = 256 # Gain = 1.0
    
    for i in range(num_steps):
        # I_t: Random current, some spikes
        if np.random.random() < 0.1:
            I_t = int(np.random.uniform(0.5, 2.0) * 256)
        else:
            I_t = 0
            
        noise_in = int(np.random.normal(0, 0.1) * 256)
        
        inputs.append((leak_k, gain_k, I_t, noise_in))
        
    return inputs

def run_model(inputs):
    neuron = FixedPointLIFNeuron()
    results = []
    for leak, gain, i_t, noise in inputs:
        spike, v = neuron.step(leak, gain, i_t, noise)
        results.append((spike, v))
    return results

def main():
    steps = 1000
    print(f"Generating {steps} steps of stimuli...")
    inputs = generate_stimuli(steps)
    
    print("Running Python Fixed-Point Model...")
    expected_results = run_model(inputs)
    
    # Write Stimuli
    with open("stimuli.txt", "w") as f:
        for inp in inputs:
            f.write(f"{inp[0]} {inp[1]} {inp[2]} {inp[3]}\n")
    print("Wrote stimuli.txt")
    
    # Write Expected Results
    with open("results_expected.txt", "w") as f:
        for res in expected_results:
            f.write(f"{res[0]} {res[1]}\n")
    print("Wrote results_expected.txt")
    
    print("\nTo complete Co-Simulation:")
    print("1. Compile the Verilog: sc_lif_neuron.v and tb_sc_lif_neuron.v")
    print("2. Run the simulation. It should read 'stimuli.txt' and produce 'results_verilog.txt'.")
    print("3. Run this script again with argument --check to compare.")
    
    if "--check" in sys.argv:
        if not os.path.exists("results_verilog.txt"):
            print("Error: results_verilog.txt not found.")
            return
            
        print("\nChecking results...")
        with open("results_verilog.txt", "r") as f:
            verilog_lines = f.readlines()
            
        if len(verilog_lines) != len(expected_results):
            print(f"Length mismatch: Exp {len(expected_results)} vs Verilog {len(verilog_lines)}")
            
        mismatches = 0
        for i, (line, expected) in enumerate(zip(verilog_lines, expected_results)):
            parts = line.strip().split()
            v_spike = int(parts[0])
            v_pot = int(parts[1])
            
            if v_spike != expected[0] or v_pot != expected[1]:
                if mismatches < 5:
                    print(f"Mismatch at step {i}: Exp ({expected[0]}, {expected[1]}) vs Got ({v_spike}, {v_pot})")
                mismatches += 1
                
        if mismatches == 0:
            print("SUCCESS: Verilog matches Python Fixed-Point Model exactly!")
        else:
            print(f"FAILURE: {mismatches} mismatches found.")

if __name__ == "__main__":
    main()
