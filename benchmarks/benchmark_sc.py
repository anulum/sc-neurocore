import time
import numpy as np
from src.sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
from src.sc_neurocore.utils.bitstreams import BitstreamEncoder, BitstreamAverager

def benchmark_bitstream_generation():
    print("Benchmarking Bitstream Generation (Bernoulli).")
    encoder = BitstreamEncoder(x_min=0.0, x_max=1.0, length=10000, mode="bernoulli")
    
    start_time = time.time()
    for _ in range(100):
        encoder.encode(0.5)
    end_time = time.time()
    
    print(f"Time for 100 encodings (L=10000): {end_time - start_time:.4f}s")

    print("\nBenchmarking Bitstream Generation (Sobol).")
    encoder_sobol = BitstreamEncoder(x_min=0.0, x_max=1.0, length=10000, mode="sobol")
    
    start_time = time.time()
    for _ in range(100):
        encoder_sobol.encode(0.5)
    end_time = time.time()
    
    print(f"Time for 100 encodings (L=10000): {end_time - start_time:.4f}s")


def benchmark_neuron():
    print("\nBenchmarking Neuron Step...")
    neuron = StochasticLIFNeuron(seed=42)
    
    start_time = time.time()
    for _ in range(100000):
        neuron.step(input_current=1.5)
    end_time = time.time()
    
    print(f"Time for 100k steps: {end_time - start_time:.4f}s")

    print("\nBenchmarking Neuron Process Bitstream...")
    neuron_bs = StochasticLIFNeuron(seed=42, refractory_period=2)
    bits = np.random.randint(0, 2, 100000).astype(np.uint8)
    
    start_time = time.time()
    spikes = neuron_bs.process_bitstream(bits, input_scale=1.5)
    end_time = time.time()
    
    print(f"Time for processing 100k bits: {end_time - start_time:.4f}s")
    print(f"Spike count: {np.sum(spikes)}")


def benchmark_averager():
    print("\nBenchmarking Averager...")
    averager = BitstreamAverager(window=1000)
    
    # Fill buffer
    for _ in range(1000):
        averager.push(1)
        
    start_time = time.time()
    for i in range(100000):
        averager.push(i % 2)
        val = averager.estimate()
    end_time = time.time()
    
    print(f"Time for 100k push+estimates: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    benchmark_bitstream_generation()
    benchmark_neuron()
    benchmark_averager()