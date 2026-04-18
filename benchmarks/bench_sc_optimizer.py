# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-Optimizer Benchmark

import time
from sc_neurocore.optimizer.sc_optimizer import SCOptimizer, HardwareBudget, LayerProfile

def benchmark():
    budget = HardwareBudget(max_luts=10_000_000, max_power_mw=50000.0) # Large FPGA / ASIC target
    optimizer = SCOptimizer(budget)
    
    # 50-layer deep network (ResNet-50 equivalent scale in layers)
    network = [
        LayerProfile(id=f"Layer_{i}", mac_count=10000 + (i*100), is_critical_path=(i%10==0))
        for i in range(50)
    ]
    
    print("--- SC-NeuroCore: SC-Optimizer Benchmark ---")
    start = time.perf_counter()
    
    # Generate current_config
    current_config = {}
    total_luts = 0
    total_power = 0.0
    candidates_per_layer = {layer.id: optimizer._generate_candidates(layer) for layer in network}
    
    for layer in network:
        cheapest = min(candidates_per_layer[layer.id], key=lambda c: c.luts_used)
        current_config[layer.id] = cheapest
        total_luts += cheapest.luts_used
        total_power += cheapest.power_used

    upgraded = True
    while upgraded:
        upgraded = False
        best_upgrade = None
        best_layer_id = None
        max_score_gain_per_lut = 0.0
        
        for layer in network:
            curr = current_config[layer.id]
            for cand in candidates_per_layer[layer.id]:
                if cand.accuracy_score > curr.accuracy_score:
                    lut_diff = cand.luts_used - curr.luts_used
                    pwr_diff = cand.power_used - curr.power_used
                    
                    if total_luts + lut_diff <= optimizer.budget.max_luts and total_power + pwr_diff <= optimizer.budget.max_power_mw:
                        score_gain = cand.accuracy_score - curr.accuracy_score
                        if layer.is_critical_path:
                            score_gain *= 2.0 
                            
                        efficiency = score_gain / lut_diff if lut_diff > 0 else float('inf')
                        if efficiency > max_score_gain_per_lut:
                            max_score_gain_per_lut = efficiency
                            best_upgrade = cand
                            best_layer_id = layer.id
                            
        if best_upgrade:
            curr = current_config[best_layer_id]
            total_luts += (best_upgrade.luts_used - curr.luts_used)
            total_power += (best_upgrade.power_used - curr.power_used)
            current_config[best_layer_id] = best_upgrade
            upgraded = True

    end = time.perf_counter()
    
    if current_config:
        total_luts = sum(c.luts_used for c in current_config.values())
        total_pwr = sum(c.power_used for c in current_config.values())
        avg_acc = sum(c.accuracy_score for c in current_config.values()) / len(current_config)
        
        print(f"Network Depth: 50 layers")
        print(f"Optimization Time: {(end - start) * 1000:.4f} ms")
        print(f"Final Configuration: {total_luts} LUTs, {total_pwr:.1f} mW, Avg Acc: {avg_acc:.4f}")
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    benchmark()
