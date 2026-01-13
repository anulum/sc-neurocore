
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.verification.safety import CodeSafetyVerifier
from sc_neurocore.meta.dao import AgentDAO
from sc_neurocore.security.immune import DigitalImmuneSystem

def run_governance_demo():
    print("--- GOVERNANCE & SAFETY DEMO ---")
    
    # 1. Code Safety
    print("\n[1] Testing Code Safety Verifier...")
    verifier = CodeSafetyVerifier()
    unsafe_code = "import os; os.system('echo dangerous')"
    safe_code = "x = 1 + 1; print(x)"
    
    print("    Analyzing Unsafe Code...")
    verifier.verify_code_safety(unsafe_code)
    
    print("    Analyzing Safe Code...")
    verifier.verify_code_safety(safe_code)
    
    # 2. DAO
    print("\n[2] Testing Agent DAO...")
    dao = AgentDAO(agent_id="Agent-Alpha")
    pid = dao.create_proposal("Upgrade to v2.1")
    dao.vote(pid, approve=True)
    dao.finalize_proposal(pid)
    
    # 3. Immune System
    print("\n[3] Testing Digital Immune System...")
    immune = DigitalImmuneSystem()
    
    # Train
    normal = np.array([0.5, 0.5, 0.5])
    immune.train_self(normal)
    
    # Scan Healthy
    print(f"    Scanning Normal State: {normal}")
    immune.scan(normal)
    
    # Scan Anomaly
    infected = np.array([0.9, 0.1, 0.0]) # High deviation
    print(f"    Scanning Anomalous State: {infected}")
    immune.scan(infected)

if __name__ == "__main__":
    run_governance_demo()
