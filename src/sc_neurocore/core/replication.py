
import os
import shutil
import sys
from dataclasses import dataclass

@dataclass
class VonNeumannProbe:
    """
    Simulates a self-replicating code entity.
    Can copy the sc-neurocore source and its own state to a new 'host'.
    """
    probe_id: int
    
    def replicate(self, destination_dir: str):  # pragma: no cover
        """
        Quine-like behavior: Copies the library source to a new location.
        Uses shutil.copytree which can fail on platform-specific special files.
        """
        print(f"Probe {self.probe_id}: Replicating to {destination_dir}...")

        # 1. Identify source root
        # (Assuming we are in src/sc_neurocore/core/replication.py)
        src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # 2. Copy source files
        # Only copy the library 'sc_neurocore' folder
        lib_src = os.path.join(src_root, 'sc_neurocore')
        lib_dst = os.path.join(destination_dir, 'sc_neurocore')

        if os.path.exists(lib_dst):
            shutil.rmtree(lib_dst)

        shutil.copytree(lib_src, lib_dst)

        # 3. Create a launch script in the new destination
        launch_script = os.path.join(destination_dir, 'launch_probe.py')
        with open(launch_script, 'w') as f:
            f.write("import sys\nimport os\n")
            f.write("sys.path.append(os.path.abspath('.'))\n")
            f.write("from sc_neurocore.core.replication import VonNeumannProbe\n")
            f.write(f"p = VonNeumannProbe(probe_id={self.probe_id + 1})\n")
            f.write("print('Probe sequence initiated in new sector.')\n")

        print(f"Probe {self.probe_id}: Success. New generation ready.")
