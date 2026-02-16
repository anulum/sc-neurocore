from dataclasses import dataclass


@dataclass
class EnergyMetrics:
    # 45nm CMOS Technology Estimates (Joules)
    E_AND: float = 0.1e-15  # 0.1 fJ
    E_XOR: float = 0.15e-15  # 0.15 fJ
    E_ADD: float = 0.5e-15  # 0.5 fJ (1-bit)
    E_MEM: float = 5.0e-15  # 5 fJ per bit read

    total_ops_and: int = 0
    total_ops_xor: int = 0
    total_bits_mem: int = 0

    def reset(self):
        self.total_ops_and = 0
        self.total_ops_xor = 0
        self.total_bits_mem = 0

    def estimate_energy(self) -> float:
        e_logic = (self.total_ops_and * self.E_AND) + (self.total_ops_xor * self.E_XOR)
        e_mem = self.total_bits_mem * self.E_MEM
        return e_logic + e_mem

    def co2_emission_g(self, carbon_intensity_g_per_kwh=475) -> float:
        # Energy in Joules -> kWh -> Grams CO2
        # 1 J = 2.77e-7 kWh
        kwh = self.estimate_energy() * 2.7778e-7
        return kwh * carbon_intensity_g_per_kwh


# Global Profiler Instance
profiler = EnergyMetrics()


def track_energy(func):
    """Decorator to track energy of a layer call (simulated)."""

    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)

        # Determine 'self' object
        # 1. If func is a bound method, it has __self__
        layer_obj = getattr(func, "__self__", None)

        # 2. If used on class def, args[0] is self
        if layer_obj is None and len(args) > 0:
            # Check if args[0] looks like a layer
            if hasattr(args[0], "n_neurons"):
                layer_obj = args[0]

        if (
            layer_obj
            and hasattr(layer_obj, "n_neurons")
            and hasattr(layer_obj, "n_inputs")
            and hasattr(layer_obj, "length")
        ):
            # Dense Layer Ops:
            ops = layer_obj.n_inputs * layer_obj.n_neurons * layer_obj.length
            profiler.total_ops_and += ops

            # Memory Read
            mem = (layer_obj.n_neurons * layer_obj.n_inputs * layer_obj.length) + (
                layer_obj.n_inputs * layer_obj.length
            )
            profiler.total_bits_mem += mem

        return res

    return wrapper
