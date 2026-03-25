# Tutorial 70: Spike-Train Compression Codec

Compress spike trains 50-200x for BCI telemetry.

```python
from sc_neurocore.spike_codec import SpikeCodec

codec = SpikeCodec(mode="lossless")
data, result = codec.compress(spike_raster)
print(result.summary())  # "SpikeCodec (lossless): 87.3x compression"
reconstructed = codec.decompress(data, T=1000, N=96)
```
