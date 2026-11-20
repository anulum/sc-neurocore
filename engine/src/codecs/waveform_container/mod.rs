// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Complete waveform container codec

//! Standalone WFC2 compression and reconstruction on row-major electrode samples.

mod background;
mod compression;
mod snippets;
mod wavelet;
mod wire;

use super::isi::{self, Entropy};
use super::{waveform_codec as kernels, waveform_stream as stream};
use wire::{Header, Reader};

/// The default maximum number of values declared by a decoded packet.
pub const MAX_DECODE_SAMPLES: usize = 1 << 28;

/// Payload mode stored in the WFC2 header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum WaveformMode {
    /// Detected spikes and an approximate continuous waveform.
    Full = 0,
    /// Detected spikes and approximate snippets on a zero background.
    Waveform = 1,
    /// Exact positions of the detected spikes.
    Spike = 2,
    /// Original int16 samples encoded with a per-channel Rice predictor.
    LosslessRice = 3,
    /// Original int16 samples encoded with the lossless streaming predictor.
    Lossless = 4,
}

impl WaveformMode {
    /// Parse the mode name used by the Python and Rust consumers.
    pub fn parse(mode: &str) -> Result<Self, String> {
        match mode {
            "full" => Ok(Self::Full),
            "waveform" => Ok(Self::Waveform),
            "spike" => Ok(Self::Spike),
            "lossless_rice" => Ok(Self::LosslessRice),
            "lossless" => Ok(Self::Lossless),
            _ => Err(format!("unknown WaveformCodec mode {mode:?}")),
        }
    }

    /// Return the public mode name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Waveform => "waveform",
            Self::Spike => "spike",
            Self::LosslessRice => "lossless_rice",
            Self::Lossless => "lossless",
        }
    }

    /// Whether this mode decodes every original int16 sample exactly.
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::Lossless | Self::LosslessRice)
    }

    fn from_byte(mode: u8) -> Result<Self, String> {
        match mode {
            0 => Ok(Self::Full),
            1 => Ok(Self::Waveform),
            2 => Ok(Self::Spike),
            3 => Ok(Self::LosslessRice),
            4 => Ok(Self::Lossless),
            _ => Err(format!("unknown WaveformCodec mode byte {mode}")),
        }
    }
}

/// Byte compression for snippets and background sections.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SectionCompression {
    /// Write a zlib DEFLATE stream.
    #[default]
    Zlib,
    /// Write a Zstandard frame with the optional waveform-zstd Cargo feature.
    Zstandard,
}

/// Encoder configuration shared by the five waveform modes.
#[derive(Clone, Debug)]
pub struct WaveformConfig {
    /// Mode of the resulting container.
    pub mode: WaveformMode,
    /// Positive multiplier for the channel noise threshold.
    pub threshold_sigma: f64,
    /// Padded sample width of every snippet, in 1..=255.
    pub snippet_samples: usize,
    /// Maximum template library size, in 1..=65535.
    pub max_templates: usize,
    /// Minimum normalised template correlation, in 0..=1.
    pub template_threshold: f64,
    /// Signed background code width, in 1..=8.
    pub quantize_bits: u8,
    /// Fixed ADC width of the streaming lossless mode, or automatic detection.
    pub adc_bits: Option<u32>,
    /// Byte compression for the lossy sections.
    pub compression: SectionCompression,
    /// Denoise the full-mode background with the symmetric db4 transform.
    pub denoise_background: bool,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            mode: WaveformMode::Full,
            threshold_sigma: 4.5,
            snippet_samples: 48,
            max_templates: 16,
            template_threshold: 0.9,
            quantize_bits: 6,
            adc_bits: None,
            compression: SectionCompression::Zlib,
            denoise_background: false,
        }
    }
}

impl WaveformConfig {
    fn header(&self, rows: usize, channels: usize) -> Result<Header, String> {
        wire::validate_quantization(self.snippet_samples, self.quantize_bits)?;
        if !self.threshold_sigma.is_finite() || self.threshold_sigma <= 0.0 {
            return Err("threshold_sigma must be positive and finite".into());
        }
        if !(0.0..=1.0).contains(&self.template_threshold) {
            return Err("template_threshold must be in [0, 1]".into());
        }
        if !(1..=65535).contains(&self.max_templates) {
            return Err("max_templates must be in 1..=65535".into());
        }
        if self.adc_bits.is_some_and(|bits| !(1..=16).contains(&bits)) {
            return Err("adc_bits must be in 1..=16".into());
        }
        Ok(Header {
            rows,
            channels,
            templates: 0,
            snippets: 0,
            snippet_samples: self.snippet_samples,
            quantize_bits: self.quantize_bits,
            spike_length_bits: 0,
            mode: self.mode,
            legacy: false,
        })
    }
}

/// A complete packet and the lengths and counts reported by the encoder.
#[derive(Debug)]
pub struct EncodedWaveform {
    /// WFC2 header, sections and CRC-32.
    pub packet: Vec<u8>,
    /// Number of events in the encoded raster.
    pub spikes: usize,
    /// Number of templates selected by the encoder.
    pub templates: usize,
    /// Number of bytes in the embedded ISI packet.
    pub spike_bytes: usize,
    /// Number of bytes in the snippet section.
    pub snippet_bytes: usize,
    /// Number of bytes in the background section.
    pub background_bytes: usize,
}

/// Exact or reconstructed row-major sample values, depending on the mode.
#[derive(Debug)]
pub enum WaveformSamples {
    /// Original integer values.
    Exact(Vec<i16>),
    /// Lossy float32 reconstruction.
    Approximate(Vec<f32>),
}

/// Reconstructed output and metadata from one waveform packet.
#[derive(Debug)]
pub struct DecodedWaveform {
    /// Mode stored in the header.
    pub mode: WaveformMode,
    /// Number of time rows.
    pub rows: usize,
    /// Number of columns.
    pub channels: usize,
    /// Binary row-major detected raster, absent for the two lossless modes.
    pub raster: Option<Vec<i8>>,
    /// Reconstructed waveform, absent for spike mode.
    pub samples: Option<WaveformSamples>,
}

/// Encode spike, waveform or full mode from a finite row-major float32 matrix.
///
/// Returns an error for invalid options, dimensions, values or wire sizes.
pub fn encode_lossy(
    waveform: &[f32],
    rows: usize,
    channels: usize,
    config: &WaveformConfig,
) -> Result<EncodedWaveform, String> {
    let mut header = config.header(rows, channels)?;
    let cells = wire::dimensions(rows, channels, usize::MAX / 2)?;
    if header.mode.is_lossless() {
        return Err("lossless WaveformCodec modes require int16 input".into());
    }
    if waveform.len() != cells || waveform.iter().any(|value| !value.is_finite()) {
        return Err("waveform must be a finite non-empty (time, channel) matrix".into());
    }
    let thresholds =
        kernels::noise_thresholds(waveform, rows, channels, -config.threshold_sigma as f32);
    let times = kernels::detect_spikes(
        waveform,
        rows,
        channels,
        &thresholds,
        config.snippet_samples / 2,
    );
    let count: usize = times.iter().map(Vec::len).sum();
    if count > u16::MAX.into() {
        return Err("spike snippet count exceeds WaveformCodec header capacity".into());
    }
    let mut raster = vec![0; cells];
    for (ch, channel_times) in times.iter().enumerate() {
        for &time in channel_times {
            raster[time * channels + ch] = 1;
        }
    }
    let snippet = snippets::encode(waveform, rows, channels, &times, count, config)?;
    let (spikes, _) = isi::encode_raster(&raster, rows, channels, 1, Entropy::Auto)?;
    let background = if header.mode == WaveformMode::Full {
        background::encode(waveform, rows, channels, &times, config)?
    } else {
        Vec::new()
    };
    header.templates = snippet.templates;
    header.snippets = count;
    header.spike_length_bits = (usize::BITS - spikes.len().leading_zeros()) as u8;
    Ok(EncodedWaveform {
        packet: header.pack(&[&spikes, &snippet.bytes, &background])?,
        spikes: count,
        templates: snippet.templates,
        spike_bytes: spikes.len(),
        snippet_bytes: snippet.bytes.len(),
        background_bytes: background.len(),
    })
}

/// Encode either lossless mode from row-major ADC values.
///
/// Returns an error for invalid options, dimensions, ADC levels or wire sizes.
pub fn encode_lossless(
    waveform: &[i16],
    rows: usize,
    channels: usize,
    config: &WaveformConfig,
) -> Result<EncodedWaveform, String> {
    let header = config.header(rows, channels)?;
    if waveform.len() != wire::dimensions(rows, channels, usize::MAX / 2)? {
        return Err("window size does not match samples x channels".into());
    }
    let body = match header.mode {
        WaveformMode::Lossless => {
            let width = config
                .adc_bits
                .unwrap_or_else(|| stream::detect_adc_bits(waveform));
            let mut body = vec![width as u8];
            body.extend(stream::StreamEncoder::new(channels, width, 1)?.encode(waveform, rows)?);
            body
        }
        WaveformMode::LosslessRice => {
            let mut body = Vec::new();
            let mut column = Vec::with_capacity(rows);
            for channel in 0..channels {
                column.extend(
                    waveform
                        .chunks_exact(channels)
                        .map(|row| i64::from(row[channel])),
                );
                wire::write_section(&mut body, &kernels::lossless_encode_channel(&column))?;
                column.clear();
            }
            body
        }
        _ => return Err("int16 lossless encoding requires a lossless WaveformCodec mode".into()),
    };
    Ok(EncodedWaveform {
        packet: header.pack(&[&body])?,
        spikes: 0,
        templates: 0,
        spike_bytes: 0,
        snippet_bytes: 0,
        background_bytes: 0,
    })
}

/// Read a WFC2 packet or a legacy WFCX spike raster.
///
/// Returns an error before decoding if the declared number of values exceeds
/// `max_samples`, or if the packet has invalid framing or content.
pub fn decode(packet: &[u8], max_samples: usize) -> Result<DecodedWaveform, String> {
    let (header, reader) = Header::read(packet, max_samples)?;
    let (raster, samples) = if header.mode.is_lossless() {
        (
            None,
            Some(WaveformSamples::Exact(decode_lossless(header, reader)?)),
        )
    } else {
        decode_lossy(header, reader)?
    };
    Ok(DecodedWaveform {
        mode: header.mode,
        rows: header.rows,
        channels: header.channels,
        raster,
        samples,
    })
}

fn decode_lossless(header: Header, mut reader: Reader<'_>) -> Result<Vec<i16>, String> {
    let mut body = Reader::new(reader.section()?);
    reader.finish()?;
    if header.mode == WaveformMode::Lossless {
        let width = body
            .u8()
            .map_err(|_| "lossless stream has an invalid ADC width")?;
        if !(1..=16).contains(&width) {
            return Err("lossless stream has an invalid ADC width".into());
        }
        return stream::StreamDecoder::new(header.channels, u32::from(width), 1)?
            .decode(body.rest(), header.rows);
    }
    let sections: Vec<_> = (0..header.channels)
        .map(|_| body.section())
        .collect::<Result<_, _>>()?;
    body.finish()?;
    let mut samples = zeroed(header.rows * header.channels, 0)?;
    for (channel, section) in sections.into_iter().enumerate() {
        let column = kernels::lossless_decode_channel(section, header.rows)?;
        for (row, value) in samples.chunks_exact_mut(header.channels).zip(column) {
            row[channel] =
                i16::try_from(value).map_err(|_| "lossless samples outside the int16 range")?;
        }
    }
    Ok(samples)
}

type LossyOutput = (Option<Vec<i8>>, Option<WaveformSamples>);

fn decode_lossy(header: Header, mut reader: Reader<'_>) -> Result<LossyOutput, String> {
    let spikes = reader.section()?;
    let snippets = reader.section()?;
    let background = reader.section()?;
    reader.finish()?;
    let raster = isi::decode_raster(spikes, header.rows, header.channels, 1)?;
    let count = raster.iter().filter(|&&value| value != 0).count();
    if count != header.snippets {
        return Err("snippet count does not match the decoded spike raster".into());
    }
    if header.mode == WaveformMode::Spike {
        if !header.legacy && (!snippets.is_empty() || !background.is_empty()) {
            return Err("spike mode requires empty snippet and background sections".into());
        }
        return Ok((Some(raster), None));
    }
    let mut waveform = if header.mode == WaveformMode::Full {
        background::decode(background, header)?
    } else {
        if !background.is_empty() {
            return Err("waveform mode must have an empty background section".into());
        }
        zeroed(header.rows * header.channels, 0.0)?
    };
    snippets::apply(snippets, header, &raster, &mut waveform)?;
    Ok((Some(raster), Some(WaveformSamples::Approximate(waveform))))
}

fn zeroed<T: Clone>(values: usize, zero: T) -> Result<Vec<T>, String> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values)
        .map_err(|_| "WaveformCodec output allocation failed")?;
    output.resize(values, zero);
    Ok(output)
}
