use std::io::Cursor;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::FromSample;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

// Taken from https://github.com/huggingface/candle/blob/main/candle-examples/examples/whisper/pcm_decode.rs
fn conv<T>(samples: &mut Vec<f32>, data: &symphonia::core::audio::AudioBuffer<T>)
where
    T: symphonia::core::sample::Sample,
    f32: FromSample<T>,
{
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)));
}

pub fn pcm_decode(cursor: Cursor<Box<[u8]>>) -> anyhow::Result<(Vec<f32>, u32)> {
    // Create the media source stream.
    let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());

    // Create a probe hint using the file's extension. [Optional]
    let hint = symphonia::core::probe::Hint::new();

    // Use the default options for metadata and format readers.
    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Use the default options for the decoder.
    let dec_opts = DecoderOptions::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .expect("unsupported codec");
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    // The decode loop.
    while let Ok(packet) = format.next_packet() {
        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, &data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, &data),
        }
    }
    Ok((pcm_data, sample_rate))
}
