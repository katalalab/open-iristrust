//! Run `cargo run --release --example benchmark -- 100`.
use iristrust::{
    filters::{FilterBank, FilterBankConfig},
    *,
};
use serde::Deserialize;
use std::{hint::black_box, time::Instant};
#[derive(Deserialize)]
struct Reference {
    normalized: NormalizedIris,
}
fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction).round() as usize]
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let iterations = std::env::args()
        .nth(1)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(100);
    if !(10..=10000).contains(&iterations) {
        return Err("iterations must be 10..=10000".into());
    }
    let workers = 4;
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()?;
    let reference: Reference =
        serde_json::from_str(include_str!("../tests/fixtures/upstream-features.json"))?;
    let bank = FilterBank::new(FilterBankConfig::default())?;
    let template = bank.encode(&reference.normalized, EyeSide::Left)?;
    let shifted = rotate_template(&template, 7)?;
    let matcher = Matcher::new(MatcherConfig::default())?;
    for _ in 0..5 {
        black_box(bank.encode(&reference.normalized, EyeSide::Left)?);
        black_box(matcher.compare(&template, &shifted)?);
    }
    let mut extraction = Vec::with_capacity(iterations);
    let mut matching = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(bank.encode(black_box(&reference.normalized), EyeSide::Left)?);
        extraction.push(start.elapsed().as_secs_f64() * 1000.0);
        let start = Instant::now();
        black_box(matcher.compare(black_box(&template), black_box(&shifted))?);
        matching.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "iterations":iterations,"workers":workers,"normalized_width":reference.normalized.image.width,
            "normalized_height":reference.normalized.image.height,"filters":2,"probe_rows":16,"probe_cols":256,
            "rotations_tested":31,"valid_bits":template.valid_bits(),
            "gabor_encode_ms_p50":percentile(&mut extraction,0.5),"gabor_encode_ms_p95":percentile(&mut extraction,0.95),
            "rotational_match_us_p50":percentile(&mut matching,0.5),"rotational_match_us_p95":percentile(&mut matching,0.95)
        }))?
    );
    Ok(())
}
