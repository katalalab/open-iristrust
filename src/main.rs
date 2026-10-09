use clap::{Parser, Subcommand};
use iristrust::{
    io::{OutputTransaction, load_image, read_json, write_json},
    *,
};
use serde::Serialize;
use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Parser)]
#[command(
    name = "iristrust",
    version,
    about = "Local iris recognition in native Rust"
)]
struct Cli {
    /// Native worker count (also set RAYON_NUM_THREADS when embedding the library).
    #[arg(long,global=true,default_value_t=4,value_parser=clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Extract a template from an infrared eye image.
    Extract {
        image: PathBuf,
        #[arg(long, value_enum)]
        eye: EyeSide,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long,conflicts_with_all=["masks","geometry"])]
        model: Option<PathBuf>,
        #[arg(long,conflicts_with_all=["model","geometry"])]
        masks: Option<PathBuf>,
        #[arg(long,conflicts_with_all=["model","masks"])]
        geometry: Option<PathBuf>,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long)]
        normalized_output: Option<PathBuf>,
        #[arg(long)]
        normalized_mask: Option<PathBuf>,
        /// Bilateral image preprocessing for the ONNX model.
        #[arg(long)]
        denoise: bool,
    },
    /// Encode an already normalized image; white mask pixels mean valid.
    Encode {
        image: PathBuf,
        #[arg(long)]
        mask: Option<PathBuf>,
        #[arg(long, value_enum)]
        eye: EyeSide,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Compare templates; output is masked Hamming distance, not a probability.
    Match {
        first: PathBuf,
        second: PathBuf,
        #[arg(long, default_value_t = 15)]
        max_rotation: usize,
        #[arg(long, default_value_t = 1024)]
        min_overlap: usize,
    },
    /// Align frames, reject distant frames and majority-vote valid phase bits.
    Aggregate {
        #[arg(required=true,num_args=1..)]
        templates: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 15)]
        max_rotation: usize,
        #[arg(long, default_value_t = 1024)]
        min_overlap: usize,
        #[arg(long, default_value_t = 1)]
        min_valid_frames: usize,
        #[arg(long, default_value_t = 0.35)]
        max_distance: f64,
    },
    /// Print template metadata without revealing code/mask words.
    Inspect { template: PathBuf },
    /// Print the default pipeline configuration.
    Config,
    /// Run local image -> template -> match -> aggregation and save artifacts.
    Demo {
        #[arg(short, long)]
        output_dir: PathBuf,
    },
}

fn print_json(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
fn pipeline(path: Option<&Path>) -> Result<IrisPipeline> {
    IrisPipeline::new(match path {
        Some(p) => read_json(p)?,
        None => PipelineConfig::default(),
    })
}
fn load_template(path: &Path) -> Result<IrisTemplate> {
    let t: IrisTemplate = read_json(path)?;
    t.validate()?;
    Ok(t)
}
fn run(cli: Cli) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads as usize)
        .build_global()
        .map_err(|_| Error::InvalidInput("could not initialize native workers".into()))?;
    match cli.command {
        Command::Extract {
            image,
            eye,
            output,
            model,
            masks,
            geometry,
            config,
            report,
            normalized_output,
            normalized_mask,
            denoise,
        } => {
            let output_paths = [
                Some(output.as_path()),
                report.as_deref(),
                normalized_output.as_deref(),
                normalized_mask.as_deref(),
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            let mut outputs = OutputTransaction::new(output_paths)?;
            let image = load_image(&image)?;
            let pipeline = pipeline(config.as_deref())?;
            let extracted = if let Some(path) = geometry {
                let geometry: Geometry = read_json(&path)?;
                pipeline.extract_with_geometry(&image, eye, &geometry, None, None)?
            } else if let Some(path) = masks {
                pipeline.extract_from_masks(&image, eye, &read_json(&path)?)?
            } else {
                #[cfg(feature = "onnx")]
                {
                    let config = iristrust::segmentation::SegmentationConfig {
                        threads: cli.threads as usize,
                        denoise,
                        ..Default::default()
                    };
                    let mut segmenter = iristrust::segmentation::OnnxSegmenter::new(
                        model.unwrap_or_else(|| "models/iris-semseg.onnx".into()),
                        config,
                    )?;
                    pipeline.extract(&image, eye, &mut segmenter)?
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = (model, denoise);
                    return Err(Error::InvalidInput(
                        "ONNX feature is disabled; supply --geometry or --masks".into(),
                    ));
                }
            };
            outputs.stage_json(&output, &extracted.template)?;
            if let Some(path) = report {
                outputs.stage_json(&path, &extracted.report)?;
            }
            if let Some(path) = normalized_output {
                outputs.stage_image(&path, &extracted.normalized.image)?;
            }
            if let Some(path) = normalized_mask {
                let mask = GrayImage::new(
                    extracted.normalized.mask.width,
                    extracted.normalized.mask.height,
                    extracted
                        .normalized
                        .mask
                        .data
                        .iter()
                        .map(|&x| if x { 255 } else { 0 })
                        .collect(),
                )?;
                outputs.stage_image(&path, &mask)?;
            }
            outputs.commit()?;
            print_json(&extracted.report)
        }
        Command::Encode {
            image,
            mask,
            eye,
            output,
            config,
        } => {
            let image = load_image(&image)?;
            let mask = match mask {
                Some(path) => {
                    let mask = load_image(&path)?;
                    Mask::new(
                        mask.width,
                        mask.height,
                        mask.data.iter().map(|&v| v >= 128).collect(),
                    )?
                }
                None => Mask::new(image.width, image.height, vec![true; image.data.len()])?,
            };
            let template = pipeline(config.as_deref())?
                .encode_normalized(&NormalizedIris { image, mask }, eye)?;
            write_json(&output, &template)?;
            inspect(&template)
        }
        Command::Match {
            first,
            second,
            max_rotation,
            min_overlap,
        } => {
            let matcher = Matcher::new(MatcherConfig {
                max_rotation,
                min_overlap_bits: min_overlap,
                normalized: false,
            })?;
            print_json(&matcher.compare(&load_template(&first)?, &load_template(&second)?)?)
        }
        Command::Aggregate {
            templates,
            output,
            report,
            max_rotation,
            min_overlap,
            min_valid_frames,
            max_distance,
        } => {
            let output_paths = [Some(output.as_path()), report.as_deref()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let mut outputs = OutputTransaction::new(output_paths)?;
            let templates = templates
                .iter()
                .map(|p| load_template(p))
                .collect::<Result<Vec<_>>>()?;
            let result = aggregate_with_report(
                &templates,
                &AggregationConfig {
                    max_rotation,
                    min_overlap_bits: min_overlap,
                    min_valid_frames,
                    max_distance,
                },
            )?;
            let summary = serde_json::json!({"accepted_frames":result.accepted_frames,"rejected_frames":result.rejected_frames,"rotations":result.rotations,"valid_bits":result.template.valid_bits()});
            outputs.stage_json(&output, &result.template)?;
            if let Some(path) = report {
                outputs.stage_json(&path, &summary)?;
            }
            outputs.commit()?;
            print_json(&summary)
        }
        Command::Inspect { template } => inspect(&load_template(&template)?),
        Command::Config => print_json(&PipelineConfig::default()),
        Command::Demo { output_dir } => demo(&output_dir),
    }
}
fn inspect(template: &IrisTemplate) -> Result<()> {
    template.validate()?;
    let dimensions: Vec<_> = template
        .planes
        .iter()
        .map(|p| serde_json::json!({"rows":p.rows,"cols":p.cols,"phase_bits":2}))
        .collect();
    print_json(
        &serde_json::json!({"version":template.version,"encoding_profile":template.encoding_profile,"eye_side":template.eye_side,"planes":dimensions,"valid_bits":template.valid_bits()}),
    )
}

fn demo(dir: &Path) -> Result<()> {
    if dir.exists() {
        return Err(Error::InvalidInput(
            "demo output directory must be new".into(),
        ));
    }
    std::fs::create_dir_all(dir)?;
    let center = Point { x: 160.0, y: 120.0 };
    let geometry = geometry_from_circles(center, 28.0, center, 85.0, 512)?;
    let data = (0..320 * 240)
        .map(|i| {
            let x = (i % 320) as f64 - center.x;
            let y = (i / 320) as f64 - center.y;
            let rho = x.hypot(y);
            let angle = y.atan2(x);
            if rho < 28.0 {
                15
            } else if rho < 90.0 {
                (110.0
                    + 10.0 * (angle * 17.0 + rho * 0.2).sin()
                    + 6.0 * (angle * 31.0 - rho * 0.15).cos()) as u8
            } else {
                180
            }
        })
        .collect();
    let image = GrayImage::new(320, 240, data)?;
    let pipeline = IrisPipeline::new(PipelineConfig::default())?;
    let extracted = pipeline.extract_with_geometry(&image, EyeSide::Left, &geometry, None, None)?;
    let decoded = extracted.template.clone();
    let rotated = rotate_template(&decoded, 5)?;
    let matcher = Matcher::new(MatcherConfig::default())?;
    let comparison = matcher.compare(&decoded, &rotated)?;
    let aggregate = aggregate_with_report(
        &[decoded.clone(), rotated.clone()],
        &AggregationConfig::default(),
    )?;
    let result = serde_json::json!({"image":"eye.png","template":"template.json","valid_bits":decoded.valid_bits(),"rotation_match":comparison,"aggregate_match":matcher.compare(&decoded,&aggregate.template)?,"extraction":extracted.report});
    let eye = dir.join("eye.png");
    let geometry_path = dir.join("geometry.json");
    let normalized = dir.join("normalized.png");
    let template = dir.join("template.json");
    let rotated_path = dir.join("rotated.json");
    let aggregate_path = dir.join("aggregate.json");
    let report = dir.join("report.json");
    let mut outputs = OutputTransaction::new([
        &eye,
        &geometry_path,
        &normalized,
        &template,
        &rotated_path,
        &aggregate_path,
        &report,
    ])?;
    outputs.stage_image(&eye, &image)?;
    outputs.stage_json(&geometry_path, &geometry)?;
    outputs.stage_image(&normalized, &extracted.normalized.image)?;
    outputs.stage_json(&template, &decoded)?;
    outputs.stage_json(&rotated_path, &rotated)?;
    outputs.stage_json(&aggregate_path, &aggregate.template)?;
    outputs.stage_json(&report, &result)?;
    outputs.commit()?;
    print_json(&result)
}
fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
