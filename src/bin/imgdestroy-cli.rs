use clap::Parser;

use imgdestroy_rs::ImageArray;

#[derive(Parser)]
#[command(
    version,
    about,
    long_about = None,
    after_help = "Log verbosity: -v debug, -vv trace; RUST_LOG overrides (e.g. RUST_LOG=warn to quiet)."
)]
struct Args {
    #[arg(value_name = "FILE")]
    in_path: String,
    #[arg(short, long, value_name = "PATH", default_value_t = String::from("./out.png"))]
    out_path: String,

    #[arg(short, long, default_value_t = 2)]
    passes: u8,
    #[arg(short='r', long, value_parser = clap::value_parser!(u8).range(..=8))]
    color_reduction: Option<u8>,
    #[arg(short, long)]
    chroma_blowout: Option<u8>,
    #[arg(short, long)]
    luma_blowout: Option<u8>,

    #[arg(long)]
    nosharpen: bool,

    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,
}

fn main() {
    let args = Args::parse();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        match args.verbosity {
            0 => "info",
            1 => "debug",
            _ => "trace",
        },
    ))
    .init();

    let sharpen_or = |img: ImageArray| {
        if !args.nosharpen { img.sharpen() } else { img }
    };
    let reduce_or = |img: ImageArray| {
        if let Some(cr) = args.color_reduction {
            img.reduce_color(cr)
        } else {
            img
        }
    };

    let img: ImageArray = image::open(&args.in_path)
        .expect("Could not read image")
        .into();

    image::RgbImage::from(sharpen_or(
        sharpen_or(reduce_or(img))
            .repeat_compression(args.passes)
            .blowout(
                args.luma_blowout.unwrap_or_default(),
                args.chroma_blowout.unwrap_or_default(),
            ),
    ))
    .save(args.out_path)
    .expect("Failed to save image");
}
