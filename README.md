# imgdestroy-rs

imgdestroy-rs is a Rust command-line tool and library that degrades images
through repeat compression, sharpening and crushing. Both a library (`imgdestroy-rs`) and a
CLI binary (`imgdestroy-cli`) are provided.

## Installation

Build from source with Cargo:

```sh
cargo install --path .
```

This installs the `imgdestroy-cli` binary into `~/.cargo/bin`. For development
or library use, clone the repository and run `cargo build`.

## Usage

```sh
imgdestroy-cli [OPTIONS] <FILE>
```

`<FILE>` is the input image in any format the `image` crate supports (PNG,
JPEG, WebP, TIFF).

Options:

| Option | Description | Default |
| --- | --- | --- |
| `-o, --out-path <PATH>` | Where the result is written | `./out.png` |
| `-p, --passes <PASSES>` | How many times the image is re-compressed (capped at 50) | `2` |
| `-r, --color-reduction <0-8>` | Posterize: quantize each channel to fewer levels (higher = fewer colors) | off |
| `-l, --luma-blowout <AMOUNT>` | Amplify brightness contrast | off |
| `-c, --chroma-blowout <AMOUNT>` | Amplify color saturation | off |
| `--nosharpen` | Skip the unsharp masking passes | sharpening on |

Blowout amounts work like strengths: `0` leaves the image alone, small
positive values exaggerate luma contrast (luma) or saturation (chroma), and
larger values push the image harder toward fully blown highlights and
saturated colors.

The output format follows the extension of `--out-path`; the default
`./out.png` writes PNG.

### Examples

Standard deep-fry: two compression passes with default sharpening:

```sh
imgdestroy-cli photo.jpg
```
Only compress, no sharpening or other effects:

```sh
imgdestroy-cli photo.jpg --nosharpen
```

Severe degradation:

```sh
imgdestroy-cli photo.jpg -o destroyed.png --passes 5 --color-reduction 3 --luma-blowout 2 --chroma-blowout 2
```
