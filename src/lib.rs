use log::info;
use ndarray::{Zip, prelude::*};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};
use ndrustfft::{nddct2, nddct2_inplace, nddct3_inplace};

const BASE_LUMA_Q: [[u16; 8]; 8] = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
];

const BASE_CHROMA_Q: [[u16; 8]; 8] = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
];

#[derive(Debug)]
pub enum ImgDestroyError {
    BufferTooSmallError,
    MemoryLayoutError,
    ShapeError,
}

impl std::fmt::Display for ImgDestroyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooSmallError => write!(f, "Buffer too small!"),
            Self::MemoryLayoutError => {
                write!(f, "Cannot obtain array slice, nonstandard memory layout")
            }
            Self::ShapeError => write!(f, "Cannot convert vector to array, wrong shape"),
        }
    }
}

impl std::error::Error for ImgDestroyError {}

fn plane2array(img: image::GrayImage) -> Result<Array2<u8>, ImgDestroyError> {
    Array2::from_shape_vec(
        (img.height() as usize, img.width() as usize),
        img.into_raw(),
    )
    .map_err(|_e| ImgDestroyError::ShapeError)
}

fn plane2img(array: &ArrayView2<u8>) -> Result<image::GrayImage, ImgDestroyError> {
    let array = array.as_standard_layout();
    match array.as_slice() {
        Some(slice) => {
            image::GrayImage::from_raw(array.dim().1 as u32, array.dim().0 as u32, slice.to_vec())
                .ok_or(ImgDestroyError::BufferTooSmallError)
        }
        None => Err(ImgDestroyError::MemoryLayoutError),
    }
}

pub fn rgb2yuv(rgb: &Array3<u8>) -> Array3<f32> {
    let mut yuv: Array3<f32> = Array3::default(rgb.raw_dim());
    let mut rgb: Array3<f32> = rgb.mapv(f32::from);

    let (r, g, b) = rgb.multi_slice_mut((s![.., .., 0], s![.., .., 1], s![.., .., 2]));
    let (y, u, v) = yuv.multi_slice_mut((s![.., .., 0], s![.., .., 1], s![.., .., 2]));

    Zip::from(&r)
        .and(&g)
        .and(&b)
        .map_assign_into(y, |r, g, b| 0.299 * r + 0.587 * g + 0.114 * b);
    Zip::from(&r)
        .and(&g)
        .and(&b)
        .map_assign_into(u, |r, g, b| -0.169 * r - 0.331 * g + 0.500 * b + 128.);
    Zip::from(&r)
        .and(&g)
        .and(&b)
        .map_assign_into(v, |r, g, b| 0.500 * r - 0.419 * g - 0.081 * b + 128.);

    yuv
}

pub fn yuv2rgb(yuv: &Array3<f32>) -> Array3<f32> {
    let mut rgb: Array3<f32> = Array3::default(yuv.raw_dim());
    let mut yuv = yuv - 128.;

    let (r, g, b) = rgb.multi_slice_mut((s![.., .., 0], s![.., .., 1], s![.., .., 2]));
    let (y, u, v) = yuv.multi_slice_mut((s![.., .., 0], s![.., .., 1], s![.., .., 2]));

    Zip::from(&y)
        .and(&v)
        .map_assign_into(r, |y, v| y + 1.402 * v + 128.);
    Zip::from(&y)
        .and(&u)
        .and(&v)
        .map_assign_into(g, |y, u, v| y - 0.334 * u - 0.714 * v + 128.);
    Zip::from(&y)
        .and(&u)
        .map_assign_into(b, |y, u| y + 1.772 * u + 128.);

    rgb
}

#[derive(Debug, Clone, Copy)]
pub struct AberrationCoords {
    compress_x: u32,
    compress_y: u32,
    compress_c: usize,
    expand_x: u32,
    expand_y: u32,
    expand_c: usize,
}

impl AberrationCoords {
    pub fn new(
        compress_x: u32,
        compress_y: u32,
        compress_channel: usize,
        expand_x: u32,
        expand_y: u32,
        expand_channel: usize,
    ) -> Self {
        AberrationCoords {
            compress_x,
            compress_y,
            compress_c: compress_channel.clamp(0, 2),
            expand_x,
            expand_y,
            expand_c: expand_channel.clamp(0, 2),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageArray {
    image: Array3<f32>,
}

impl ImageArray {
    pub fn new(height: usize, width: usize) -> Self {
        ImageArray {
            image: Array3::zeros((height, width, 3)),
        }
    }

    fn get_u8(&self) -> Array3<u8> {
        self.image.mapv(|v| v.clamp(0., 255.) as u8)
    }
    fn get_f32(&self) -> Array3<f32> {
        self.image.clone()
    }

    pub fn sharpen(mut self) -> Self {
        if let Some(mat) = self
            .image
            .axis_iter(Axis(2))
            .map(|mat| {
                mat.conv(
                    &array![[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]],
                    ConvMode::Same,
                    PaddingMode::Replicate,
                )
                .unwrap_or(mat.to_owned())
                .insert_axis(Axis(2))
            })
            .reduce(|acc, mat| ndarray::concatenate![Axis(2), acc, mat])
        {
            self.image = mat;
        }
        self
    }

    pub fn reduce_color(mut self, factor: u8) -> Self {
        let f: f32 = 2_u16.pow(factor.clamp(0, 8).into()).clamp(1, 255).into();
        self.image = (self.image / f).round() * f;
        self
    }

    pub fn blowout(mut self, luma_factor: u8, chroma_factor: u8) -> Self {
        if luma_factor == 0 && chroma_factor == 0 {
            return self;
        }

        let mut yuv = rgb2yuv(&self.get_u8());
        let luma_factor: f32 = luma_factor.into();
        let chroma_factor: f32 = chroma_factor.into();

        let chroma_premult = yuv
            .slice(s![.., .., 0])
            .mapv(|v| chroma_factor * (128. - (127. - v).abs()) / 128.);

        if luma_factor > 0. {
            yuv.slice_mut(s![.., .., 0])
                .map_inplace(|v| *v = (luma_factor * (*v - 128.)).clamp(-128., 128.) + 128.0);
        }
        if chroma_factor > 0. {
            let arr = Zip::from(yuv.slice(s![.., .., 1..]))
                .and(ndarray::stack![Axis(2), chroma_premult, chroma_premult].view())
                .par_map_collect(|v, c| (c * (v - 128.)).clamp(-128., 128.) + 128.);
            yuv.slice_mut(s![.., .., 1..]).assign(&arr);
        }
        self.image = yuv2rgb(&yuv);
        self
    }

    pub fn chroma_aberrate(mut self, amount: AberrationCoords) -> Self {
        if amount.compress_x != 0 || amount.compress_y != 0 {
            let cx = amount.compress_x.clamp(0, (self.image.dim().1 / 2) as u32);
            let cy = amount.compress_y.clamp(0, (self.image.dim().0 / 2) as u32);

            plane2img(&self.get_u8().slice(s![.., .., amount.compress_c]))
                .map(|mut plane| {
                    let (w, h) = (plane.width(), plane.height());
                    let top = image::DynamicImage::ImageLuma8(plane.clone())
                        .thumbnail_exact(w - 2 * cx, h - 2 * cy);
                    image::imageops::overlay(&mut plane, &top.to_luma8(), cx.into(), cy.into());

                    plane2array(plane)
                        .map(|plane| {
                            self.image
                                .slice_mut(s![.., .., amount.compress_c])
                                .assign(&plane.mapv(f32::from));
                        })
                        .unwrap_or_default();
                })
                .unwrap_or_default();
        }

        if amount.expand_x != 0 || amount.expand_y != 0 {
            plane2img(&self.get_u8().slice(s![.., .., amount.expand_c]))
                .map(|plane| {
                    let (w, h) = (plane.width(), plane.height());
                    let plane = image::DynamicImage::ImageLuma8(plane)
                        .resize_exact(
                            w + 2 * amount.expand_x,
                            h + 2 * amount.expand_y,
                            image::imageops::Triangle,
                        )
                        .crop_imm(amount.expand_x, amount.expand_y, w, h);

                    plane2array(plane.into())
                        .map(|plane| {
                            self.image
                                .slice_mut(s![.., .., amount.expand_c])
                                .assign(&plane.mapv(f32::from));
                        })
                        .unwrap_or_default();
                })
                .unwrap_or_default();
        }
        self
    }

    pub fn compress(mut self, quality: Option<u8>) -> Self {
        self.image = ImageCompressor::from_rgb(self.get_u8())
            .compress(quality)
            .into();
        self
    }

    pub fn repeat_compression(mut self, passes: u8) -> Self {
        if passes == 0 {
            return self;
        }

        let comp = ImageCompressor::from_rgb(self.get_u8());
        self.image = ndarray::linspace::<f32>(50., 1., passes.clamp(1, 50).into())
            .map(|q| q.trunc() as u8)
            .inspect(|q| info!("Compressing with quality {q}"))
            .fold(comp, |comp, q| comp.compress(Some(q)))
            .into();

        self
    }
}

impl From<Array3<u8>> for ImageArray {
    fn from(value: Array3<u8>) -> Self {
        ImageArray {
            image: value.mapv(f32::from),
        }
    }
}

impl From<Array3<f32>> for ImageArray {
    fn from(value: Array3<f32>) -> Self {
        ImageArray { image: value }
    }
}

impl From<image::DynamicImage> for ImageArray {
    fn from(value: image::DynamicImage) -> Self {
        let image = Array3::from_shape_vec(
            (value.height() as usize, value.width() as usize, 3),
            value.into_rgb8().into_raw(),
        )
        .unwrap_or_default()
        .mapv(f32::from);
        ImageArray { image }
    }
}

impl From<ImageArray> for Array3<u8> {
    fn from(value: ImageArray) -> Self {
        value.get_u8()
    }
}

impl From<ImageArray> for Array3<f32> {
    fn from(value: ImageArray) -> Self {
        value.get_f32()
    }
}

impl From<ImageArray> for image::RgbImage {
    fn from(value: ImageArray) -> Self {
        let array = value.get_u8();
        array.as_standard_layout().as_slice().map_or(
            image::RgbImage::new(array.dim().1 as u32, array.dim().0 as u32),
            |slice| {
                image::RgbImage::from_raw(
                    array.dim().1 as u32,
                    array.dim().0 as u32,
                    slice.to_vec(),
                )
                .unwrap_or(image::RgbImage::new(
                    array.dim().1 as u32,
                    array.dim().0 as u32,
                ))
            },
        )
    }
}

fn simple_subsample(mat: &ArrayView2<f32>, factor: usize) -> Array2<f32> {
    let mat = mat
        .conv(
            &array![[0.25, 0.25], [0.25, 0.25]],
            // Same anchor = bottom left, not top right
            ConvMode::Explicit {
                padding: [[0, 1], [0, 1]],
                strides: [1, 1],
            },
            PaddingMode::Replicate,
        )
        .unwrap_or(mat.to_owned());
    mat.slice_move(s![..;factor, ..;factor])
}

fn simple_unsubsample(img: &ArrayView2<f32>, factor: usize) -> Array2<f32> {
    Array2::<f32>::from_shape_fn(img.raw_dim() * factor, |(i, j)| {
        img.get((i / factor, j / factor)).copied().unwrap_or(0.0)
    })
}

#[derive(Clone)]
pub struct ImageCompressor {
    image: Array3<f32>,
    shape: (usize, usize),

    luma_q: Array2<u16>,
    chroma_q: Array2<u16>,

    dfft_handler: ndrustfft::DctHandler<f32>,
}

impl ImageCompressor {
    pub fn from_rgb(rgb: Array3<u8>) -> Self {
        ImageCompressor::from_yuv(rgb2yuv(&rgb))
    }

    pub fn from_yuv(yuv: Array3<f32>) -> Self {
        let shape = (yuv.dim().0, yuv.dim().1);
        ImageCompressor {
            image: yuv,
            shape,
            luma_q: Array2::<u16>::from_shape_fn((8, 8), |(i, j)| BASE_LUMA_Q[i][j]),
            chroma_q: Array2::<u16>::from_shape_fn((8, 8), |(i, j)| BASE_CHROMA_Q[i][j]),
            dfft_handler: ndrustfft::DctHandler::new(8),
        }
    }

    fn scale_quantizers(&self, factor: Option<u8>) -> (Array2<f32>, Array2<f32>) {
        let factor: u16 = factor.unwrap_or(50).clamp(0, 100).into();
        let scale = u32::from(if factor < 50 {
            5000 / factor
        } else {
            200 - 2 * factor
        });

        let scaler = |mat: &Array2<u16>| {
            ((scale * mat.mapv(u32::from) + 50) / 100)
                .mapv(|v| v as f32)
                .clamp(1., f32::MAX)
        };
        (scaler(&self.luma_q), scaler(&self.chroma_q))
    }

    fn compress_tile(
        &self,
        tile: &ArrayView2<f32>,
        out: &mut ArrayViewMut2<f32>,
        quantizer: &Array2<f32>,
    ) {
        nddct2(&(tile - 128.), out, &self.dfft_handler, 0);
        nddct2_inplace(out, &self.dfft_handler, 1);

        Zip::from(&mut *out)
            .and(quantizer.view())
            .for_each(|v, &q| *v = (*v / q).round() * q);

        nddct3_inplace(out, &self.dfft_handler, 0);
        nddct3_inplace(out, &self.dfft_handler, 1);

        out.mapv_inplace(|v| (v / 256. + 128.).clamp(0., 255.));
    }

    fn compress_y(&mut self, quantizer: &Array2<f32>) {
        let mut y2: Array2<f32> = Array2::zeros((
            self.shape.0 + (8 - self.shape.0 % 8),
            self.shape.1 + (8 - self.shape.1 % 8),
        ));
        let mut y_padded = Array2::<f32>::zeros(y2.dim());
        y_padded
            .slice_mut(s![..self.image.dim().0, ..self.image.dim().1])
            .assign(&self.image.slice(s![.., .., 0]));

        let width = y2.ncols() / 8;
        y_padded
            .windows_with_stride((8, 8), (8, 8))
            .into_iter()
            .enumerate()
            .for_each(|(idx, tile)| {
                self.compress_tile(
                    &tile,
                    &mut y2.slice_mut(s![
                        8 * (idx / width)..8 * (idx / width + 1),
                        8 * (idx % width)..8 * (idx % width + 1)
                    ]),
                    quantizer,
                );
            });
        self.image
            .slice_mut(s![.., .., 0])
            .assign(&y2.slice(s![..self.shape.0, ..self.shape.1]));
    }

    fn compress_uv(&mut self, quantizer: &Array2<f32>) {
        let mut uv2: Array3<f32> = Array3::zeros((
            self.shape.0 / 2 + (8 - (self.shape.0 / 2) % 8),
            self.shape.1 / 2 + (8 - (self.shape.1 / 2) % 8),
            2,
        ));

        let width = uv2.dim().1 / 8;
        let shape = (uv2.dim().0, uv2.dim().1);
        self.image
            .slice(s![.., .., 1..])
            .axis_iter(Axis(2))
            .map(|mat| simple_subsample(&mat, 2))
            .map(|mat| {
                Array2::<f32>::from_shape_fn(shape, |(i, j)| {
                    mat.get((i, j)).copied().unwrap_or_default()
                })
            })
            .enumerate()
            .for_each(|(channel, mat)| {
                mat.windows_with_stride((8, 8), (8, 8))
                    .into_iter()
                    .enumerate()
                    .for_each(|(idx, tile)| {
                        self.compress_tile(
                            &tile,
                            &mut uv2.slice_mut(s![
                                8 * (idx / width)..8 * (idx / width + 1),
                                8 * (idx % width)..8 * (idx % width + 1),
                                channel
                            ]),
                            quantizer,
                        );
                    });
            });
        self.image.slice_mut(s![.., .., 1]).assign(
            &simple_unsubsample(&uv2.slice(s![.., .., 0]), 2)
                .slice(s![..self.shape.0, ..self.shape.1]),
        );
        self.image.slice_mut(s![.., .., 2]).assign(
            &simple_unsubsample(&uv2.slice(s![.., .., 1]), 2)
                .slice(s![..self.shape.0, ..self.shape.1]),
        );
    }

    pub fn compress(mut self, quality: Option<u8>) -> Self {
        let (luma_q, chroma_q) = self.scale_quantizers(quality);

        self.compress_y(&luma_q);
        self.compress_uv(&chroma_q);

        self
    }
}

impl std::fmt::Debug for ImageCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageCompressor")
            .field("image", &self.image)
            .field("shape", &self.shape)
            .field("luma_q", &self.luma_q)
            .finish()
    }
}

impl PartialEq for ImageCompressor {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
    }
}

impl From<ImageCompressor> for Array3<f32> {
    fn from(value: ImageCompressor) -> Self {
        yuv2rgb(&value.image)
    }
}

impl From<ImageCompressor> for Array3<u8> {
    fn from(value: ImageCompressor) -> Self {
        yuv2rgb(&value.image).mapv(|v| v.clamp(0., 255.) as u8)
    }
}

impl From<ImageCompressor> for ImageArray {
    fn from(value: ImageCompressor) -> Self {
        let image: Array3<u8> = value.into();
        ImageArray::from(image)
    }
}

impl From<ImageArray> for ImageCompressor {
    fn from(value: ImageArray) -> Self {
        let image: Array3<u8> = value.into();
        ImageCompressor::from_rgb(image)
    }
}
