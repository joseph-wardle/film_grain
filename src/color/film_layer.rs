use crate::color::spectral_response::SpectralResponse;
use crate::rendering::FilmGrainOptions;

pub struct FilmConfiguration {
    pub layers: Vec<FilmLayer>,
}

pub struct FilmLayer {
    /// A descriptive name for this layer (e.g. "Cyan", "Magenta", or a custom label).
    pub name: String,
    /// The spectral response function for this layer.
    pub spectral_response: Box<dyn SpectralResponse + Send + Sync>,
    /// Optional grain options specific to this layer.
    pub grain_options: Option<FilmGrainOptions>,
}