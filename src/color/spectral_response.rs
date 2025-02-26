pub trait SpectralResponse {
    /// Returns the spectral response of the sensor at the given wavelength.
    fn response(&self, wavelength: f32) -> f32;
}