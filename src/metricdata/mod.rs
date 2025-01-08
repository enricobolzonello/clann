pub mod euclideandata;
pub mod angulardata;

pub trait MetricData {
    type DataType;

    fn distance(&self, i: usize, j: usize) -> f32;
    fn all_distances(&self, j: usize, out: &mut [f32]);
    fn num_points(&self) -> usize;
    fn dimensions(&self) -> usize;
    fn get_point(&self, i: usize) -> &[Self::DataType];
}

pub trait Subset {
    type Out: MetricData;
    fn subset<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self::Out;
}

use ndarray::ArrayView1;

pub use self::euclideandata::EuclideanData;
pub use self::angulardata::AngularData;