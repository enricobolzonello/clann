pub mod euclideandata;
pub mod angulardata;

pub trait MetricData {
    type DataType;

    fn distance(&self, i: usize, j: usize) -> f32;
    fn all_distances(&self, j: usize, out: &mut [f32]);
    fn num_points(&self) -> usize;
    fn dimensions(&self) -> usize;
    fn get_point(&self, i: usize) -> &[Self::DataType];
    fn distance_point(&self, i: usize, point: &[Self::DataType]) -> f32; 
}

pub trait Subset {
    type Out: MetricData;
    fn subset(&self, indices: &Vec<usize>) -> Self::Out;
}

pub use self::euclideandata::EuclideanData;
pub use self::angulardata::AngularData;