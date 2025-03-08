use std::cmp::Ordering;
use std::fs;

use hdf5::File;
use log::debug;
use ndarray::{Array, Ix1, Ix2};
use ndarray::{Array2, Axis};

pub(crate) mod metrics;

use rand::thread_rng;
use rand::Rng;

use crate::metricdata::{MetricData, Subset};
use crate::puffinn_binds::IndexableSimilarity;

pub(crate) use metrics::RunMetrics;

pub struct Hdf5Dataset {
    pub dataset_array: Array<f32, Ix2>,
    pub dataset_queries: Array<f32, Ix2>,
    pub ground_truth_distances: Array<f32, Ix2>,
}

pub fn load_hdf5_dataset(filepath: &str) -> Result<Hdf5Dataset, String> {
    let file =
        File::open(filepath).map_err(|e| format!("Error opening file '{}': {}", filepath, e))?;

    let dataset = file
        .dataset("train")
        .map_err(|e| format!("Error opening dataset 'train': {}", e))?;
    let queries = file
        .dataset("test")
        .map_err(|e| format!("Error opening dataset 'test': {}", e))?;
    let distances = file
        .dataset("distances")
        .map_err(|e| format!("Error opening dataset 'distances': {}", e))?;

    // Read the dataset into an ndarray
    let dataset_array = dataset
        .read::<f32, Ix2>()
        .map_err(|e| format!("Error reading dataset as f32 array: {}", e))?;
    let dataset_queries = queries
        .read::<f32, Ix2>()
        .map_err(|e| format!("Error reading dataset as f32 array: {}", e))?;
    let ground_truth_distances = distances
        .read::<f32, Ix2>()
        .map_err(|e| format!("Error reading dataset as f32 array: {}", e))?;

    debug!("Loaded dataset with shape: {:?}", dataset_array.dim());

    Ok(Hdf5Dataset {
        dataset_array,
        dataset_queries,
        ground_truth_distances,
    })
}

fn threshold(distances: &Array<f32, Ix1>, count: usize, epsilon: f32) -> f32 {
    // Assuming distances need to be sorted first since we're finding the k-th smallest
    let mut sorted_distances: Vec<f32> = distances.to_vec();
    sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_distances[count - 1] + epsilon
}

pub(crate) fn get_recall_values(
    dataset_distances: &Array<f32, Ix2>,
    run_distances: &[Vec<f32>],
    count: usize,
) -> (f32, f32, Vec<f32>) {
    let mut recalls = Vec::with_capacity(run_distances.len());

    for i in 0..run_distances.len() {
        // Get threshold from dataset (ground truth) distances
        let t = threshold(&dataset_distances.row(i).to_owned(), count, 1e-3);

        // Count matches in our search results
        let mut actual = 0;
        for &d in run_distances[i].iter().take(count) {
            if d <= t {
                actual += 1;
            }
        }
        recalls.push(actual as f32);
    }

    let mean_recall = recalls.iter().sum::<f32>() / (recalls.len() as f32 * count as f32);
    let std_recall = {
        let mean = recalls.iter().sum::<f32>() / recalls.len() as f32;
        (recalls.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / recalls.len() as f32).sqrt()
            / count as f32
    };

    (mean_recall, std_recall, recalls)
}

pub(crate) fn db_exists(db_file_path: &str) -> bool {
    fs::metadata(db_file_path).is_ok()
}

pub fn generate_random_unit_vectors(n: usize, dimensions: usize) -> Array2<f32> {
    let mut rng = thread_rng();
    let mut data = Array2::<f32>::zeros((n, dimensions));

    for mut row in data.axis_iter_mut(Axis(0)) {
        let vec: Vec<f32> = (0..dimensions).map(|_| rng.gen::<f32>()).collect();
        let norm: f32 = vec.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        row.assign(&ndarray::arr1(
            &vec.iter().map(|x| x / norm).collect::<Vec<f32>>(),
        ));
    }

    data
}

pub fn brute_force_search<T>(metric_data: &T, query: &[T::DataType], k: usize) -> Vec<u32>
where
    T: MetricData + IndexableSimilarity<T> + Subset,
    <T as Subset>::Out: IndexableSimilarity<<T as Subset>::Out>,
{
    let mut distances: Vec<(u32, f32)> = (0..metric_data.num_points() as u32)
        .map(|i| {
            let dist = metric_data.distance_point(i as usize, query);
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    distances.into_iter().take(k).map(|(idx, _)| idx).collect()
}
