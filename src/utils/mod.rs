use hdf5::File;
use log::debug;
use ndarray::{Array, Ix1, Ix2};

pub mod metrics;

pub fn load_hdf5_dataset(filepath: &str) -> Result<(Array<f32, Ix2>, Array<f32, Ix2>, Array<f32, Ix2>), String> {
    let file = File::open(filepath).map_err(|e| format!("Error opening file '{}': {}", filepath, e))?;

    let dataset = file.dataset("train").map_err(|e| format!("Error opening dataset 'train': {}", e))?;
    let queries = file.dataset("test").map_err(|e| format!("Error opening dataset 'test': {}", e))?;
    let distances = file.dataset("distances").map_err(|e| format!("Error opening dataset 'distances': {}", e))?;

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

    Ok((dataset_array, dataset_queries, ground_truth_distances))
}

fn threshold(distances: &Array<f32, Ix1>, count: usize, epsilon: f32) -> f32 {
    // Assuming distances need to be sorted first since we're finding the k-th smallest
    let mut sorted_distances: Vec<f32> = distances.to_vec();
    sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_distances[count - 1] + epsilon
}

pub fn get_recall_values(
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
        (recalls
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f32>()
            / recalls.len() as f32)
            .sqrt()
            / count as f32
    };

    (mean_recall, std_recall, recalls)
}
