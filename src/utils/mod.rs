use hdf5::File;
use log::debug;
use ndarray::{Array, Ix2};

pub mod metrics;

pub fn load_hdf5_dataset(filepath: &str) -> Result<(Array<f32, Ix2>, Array<f32, Ix2>), String> {
    let file = File::open(filepath).map_err(|e| format!("Error opening file '{}': {}", filepath, e))?;

    let dataset = file.dataset("train").map_err(|e| format!("Error opening dataset 'train': {}", e))?;
    let queries = file.dataset("test").map_err(|e| format!("Error opening dataset 'test': {}", e))?;

    // Read the dataset into an ndarray
    let dataset_array = dataset
        .read::<f32, Ix2>()
        .map_err(|e| format!("Error reading dataset as f32 array: {}", e))?;
    let dataset_queries = queries
        .read::<f32, Ix2>()
        .map_err(|e| format!("Error reading dataset as f32 array: {}", e))?;

    debug!("Loaded dataset with shape: {:?}", dataset_array.dim());

    Ok((dataset_array, dataset_queries))
}