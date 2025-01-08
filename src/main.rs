use clann::{init, metricdata::AngularData};
use hdf5::File;
use ndarray::{Array, Ix2};
use tracing::debug;

pub fn load_hdf5_dataset(filepath: &str) -> Result<Array<f32, Ix2>, String> {
    // Open the file safely
    let file = File::open(filepath).map_err(|e| format!("Error opening file '{}': {}", filepath, e))?;

    // Open the dataset named "test"
    let dataset = file.dataset("test").map_err(|e| format!("Error opening dataset 'test': {}", e))?;

    // Read the dataset into an ndarray
    let array = dataset
        .read::<f32, Ix2>()
        .map_err(|e| format!("Error reading dataset as f32 array: {}", e))?;

    debug!("Loaded dataset with shape: {:?}", array.dim());

    Ok(array)
}


fn main() {
    let data_raw = load_hdf5_dataset("/home/bolzo/puffinn-tests/datasets/glove-25-angular.hdf5").unwrap();
    let data = AngularData::new(data_raw);
    let _index = init(data).unwrap();

    println!("done!")
}
