use clann::{build, init, metricdata::AngularData, search};
use hdf5::File;
use ndarray::{Array, Ix2};
use tracing::debug;

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


fn main() {
    let (data_raw, queries) = load_hdf5_dataset("/home/bolzo/puffinn-tests/datasets/glove-25-angular.hdf5").unwrap();
    let data = AngularData::new(data_raw);
    let mut index = init(data).unwrap();

    let _= build(&mut index).map_err(|e| eprintln!("Error: {}", e));

    let _ = search(&index, queries.row(0).as_slice().unwrap(), 3, 0.9);

    println!("done!")
}
