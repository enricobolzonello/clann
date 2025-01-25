use clann::{metricdata::{AngularData, MetricData, Subset}, puffinn_binds::PuffinnIndex, utils::load_hdf5_dataset};
use criterion::criterion_main;
use rand::{seq::SliceRandom, thread_rng};

// Binary search to find the lower limit for the memory
fn find_memory_limit(
    dataset_name: &str,
    initial_min: usize,
    initial_max: usize,
    kb_per_point: usize,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let dataset_path = format!("./datasets/{}.hdf5", dataset_name);
    let (data_raw, _, _) = load_hdf5_dataset(&dataset_path)?;
    let data = AngularData::new(data_raw);

    let mut min_limit = initial_min;
    let mut max_limit = initial_max;
    let mut best_limit = initial_max;
    let mut optimal_points = 0;

    while min_limit <= max_limit {
        let memory_limit = (min_limit + max_limit) / 2;
        let random_indexes = memory_limit / (kb_per_point * 1024);

        let mut rng = thread_rng();
        let data_indices: Vec<usize> = (0..data.num_points())
            .collect::<Vec<usize>>()
            .choose_multiple(&mut rng, random_indexes)
            .cloned()
            .collect();
        
        let sub_data = data.subset(data_indices);

        // Attempt to create the index
        match PuffinnIndex::new(&sub_data, memory_limit) {
            Ok(_) => {
                max_limit = memory_limit - 1;
                best_limit = best_limit.min(memory_limit);
                optimal_points = random_indexes;
            }
            Err(_) => {
                min_limit = memory_limit + 1;
            }
        }
    }

    Ok((best_limit, optimal_points))
}

fn run_memory_tests() {
    let dataset_name = "glove-25-angular";
    let kb_per_point_values = [1, 2, 4, 8, 10, 12, 15];
    let initial_min = 100_000;
    let initial_max = 400_000;

    println!("{0: <13} | {1: <15} | {2: <10}", "KB per Point", "Memory Limit", "N Points");
    println!("{:-<13}-|-{:-<15}-|-{:-<10}", "", "", "");

    for &kb_per_point in &kb_per_point_values {
        match find_memory_limit(dataset_name, initial_min, initial_max, kb_per_point) {
            Ok((best_limit, optimal_points)) => {
                println!(
                    "{0: <13} | {1: <15} | {2: <10}", 
                    kb_per_point, 
                    best_limit, 
                    optimal_points
                );
            }
            Err(e) => eprintln!(
                "Error finding memory limit for {} kb per point: {}",
                kb_per_point, e
            ),
        }
    }
}

criterion_main!(run_memory_tests);
