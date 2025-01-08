use ndarray::prelude::*;

use crate::metricdata::MetricData;

fn argmax(v: &[f32]) -> usize {
    let mut i = 0;
    let mut m = v[i];
    for idx in 1..v.len() {
        if v[idx] > m {
            i = idx;
            m = v[idx];
        }
    }
    i
}

/// Returns a tuple of two elements: the centers, the assignment, and the radius.
/// The centers array is a vector of indices into the input data.
/// The assignment is a vector of indices into the centers array,
/// with the same length as there are input rows.
pub fn greedy_minimum_maximum<D: MetricData>(
    data: &D,
    k: usize,
) -> (Array1<usize>, Array1<usize>, Array1<f32>) {
    let n = data.num_points();
    if n <= k {
        // Each point is its own center
        let centers = Array1::<usize>::from_iter(0..n);
        let assignment = Array1::<usize>::from_iter(0..n);
        return (centers, assignment, Array1::<f32>::zeros(n));
    }

    let first_center = 0usize;
    let mut centers: Array1<usize> = Array1::zeros(k);
    centers[0] = first_center;
    let mut distances = vec![f32::INFINITY; n];
    let mut new_distances = vec![f32::INFINITY; n];
    let mut assignment = Array1::<usize>::zeros(n);

    data.all_distances(first_center, &mut distances);

    for idx in 1..k {
        // FIXME: in a multithreaded context this call deadlocks
        // crate::check_signals();
        let farthest = argmax(&distances);
        centers[idx] = farthest;
        data.all_distances(farthest, &mut new_distances);
        for i in 0..n {
            if new_distances[i] < distances[i] {
                assignment[i] = idx;
                distances[i] = new_distances[i];
            }
        }
    }

    let mut radii: Array1<f32> = Array1::zeros(k);

    for i in 0..n {
        radii[assignment[i]] = radii[assignment[i]].max(distances[i]);
    }

    (centers, assignment, radii)
}