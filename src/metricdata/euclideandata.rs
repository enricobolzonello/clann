use ndarray::{prelude::*, Data, OwnedRepr};

use crate::metricdata::{MetricData, Subset};

pub struct EuclideanData<S: Data<Elem = f32>> {
    data: ArrayBase<S, Ix2>,
    squared_norms: Array1<f32>,
}

impl<S: Data<Elem = f32>> EuclideanData<S> {
    pub fn new(data: ArrayBase<S, Ix2>) -> Self {
        let norms = data.rows().into_iter().map(|row| row.dot(&row)).collect();

        Self {
            data,
            squared_norms: norms,
        }
    }
}

impl<S: Data<Elem = f32>> MetricData for EuclideanData<S> {
    type DataType = S::Elem;

    fn distance(&self, i: usize, j: usize) -> f32 {
        let sq_eucl = self.squared_norms[i] + self.squared_norms[j]
            - 2.0 * self.data.row(i).dot(&self.data.row(j));
        if sq_eucl < 0.0 {
            0.0
        } else {
            sq_eucl.sqrt()
        }
    }

    fn distance_point(&self, i: usize, point: &[Self::DataType]) -> f32 {
        let row = self.data.row(i);
        let sq_eucl = self.squared_norms[i] 
            + point.iter().map(|&x| x * x).sum::<f32>() 
            - 2.0 * row.dot(&ndarray::ArrayView1::from(point));
        
        if sq_eucl < 0.0 {
            0.0
        } else {
            sq_eucl.sqrt()
        }
    }

    fn all_distances(&self, j: usize, out: &mut [f32]) {
        // OPTIMIZE: try using matrix vector product, for instance
        assert_eq!(out.len(), self.data.nrows());
        for (i, oo) in out.iter_mut().enumerate() {
            *oo = self.distance(i, j);
        }
    }

    fn num_points(&self) -> usize {
        self.data.nrows()
    }

    fn dimensions(&self) -> usize {
        self.data.ncols()
    }

    fn get_point(&self, i: usize) -> &[Self::DataType] {
        self.data.row(i).to_slice().unwrap()
    }
    
}

impl<S: Data<Elem = f32>> Subset for EuclideanData<S> {
    type Out = EuclideanData<OwnedRepr<f32>>;
    fn subset(&self, indices: &[usize]) -> Self::Out {
        EuclideanData::new(self.data.select(Axis(0), indices))
    }
}
