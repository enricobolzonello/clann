use ndarray::{prelude::*, Data, OwnedRepr};

use crate::metricdata::{MetricData, Subset};

pub struct AngularData<S: Data<Elem=f32>> {
    data: ArrayBase<S, Ix2>,
    norms: Array1<f32>,
}

impl<S: Data<Elem = f32>> AngularData<S> {
    pub fn new(data: ArrayBase<S, Ix2>) -> Self {
        let norms = data.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();

        Self {
            data,
            norms,
        }
    }
}

impl<S: Data<Elem = f32>> MetricData for AngularData<S> {
    type DataType = S::Elem;

    fn distance(&self, i: usize, j: usize) -> f32 {
        1.0 - ( self.data.row(i).dot(&self.data.row(j)) / (self.norms[i] * self.norms[j]) )
    }

    fn all_distances(&self, j: usize, out: &mut [f32]){
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

impl<S: Data<Elem = f32>> Subset for AngularData<S> {
    type Out = AngularData<OwnedRepr<f32>>;
    fn subset<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self::Out {
        let indices: Vec<usize> = indices.into_iter().collect();
        AngularData::new(self.data.select(Axis(0), &indices))
    }
}