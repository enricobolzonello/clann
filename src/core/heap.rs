use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Element {
    pub(crate) distance: OrderedFloat<f32>,
    pub(crate) point_index: usize,
}

pub(crate) struct TopKClosestHeap {
    heap: BinaryHeap<Element>, 
    length: usize,
}

impl TopKClosestHeap {
    pub(crate) fn new(top_n: usize) -> Self {
        TopKClosestHeap {
            heap: BinaryHeap::with_capacity(top_n),
            length: top_n,
        }
    }

    pub(crate) fn add(&mut self, element: Element) -> bool {
        if self.heap.len() < self.length {
            self.heap.push(element);
        } else if let Some(max) = self.heap.peek() {
            if element.distance < max.distance {
                // Remove the largest element if the new element is smaller
                self.heap.pop();
                self.heap.push(element);
            } else {
                return false;
            }
        }
        true
    }

    pub(crate) fn get_top(&self) -> Option<(usize, f32)> {
        self.heap.peek().map(|e| (e.point_index, e.distance.0))
    }

    pub(crate) fn to_list(&self) -> Vec<(f32, usize)> {
        let mut elements: Vec<_> = self.heap.iter()
            .map(|e| (e.distance.into_inner(), e.point_index))
            .collect();
        elements.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        elements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_elements_less_than_capacity() {
        let mut heap = TopKClosestHeap::new(3);

        assert!(heap.add(Element {
            distance: OrderedFloat(2.5),
            point_index: 0,
        }));
        assert!(heap.add(Element {
            distance: OrderedFloat(1.5),
            point_index: 1,
        }));

        let elements = heap.to_list();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements, vec![(1.5, 1), (2.5, 0)]);
    }

    #[test]
    fn test_add_elements_exceeding_capacity() {
        let mut heap = TopKClosestHeap::new(3);

        heap.add(Element {
            distance: OrderedFloat(3.0),
            point_index: 0,
        });
        heap.add(Element {
            distance: OrderedFloat(2.0),
            point_index: 1,
        });
        heap.add(Element {
            distance: OrderedFloat(1.0),
            point_index: 2,
        });

        // Adding an element with a smaller distance should replace the largest
        heap.add(Element {
            distance: OrderedFloat(0.5),
            point_index: 3,
        });

        let elements = heap.to_list();
        assert_eq!(elements.len(), 3);
        println!("{:?}", elements);
        assert!(elements.contains(&(1.0, 2)));
        assert!(elements.contains(&(2.0, 1)));
        assert!(elements.contains(&(0.5, 3)));
    }

    #[test]
    fn test_no_replace_if_distance_is_larger() {
        let mut heap = TopKClosestHeap::new(3);

        heap.add(Element {
            distance: OrderedFloat(3.0),
            point_index: 0,
        });
        heap.add(Element {
            distance: OrderedFloat(2.0),
            point_index: 1,
        });
        heap.add(Element {
            distance: OrderedFloat(1.0),
            point_index: 2,
        });

        // Adding an element with a larger distance does not replace the largest
        assert!(!heap.add(Element {
            distance: OrderedFloat(4.0),
            point_index: 3,
        }));

        let elements = heap.to_list();
        assert_eq!(elements.len(), 3);
        assert!(!elements.contains(&(4.0, 3)));
    }

    #[test]
    fn test_get_top_element() {
        let mut heap = TopKClosestHeap::new(2);

        heap.add(Element {
            distance: OrderedFloat(2.0),
            point_index: 1,
        });
        heap.add(Element {
            distance: OrderedFloat(1.0),
            point_index: 2,
        });

        assert_eq!(heap.get_top(), Some((1,2.0)));

        heap.add(Element {
            distance: OrderedFloat(0.5),
            point_index: 3,
        });

        assert_eq!(heap.get_top(), Some((2,1.0)));
    }

    #[test]
    fn test_empty_heap() {
        let heap = TopKClosestHeap::new(3);
        assert_eq!(heap.to_list().len(), 0);
        assert_eq!(heap.get_top(), None);
    }
}
