package nanofaiss

import (
	"sort"

	"gonum.org/v1/gonum/mat"

	"github.com/crowaixyz/nanofaiss/utils"
)

type IndexFlat struct {
	size int32
	cap  int32
	dim  int32
	data []mat.VecDense // data is a slice of vectors, each represented as a gonum VecDense
}

func (iflat *IndexFlat) Init(n int32, d int32) {
	iflat.size = 0
	iflat.cap = n
	iflat.dim = d
	iflat.data = make([]mat.VecDense, n)
}

func (iflat *IndexFlat) Search(x []float64, k int32, metric_type MetricType) ([]int32, [][]float64) {
	if len(x) != int(iflat.dim) {
		panic("IndexFlat: Search: input vector dimension is not equal to index dimension")
	}

	// L2(Euclidean) distance, more bigger, more different
	if metric_type == METRIC_L2 {
		return iflat.knn_search_l2_metric(x, k)
	} else if metric_type == METRIC_IP { // inner product, more bigger, more similar
		return iflat.knn_search_ip_metric(x, k)
	} else if metric_type == METRIC_COSINE { // cosine similarity, more bigger, more similar
		return iflat.knn_search_cosine_metric(x, k)
	} else {
		panic("IndexFlat: Search: invalid metric type")
	}
}

func (iflat *IndexFlat) Add(x []float64) {
	if len(x) != int(iflat.dim) {
		panic("IndexFlat: Add: input vector dimension is not equal to index dimension")
	}

	if iflat.size >= iflat.cap {
		panic("IndexFlat: Add: index is full")
	}
	if len(x) != int(iflat.dim) {
		panic("IndexFlat: Add: input vector dimension is not equal to index dimension")
	}

	iflat.size++
	iflat.data[iflat.size-1] = *mat.NewVecDense(int(iflat.dim), x)
}

func (iflat *IndexFlat) Remove() {
	iflat.size = 0
	iflat.data = nil
}

func (iflat *IndexFlat) knn_search_l2_metric(x []float64, k int32) ([]int32, [][]float64) {
	var distance_max_heap utils.DistanceMaxHeap
	distance_max_heap.Init(k)

	for i := int32(0); i < iflat.size; i++ {
		distance := utils.L2Distance(iflat.data[i], *mat.NewVecDense(int(iflat.dim), x))

		if distance_max_heap.Size() < k {
			distance_max_heap.Push(distance, i)
		} else {
			if distance < distance_max_heap.Peek() {
				distance_max_heap.Pop()
				distance_max_heap.Push(distance, i)
			}
		}
	}

	// sort the idxs and select vectors by idxs
	idxs := distance_max_heap.Idxs()
	sort.Slice(idxs, func(i, j int) bool {
		return idxs[i] < idxs[j]
	})

	vecs := make([][]float64, len(idxs))
	for i := range idxs {
		vecs[i] = iflat.data[idxs[i]].RawVector().Data
	}

	return idxs, vecs
}

func (iflat *IndexFlat) knn_search_ip_metric(x []float64, k int32) ([]int32, [][]float64) {
	var distance_min_heap utils.DistanceMinHeap
	distance_min_heap.Init(k)

	for i := int32(0); i < iflat.size; i++ {
		distance := utils.InnerProductDistance(iflat.data[i], *mat.NewVecDense(int(iflat.dim), x))

		if distance_min_heap.Size() < k {
			distance_min_heap.Push(distance, i)
		} else {
			if distance > distance_min_heap.Peek() {
				distance_min_heap.Pop()
				distance_min_heap.Push(distance, i)
			}
		}
	}

	// sort the idxs and select vectors by idxs
	idxs := distance_min_heap.Idxs()
	sort.Slice(idxs, func(i, j int) bool {
		return idxs[i] < idxs[j]
	})

	vecs := make([][]float64, len(idxs))
	for i := range idxs {
		vecs[i] = iflat.data[idxs[i]].RawVector().Data
	}

	return idxs, vecs
}

func (iflat *IndexFlat) knn_search_cosine_metric(x []float64, k int32) ([]int32, [][]float64) {
	var distance_min_heap utils.DistanceMinHeap
	distance_min_heap.Init(k)

	for i := int32(0); i < iflat.size; i++ {
		distance := utils.CosineDistance(iflat.data[i], *mat.NewVecDense(int(iflat.dim), x))

		if distance_min_heap.Size() < k {
			distance_min_heap.Push(distance, i)
		} else {
			if distance > distance_min_heap.Peek() {
				distance_min_heap.Pop()
				distance_min_heap.Push(distance, i)
			}
		}
	}

	// sort the idxs and select vectors by idxs
	idxs := distance_min_heap.Idxs()
	sort.Slice(idxs, func(i, j int) bool {
		return idxs[i] < idxs[j]
	})

	vecs := make([][]float64, len(idxs))
	for i := range idxs {
		vecs[i] = iflat.data[idxs[i]].RawVector().Data
	}

	return idxs, vecs
}
