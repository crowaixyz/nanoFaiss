package nanofaiss

type Index interface {
	Init(n int32, d int32)
	Search(x []float64, k int32, metric_type MetricType) ([]int32, [][]float64)
	Add(x []float64)
	BatchAdd(x [][]float64)
	Remove()
}
