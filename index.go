package nanofaiss

type Index interface {
	Init(n int32, d int32)
	Search(x []float32, k int32, metric_type MetricType) ([]int32, [][]float32)
	Add(x []float32)
	Remove()
}
