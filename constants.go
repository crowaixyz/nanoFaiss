package nanofaiss

type MetricType int

// similarity metric type
const (
	METRIC_L2 MetricType = iota
	METRIC_IP
	METRIC_COSINE
)
