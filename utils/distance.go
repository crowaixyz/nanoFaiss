package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func L2Distance(a, b mat.VecDense) float64 {
	diff := mat.NewVecDense(a.Len(), nil)
	diff.SubVec(&a, &b)
	dot := mat.Dot(diff, diff)
	distance := math.Sqrt(dot)

	return distance
}

func InnerProductDistance(a, b mat.VecDense) float64 {
	return mat.Dot(&a, &b)
}

func CosineDistance(a, b mat.VecDense) float64 {
	dot := mat.Dot(&a, &b)
	norm_a := mat.Norm(&a, 2)
	norm_b := mat.Norm(&b, 2)
	return dot / (norm_a * norm_b)
}
