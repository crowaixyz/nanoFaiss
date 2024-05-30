package kmeans

import "gonum.org/v1/gonum/mat"

type Cluster struct {
	size     int32
	center   mat.VecDense
	vec_idxs map[int32]bool
}

func (c *Cluster) Size() int32 {
	return c.size
}

func (c *Cluster) Center() *mat.VecDense {
	return &c.center
}

func (c *Cluster) VecIdxs() map[int32]bool {
	return c.vec_idxs
}
