package nanofaiss

import (
	"gonum.org/v1/gonum/mat"

	"github.com/crowaixyz/nanofaiss/pkg/kmeans"
)

type IndexIVFFlat struct {
	size int32
	cap  int32
	dim  int32
	vecs []mat.VecDense

	nlist    int32
	clusters []kmeans.Cluster
}

func (ivf *IndexIVFFlat) Train(index_flat *IndexFlat, nlist int32, max_iterations int32, delta_threshold float64) {
	ivf.size = index_flat.size
	ivf.cap = index_flat.cap
	ivf.dim = index_flat.dim
	ivf.vecs = index_flat.vecs

	ivf.nlist = nlist

	km := kmeans.NewWithOptions(nlist, max_iterations, delta_threshold)
	ivf.clusters = km.Train(ivf.vecs, ivf.dim)
}

func (ivf *IndexIVFFlat) Search(x []float64, k int32, nprobe int32) ([]int32, [][]float64) {
	if nprobe >= ivf.nlist {
		nprobe = ivf.nlist // nprobe should not be greater than nlist
	}

	// step 1. get top nprobe cluster based on distance with cluster center
	var center_index IndexFlat
	center_index.Init(ivf.nlist, ivf.dim)

	for _, c := range ivf.clusters {
		center_index.Add(c.Center().RawVector().Data)
	}

	cluster_idxs, _ := center_index.Search(x, nprobe, METRIC_L2)  // only support L2 distance

	// step 2. search top k vectors from selected clusters in step 1
	var candidate_index IndexFlat
	candidate_size := int32(0)
	for _, idx := range cluster_idxs {
		cluster := ivf.clusters[idx]
		candidate_size += cluster.Size()
	}

	candidate_index.Init(candidate_size, ivf.dim)

	for _, ci := range cluster_idxs {
		cluster := ivf.clusters[ci]
		vec_idxs := cluster.VecIdxs()

		for vi := range vec_idxs {
			candidate_index.Add(ivf.vecs[vi].RawVector().Data)
		}
	}

	return candidate_index.Search(x, k, METRIC_L2)
}
