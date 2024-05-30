package kmeans

import (
	"math"
	"math/rand"
	"time"

	"github.com/crowaixyz/nanofaiss/utils"
	"gonum.org/v1/gonum/mat"
)

type KMeans struct {
	nlist           int32
	max_iterations  int32
	delta_threshold float64
}

func NewWithOptions(nlist int32, max_interations int32, delta_threshold float64) KMeans {
	return KMeans{
		nlist:           nlist,
		max_iterations:  max_interations,
		delta_threshold: delta_threshold,
	}
}

func (km *KMeans) Train(vecs []mat.VecDense, dim int32) []Cluster {
	vec_size := int32(len(vecs))
	init_cluster_size := vec_size / km.nlist / 2  // set the default init size of cluster

	// step 1. Initialize the centroids
	// FIXME: randomly select the centroids temporarily, better way to init centroids ?
	rand_centroid_idxs := generate_random_numbers(vec_size, km.nlist)
	clusters := make([]Cluster, km.nlist)
	for i := int32(0); i < km.nlist; i++ {
		clusters[i].size = 0
		clusters[i].vec_idxs = make(map[int32]bool, init_cluster_size)
		clusters[i].center = vecs[rand_centroid_idxs[i]]
	}

	// step 2. Iterate until convergence: reach interation limit or adjust rate lower than threshold
	// FIXME: bad performance
	for i := int32(0); i < km.max_iterations; i++ {
		vec_cluster_map := make(map[int32]int32, vec_size)
		vec_adjust_num := 0
		// step 2.1. Assign each vector to the nearest cluster
		for j := int32(0); j < vec_size; j++ {
			// calculate distance between vector and cluster centroid, only support L2 distance
			min_dist := math.MaxFloat64
			min_dist_cluster_idx := int32(-1)
			if _, ok := vec_cluster_map[j]; ok {
				min_dist_cluster_idx = vec_cluster_map[j]
			}
			for k := int32(0); k < km.nlist; k++ {
				v := vecs[j]
				c := clusters[k].center
				dist := utils.L2Distance(v, c)
				if dist < min_dist {
					min_dist = dist
					min_dist_cluster_idx = k
				}
			}

			// check if should change the cluster index of vector
			c_idx := vec_cluster_map[j]
			if min_dist_cluster_idx != c_idx {
				// delete vec index from original cluster
				delete(clusters[c_idx].vec_idxs, j)
				clusters[c_idx].size -= 1

				// put vec index to new cluster
				clusters[min_dist_cluster_idx].vec_idxs[j] = true
				clusters[min_dist_cluster_idx].size += 1

				// change cluster index of vector
				vec_cluster_map[j] = min_dist_cluster_idx

				// increase vector adjust number
				vec_adjust_num += 1
			}

			// break if vec adjust cluster index rate less than threshold
			if (float64(vec_adjust_num) / float64(vec_size)) <= km.delta_threshold {
				break
			}
		}

		// step 2.2 set the centeroid of cluster to the mean value of all vectors in the cluster
		for l := int32(0); l < km.nlist; l++ {
			// FIXME: if cluster is empty, move the first vector from every other cluster
			if clusters[l].size <= 0 {
				for m := int32(0); m < km.nlist; m++ {
					if m != l && clusters[m].size > 1 {
						// move the first vector of cluster m to cluster l
						for v_idx := range clusters[m].vec_idxs {
							// delete vector from cluster m
							delete(clusters[m].vec_idxs, v_idx)
							clusters[m].size -= 1

							// add vector to new cluster l
							clusters[l].vec_idxs[v_idx]= true
							clusters[l].size += 1

							break
						}
					}
				}
			}
		}
		for k := int32(0); k < km.nlist; k++ {
			clusters[k].center = vec_mean(clusters[k].vec_idxs, vecs, dim)
		}
	}

	// step 3. Return the clusters
	return clusters
}

func generate_random_numbers(upper int32, n int32) []int32 {
	rand_nums := make(map[int32]int, n)

	for i := int32(0); i < n; i++ {
		tmp := rand.New(rand.NewSource(time.Now().UnixNano())).Int31n(upper)
		if _, ok := rand_nums[tmp]; !ok {
			rand_nums[tmp] = 1  // 1 - no useful meaning
		} else {
			for {
				tmp = rand.New(rand.NewSource(time.Now().UnixNano())).Int31n(upper)
				if _, ok := rand_nums[tmp]; ok {
					continue
				} else {
					rand_nums[tmp] = 1
					break
				}
			}
		}
	}

	num_list := []int32{}
	for k := range rand_nums {
		num_list = append(num_list, int32(k))
	}
	
	return num_list
}


// func generate_random_centroid(dim int32) mat.VecDense {
// 	vec := mat.NewVecDense(int(dim), nil)
// 	for i := int32(0); i < dim; i++ {
// 		vec.SetVec(int(i), rand.New(rand.NewSource(time.Now().UnixNano())).Float64())
// 	}
// 	return *vec
// }


func vec_mean(vec_idxs map[int32]bool, vecs []mat.VecDense, dim int32) mat.VecDense {
	mean := mat.NewVecDense(int(dim), nil)
	for idx := range vec_idxs {
		v := vecs[idx]
		mean.AddVec(mean, &v)
	}

	mean.ScaleVec(1/float64(len(vec_idxs)), mean)

	return *mean
}