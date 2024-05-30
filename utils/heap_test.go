package utils

import (
	"os"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMain(m *testing.M) {
	setup()
	os.Exit(m.Run())
}

func TestDistanceMaxHeap_Push(t *testing.T) {
	Convey("DistanceMaxHeap_Push", t, func() {
		type args struct {
			dist []float64
			idx  []int32
		}
		tests := []struct {
			name      string
			args      args
			want_dist []float64
			want_idxs []int32
		}{
			{
				name: "test case 1",
				args: args{
					dist: []float64{1, 2, 3},
					idx:  []int32{1, 2, 3},
				},
				want_dist: []float64{3, 1, 2},
				want_idxs: []int32{3, 1, 2},
			},
			{
				name: "test case 2",
				args: args{
					dist: []float64{3, 2, 1},
					idx:  []int32{1, 2, 3},
				},
				want_dist: []float64{3, 2, 1},
				want_idxs: []int32{1, 2, 3},
			},
			{
				name: "test case 3",
				args: args{
					dist: []float64{1, 2, 3, 4, 5, 6, 7},
					idx:  []int32{1, 2, 3, 4, 5, 6, 7},
				},
				want_dist: []float64{5, 4, 2, 1, 3},
				want_idxs: []int32{5, 4, 2, 1, 3},
			},
			{
				name: "test case 4",
				args: args{
					dist: []float64{7, 6, 5, 4, 3, 2, 1},
					idx:  []int32{1, 2, 3, 4, 5, 6, 7},
				},
				want_dist: []float64{5, 4, 2, 3, 1},
				want_idxs: []int32{3, 4, 6, 5, 7}, // 3, 4, 5, 5, 5 ???
			},
		}

		for _, tt := range tests {
			Convey(tt.name, func() {
				var distance_max_heap DistanceMaxHeap
				distance_max_heap.Init(5)

				for i := 0; i < len(tt.args.dist); i++ {
					distance_max_heap.Push(tt.args.dist[i], tt.args.idx[i])
				}

				So(distance_max_heap.Distance(), ShouldResemble, tt.want_dist)
				So(distance_max_heap.Idxs(), ShouldResemble, tt.want_idxs)
			})
		}
	})
}

func TestDistanceMinHeap_Push(t *testing.T) {
	Convey("DistanceMinHeap_Push", t, func() {
		type args struct {
			dist []float64
			idx  []int32
		}
		tests := []struct {
			name      string
			args      args
			want_dist []float64
			want_idxs []int32
		}{
			{
				name: "test case 1",
				args: args{
					dist: []float64{1, 2, 3},
					idx:  []int32{1, 2, 3},
				},
				want_dist: []float64{1, 2, 3},
				want_idxs: []int32{1, 2, 3},
			},
			{
				name: "test case 2",
				args: args{
					dist: []float64{3, 2, 1},
					idx:  []int32{1, 2, 3},
				},
				want_dist: []float64{1, 3, 2},
				want_idxs: []int32{3, 1, 2},
			},
			{
				name: "test case 3",
				args: args{
					dist: []float64{1, 2, 3, 4, 5, 6, 7},
					idx:  []int32{1, 2, 3, 4, 5, 6, 7},
				},
				want_dist: []float64{3, 4, 6, 5, 7},
				want_idxs: []int32{3, 4, 6, 5, 7}, // 3, 4, 5, 5, 5 ???
			},
			{
				name: "test case 4",
				args: args{
					dist: []float64{7, 6, 5, 4, 3, 2, 1},
					idx:  []int32{1, 2, 3, 4, 5, 6, 7},
				},
				want_dist: []float64{3, 4, 6, 7, 5},
				want_idxs: []int32{5, 4, 2, 1, 3},
			},
		}

		for _, tt := range tests {
			Convey(tt.name, func() {
				var distance_min_heap DistanceMinHeap
				distance_min_heap.Init(5)

				for i := 0; i < len(tt.args.dist); i++ {
					distance_min_heap.Push(tt.args.dist[i], tt.args.idx[i])
				}

				So(distance_min_heap.Distance(), ShouldResemble, tt.want_dist)
				So(distance_min_heap.Idxs(), ShouldResemble, tt.want_idxs)
			})
		}
	})
}
