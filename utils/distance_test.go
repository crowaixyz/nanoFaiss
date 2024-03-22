package utils

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestL2Distance(t *testing.T) {
	Convey("L2Distance", t, func() {
		// define test cases
		type args struct {
			a mat.VecDense
			b mat.VecDense
		}
		tests := []struct {
			name string
			args args
			want float64
		}{
			{
				name: "test case 1",
				args: args{
					a: *mat.NewVecDense(3, []float64{1, 2, 3}),
					b: *mat.NewVecDense(3, []float64{1, 2, 3}),
				},
				want: 0,
			},
			{
				name: "test case 2",
				args: args{
					a: *mat.NewVecDense(3, []float64{1, 2, 3}),
					b: *mat.NewVecDense(3, []float64{4, 5, 6}),
				},
				want: 5.196152422706632,
			},
			// {
			// 	name: "test case 3",
			// 	args: args{
			// 		a: *mat.NewVecDense(3, []float64{1, 2, 3}),
			// 		b: *mat.NewVecDense(4, []float64{1, 2, 3, 4}), // should be a vec of length 3
			// 	},
			// 	want: 0,
			// },
		}

		// run tests
		for _, tt := range tests {
			Convey(tt.name, func() {
				got := L2Distance(tt.args.a, tt.args.b)
				So(got, ShouldAlmostEqual, tt.want)
			})
		}
	})
}

func TestInnerProductDistance(t *testing.T) {
	Convey("InnerProductDistance", t, func() {
		// define test cases
		type args struct {
			a mat.VecDense
			b mat.VecDense
		}
		tests := []struct {
			name string
			args args
			want float64
		}{
			{
				name: "test case 1",
				args: args{
					a: *mat.NewVecDense(3, []float64{1, 2, 3}),
					b: *mat.NewVecDense(3, []float64{1, 2, 3}),
				},
				want: 14,
			},
			{
				name: "test case 2",
				args: args{
					a: *mat.NewVecDense(3, []float64{1, 2, 3}),
					b: *mat.NewVecDense(3, []float64{4, 5, 6}),
				},
				want: 32,
			},
		}

		// run tests
		for _, tt := range tests {
			Convey(tt.name, func() {
				got := InnerProductDistance(tt.args.a, tt.args.b)
				So(got, ShouldAlmostEqual, tt.want)
			})
		}
	})
}

func TestCosineDistance(t *testing.T) {
	Convey("CosineDistance", t, func() {
		// define test cases
		type args struct {
			a mat.VecDense
			b mat.VecDense
		}
		tests := []struct {
			name string
			args args
			want float64
		}{
			{
				name: "test case 1",
				args: args{
					a: *mat.NewVecDense(3, []float64{1, 2, 3}),
					b: *mat.NewVecDense(3, []float64{1, 2, 3}),
				},
				want: 1.0,
			},
			{
				name: "test case 2",
				args: args{
					a: *mat.NewVecDense(3, []float64{1, 2, 3}),
					b: *mat.NewVecDense(3, []float64{4, 5, 6}),
				},
				want: 0.9746318461970761,
			},
		}

		// run tests
		for _, tt := range tests {
			Convey(tt.name, func() {
				got := CosineDistance(tt.args.a, tt.args.b)
				So(got, ShouldAlmostEqual, tt.want)
			})
		}
	})
}
