// Copyright (c) 2024 The nanoFaiss Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package utils provides utility functions for the nanoFaiss project.

package utils

// Heap is an interface for min heap and max heap
// use min or max heap to maintain the k-nearest neighbors.
type Heap interface {
	Init(k int32)
	Push(dist float64, idx int32)
	Pop() float64
	Peek() float64
	Distance() []float64
	Idxs() []int32
	Size() int32
	Capacity() int32
	Empty() bool
	HeapifyDown(i int32)
	HeapifyUp(i int32)
}

// DistanceMinHeap maintains the k-nearest neighbors' distance and their corresponding idxs in embedding list
// suitable for cosine similarity metric
type DistanceMinHeap struct {
	distance []float64
	idxs     []int32
	size     int32
	capacity int32
}

func (minh *DistanceMinHeap) Init(k int32) {
	minh.size = 0
	minh.capacity = k
	minh.distance = make([]float64, k)
	minh.idxs = make([]int32, k)
}

func (minh *DistanceMinHeap) Push(dist float64, idx int32) {
	if minh.size < minh.capacity {
		minh.size++

		minh.distance[minh.size-1] = dist
		minh.idxs[minh.size-1] = idx

		minh.HeapifyUp(minh.size - 1)
	} else {
		if dist > minh.Peek() {
			// pop the top element
			minh.Pop()

			// push the new element
			minh.size++
			minh.distance[minh.size-1] = dist
			minh.idxs[minh.size-1] = idx
			minh.HeapifyUp(minh.size - 1)
		}
	}
}

func (minh *DistanceMinHeap) Pop() float64 {
	top_val := minh.distance[0]

	minh.distance[0] = minh.distance[minh.size-1]
	minh.idxs[0] = minh.idxs[minh.size-1]
	minh.size--
	minh.HeapifyDown(0)

	return top_val
}

func (minh DistanceMinHeap) Peek() float64 {
	return minh.distance[0]
}

func (minh DistanceMinHeap) Distance() []float64 {
	return minh.distance[:minh.size]
}

func (minh DistanceMinHeap) Idxs() []int32 {
	return minh.idxs[:minh.size]
}

func (minh DistanceMinHeap) Size() int32 {
	return minh.size
}

func (minh DistanceMinHeap) Capacity() int32 {
	return minh.capacity
}

func (minh DistanceMinHeap) Empty() bool {
	return minh.size == 0
}

func (minh *DistanceMinHeap) HeapifyDown(i int32) {
	left := left_child(i)
	right := right_child(i)
	smallest := i

	if left < minh.size && minh.distance[left] < minh.distance[smallest] {
		smallest = left
	}

	if right < minh.size && minh.distance[right] < minh.distance[smallest] {
		smallest = right
	}

	if smallest != i {
		minh.distance[i], minh.distance[smallest] = minh.distance[smallest], minh.distance[i]
		minh.idxs[i], minh.idxs[smallest] = minh.idxs[smallest], minh.idxs[i]

		minh.HeapifyDown(smallest)
	}
}

func (minh *DistanceMinHeap) HeapifyUp(i int32) {
	for i > 0 && minh.distance[i] < minh.distance[parent(i)] {
		minh.distance[i], minh.distance[parent(i)] = minh.distance[parent(i)], minh.distance[i]
		minh.idxs[i], minh.idxs[parent(i)] = minh.idxs[parent(i)], minh.idxs[i]

		i = parent(i)
	}
}

// DistanceMaxHeap maintain the k-nearest neighbors' distance, suitable for L2 and IP similarity metric
type DistanceMaxHeap struct {
	distance []float64
	idxs     []int32
	size     int32
	capacity int32
}

func (maxh *DistanceMaxHeap) Init(k int32) {
	maxh.size = 0
	maxh.capacity = k
	maxh.distance = make([]float64, k)
	maxh.idxs = make([]int32, k)
}

func (maxh *DistanceMaxHeap) Push(dist float64, idx int32) {
	if maxh.size < maxh.capacity {
		maxh.size++
		maxh.distance[maxh.size-1] = dist
		maxh.idxs[maxh.size-1] = idx

		maxh.HeapifyUp(maxh.size - 1)
	} else {
		if dist < maxh.Peek() {
			// pop the top element
			maxh.Pop()

			// push the new element
			maxh.size++
			maxh.distance[maxh.size-1] = dist
			maxh.idxs[maxh.size-1] = idx
			maxh.HeapifyUp(maxh.size - 1)
		}
	}
}

func (maxh *DistanceMaxHeap) Pop() float64 {
	top_val := maxh.distance[0]

	maxh.distance[0] = maxh.distance[maxh.size-1]
	maxh.idxs[0] = maxh.idxs[maxh.size-1]
	maxh.size--
	maxh.HeapifyDown(0)

	return top_val
}

func (maxh DistanceMaxHeap) Peek() float64 {
	return maxh.distance[0]
}

func (maxh DistanceMaxHeap) Distance() []float64 {
	return maxh.distance[:maxh.size]
}

func (maxh DistanceMaxHeap) Idxs() []int32 {
	return maxh.idxs[:maxh.size]
}

func (maxh DistanceMaxHeap) Size() int32 {
	return maxh.size
}

func (maxh DistanceMaxHeap) Capacity() int32 {
	return maxh.capacity
}

func (maxh DistanceMaxHeap) Empty() bool {
	return maxh.size == 0
}

func (maxh *DistanceMaxHeap) HeapifyDown(i int32) {
	left := left_child(i)
	right := right_child(i)
	largest := i

	if left < maxh.size && maxh.distance[left] > maxh.distance[largest] {
		largest = left
	}

	if right < maxh.size && maxh.distance[right] > maxh.distance[largest] {
		largest = right
	}

	if largest != i {
		maxh.distance[i], maxh.distance[largest] = maxh.distance[largest], maxh.distance[i]
		maxh.idxs[i], maxh.idxs[largest] = maxh.idxs[largest], maxh.idxs[i]

		maxh.HeapifyDown(largest)
	}
}

func (maxh *DistanceMaxHeap) HeapifyUp(i int32) {
	for i > 0 && maxh.distance[i] > maxh.distance[parent(i)] {
		maxh.distance[i], maxh.distance[parent(i)] = maxh.distance[parent(i)], maxh.distance[i]
		maxh.idxs[i], maxh.idxs[parent(i)] = maxh.idxs[parent(i)], maxh.idxs[i]

		i = parent(i)
	}
}

// utility functions
func left_child(i int32) int32 {
	return 2*i + 1
}

func right_child(i int32) int32 {
	return 2*i + 2
}

func parent(i int32) int32 {
	return (i - 1) / 2
}
