#include <vector>
#include <iostream>
#include <algorithm>

#include "stdlib.h"
#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include "enumerations.hpp"
#include "graph.h"

#define WARP_NUM 32

__global__ void neighborHandling_kernel() {

}

// Determines the number of to process edges in each warp
// changed_mask is a mask of all edges marked as changed in L, (0 = unchanged, 1 = changed)
// X is an array where X[warpid] denotes the number of edges to process in each warp
__global__ void count_edges(
    const int * changed_mask,
    int * X,
    const int numEdges) {

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    int warp_id = thread_id / WARP_NUM;
    int num_warps = num_threads % WARP_NUM ? num_threads / WARP_NUM + 1 : num_threads / WARP_NUM;

    int iter = numEdges % num_threads ? numEdges / num_threads + 1 : numEdges / num_threads;

    for (int i = 0; i < iter; i++) {
        int dataid = thread_id + i * num_threads;
        if (dataid < numEdges) {
            int adj_warp_id = warp_id + i * num_warps;

            if (changed_mask[dataid] == 1)
                atomicAdd(&X[adj_warp_id], 1);
        }
    }
}

// Uses exclusive parallel prefix sum to determine the offset of each warps first to-process edge
// X is an array of the number of to process edge in each warp (X[i] = # tpe in warp i)
// Y will be parallel prefix sum  of X (gets offset of start of tpe set in X)
__global__ void get_offset(
    const int * X,
    int * Y,
    const int numWarps) {

    extern __shared__ int a[];

    int tid = threadIdx.x;

    if (tid == 0)
        a[0] = 0;
    else 
        a[tid] = X[tid-1];
    __syncthreads();

    for (int offset = 1; offset < numWarps; offset *= 2) {
        if (tid >= offset) {
            int newval = a[tid-offset] + a[tid];
            __syncthreads();
            a[tid] = newval;
        }
    }

    Y[tid] = a[tid];
}

void work_efficient_out_core(
    const struct edge * L, 
    const int * dist_prev, 
    int * dist_curr, 
    const int numVertices, 
    int numEdges, 
    const int blockSize, 
    const int blockNum) {

    numEdges = 100;

    // Create mask array
    int * mask = (int *)calloc(numEdges, sizeof(int));

    mask[0] = 1;
    mask[12] = 1;
    mask[13] = 1;
    mask[31] = 1;

    mask[60] = 1;
    mask[61] = 1;

    mask[80] = 1;

    mask[99] = 1;
    mask[98] = 1;

    // Create sum array

    std::cout << "Num Edges: " << numEdges << std::endl;
    std::cout << "Warp Num: " << WARP_NUM << std::endl;
    std::cout << "Num Edges mod Warp Num: " << numEdges % WARP_NUM << std::endl;

    int warps = numEdges % WARP_NUM ? numEdges / WARP_NUM + 1 : numEdges / WARP_NUM;
    int * X = (int *)calloc(warps, sizeof(int));

    // for (int i = 0; i < warps; i++) {
    //     std::cout << "X[" << i << "] = " << X[i] << std::endl;
    //     std::cin.get();
    // }

    int * Y = (int *)calloc(warps, sizeof(int));

    // Allocate mem on device
    struct edge * d_L;
    int * d_dist_prev;
    int * d_dist_curr;
    int * d_mask;
    int * d_X;
    int * d_Y;
    cudaMalloc((void **)&d_L, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist_prev, numVertices * sizeof(int));
    cudaMalloc((void **)&d_dist_curr, numVertices * sizeof(int));
    cudaMalloc((void **)&d_mask, numEdges * sizeof(int));
    cudaMalloc((void **)&d_X, warps * sizeof(int));
    cudaMalloc((void **)&d_Y, warps * sizeof(int));
    cudaMemcpy(d_L, L, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, numEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, warps * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, warps * sizeof(int), cudaMemcpyHostToDevice);

    // test step 1
    count_edges<<<blockNum, blockSize>>>(d_mask, d_X, numEdges);

    cudaMemcpy(X, d_X, warps * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Warps: " << warps << std::endl;

    for (int i = 0; i < warps; i++) {
        std::cout << "X[" << i << "] = " << X[i] << std::endl;
        std::cin.get();
    }

    std::cout << "Computing prefix sum..." << std::endl;

    // test step 2
    get_offset<<<1, warps, warps * sizeof(int)>>>(d_X, d_Y, warps);

    cudaMemcpy(Y, d_Y, warps * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < warps; i++) {
        std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
        std::cin.get();
    }

    

    // Bellman Ford Algorithm Loop
    // for (int i = 0; i < numVertices - 1; i++) {

    // }


}

void work_efficient_in_core() {

}

void neighborHandler(
    std::vector<edge> * peeps, 
    const int blockSize, 
    const int blockNum, 
    const int numVertices, 
    const enum SyncMode syncMethod) {


    setTime();

    // Create edge list array
    int numEdges = peeps->size();
    struct edge * L = (struct edge *)malloc(sizeof(struct edge) * numEdges);
    std::vector<edge>::iterator itr;
    int edge_num = 0;
    for (itr = peeps->begin(); itr < peeps->end(); ++itr) {
        L[edge_num].u = itr->u;
        L[edge_num].v = itr->v;
        L[edge_num].w = itr->w;
        edge_num++;
    }

    int * dist_prev = (int *)malloc(numVertices * sizeof(int));
    int * dist_curr = (int *)malloc(numVertices * sizeof(int));

    for (int i = 0; i < numVertices; i++) {
        if (i == 0) {
            dist_prev[i] = 0;
            dist_curr[i] = 0;
        } else {
            dist_prev[i] = SSSP_INF;
            dist_curr[i] = SSSP_INF;
        }
    }

    switch (syncMethod) {
        case OutOfCore:
            work_efficient_out_core(L, dist_prev, dist_curr, numVertices, numEdges, blockSize, blockNum);
            break;
        case InCore:
            work_efficient_in_core();
            break;
        default:
            std::cout << "Invalid processing method" << std::endl;
    }


    std::cout << "Took " << getTime() << "ms.\n";
}