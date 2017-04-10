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

// Determines the number of to process edges in each warp
// changed_mask is a mask of all edges marked as changed in L, (0 = unchanged, 1 = changed)
// X is an array where X[warpid] denotes the number of edges to process in each warp
__global__ void count_edges(
    const struct edge * L,
    const int * changed_mask,
    int * X,
    int * num_tpe,
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

            if (changed_mask[L[dataid].u] == 1) {
                atomicAdd(&X[adj_warp_id], 1);
                atomicAdd(num_tpe, 1);
            }
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

// step 3 of filtering process
// Copies all tpe (to-process edges) into the array T based on their offset
// L is edge list, changed_mask shows which edges are tpe, T will be the filtered array of edges
__global__ void copy_tpe(
    const struct edge * L,
    const int * changed_mask,
    struct edge * T,
    const int numEdges) {

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    int warp_id = thread_id / WARP_NUM;
    int num_warps = num_threads % WARP_NUM ? num_threads / WARP_NUM + 1 : num_threads / WARP_NUM;
    int lane = thread_id % WARP_NUM;

    int iter = numEdges % num_threads ? numEdges / num_threads + 1 : numEdges / num_threads;

    for (int i = 0; i < iter; i++) {
        int dataid = thread_id + i * num_threads;
        if (dataid < numEdges) {
            int adj_warp_id = warp_id + i * num_warps;
            int bal = __ballot(changed_mask[L[dataid].u] == 1);
            int localid = __popc(bal<<(32-lane));
            T[localid+adj_warp_id] = L[dataid];
        }
    }

}

__global__ void bmf_tpe_outcore_kernel(
    const struct edge * T,
    const int num_tpe,
    const int * dist_prev,
    int * dist_curr,
    int * changed_mask) {

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / WARP_NUM;
    int laneid = threadIdx.x % WARP_NUM;

    int load = num_tpe % WARP_NUM == 0 ? num_tpe / WARP_NUM : num_tpe / WARP_NUM + 1;
    int beg = load * warp_id;
    int end = min(num_tpe, beg + load);
    beg = beg + laneid;

    for (int i = beg; i < end; i += 32) {
        int u = T[i].u;
        int v = T[i].v;
        int w = T[i].w;

        int newDist = dist_prev[u] == SSSP_INF ? SSSP_INF : dist_prev[u] + w;
        if (newDist < dist_prev[v]) {
            changed_mask[v] = 1;
            atomicMin(&dist_curr[v], newDist);
        }
    }
}

__global__ void bmf_tpe_incore_kernel(
    const struct edge * T,
    const int num_tpe,
    int * dist,
    int * changed_mask) {

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / WARP_NUM;
    int laneid = threadIdx.x % WARP_NUM;

    int load = num_tpe % WARP_NUM == 0 ? num_tpe / WARP_NUM : num_tpe / WARP_NUM + 1;
    int beg = load * warp_id;
    int end = min(num_tpe, beg + load);
    beg = beg + laneid;

    for (int i = beg; i < end; i += 32) {
        int u = T[i].u;
        int v = T[i].v;
        int w = T[i].w;

        int newDist = dist[u] == SSSP_INF ? SSSP_INF : dist[u] + w;
        if (newDist < dist[v]) {
            changed_mask[v] = 1;
            atomicMin(&dist[v], newDist);
        }
    }
}

void work_efficient_out_core(
    const struct edge * L, 
    int * dist_prev, 
    int * dist_curr, 
    const int numVertices, 
    int numEdges, 
    const int blockSize, 
    const int blockNum) {

    // T will store tpe
    struct edge * T = (struct edge *)malloc(sizeof(struct edge) * numEdges);

    // changed_mask will store a mask containing a 1 flag for each change in the dist_curr arr
    int * changed_mask = (int *)calloc(numVertices, sizeof(int));

    // Start out T with all edges
    for (int i = 0; i < numEdges; i++) {
        T[i] = L[i];
    }

    int warpsNeeded = numEdges % WARP_NUM ? numEdges / WARP_NUM + 1 : numEdges / WARP_NUM;
    warpsNeeded = min(64, warpsNeeded);
    int * X = (int *)calloc(warpsNeeded, sizeof(int));
    int * Y = (int *)calloc(warpsNeeded, sizeof(int));

    int * num_tpe = (int *)malloc(sizeof(int));
    *num_tpe = numEdges;

    // Allocate memory on device
    struct edge * d_L;
    struct edge * d_T;
    int * d_dist_prev;
    int * d_dist_curr;
    int * d_changed_mask;
    int * d_X;
    int * d_Y;
    int * d_num_tpe;
    cudaMalloc((void **)&d_L, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_T, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist_prev, numVertices * sizeof(int));
    cudaMalloc((void **)&d_dist_curr, numVertices * sizeof(int));
    cudaMalloc((void **)&d_changed_mask, numVertices * sizeof(int));
    cudaMalloc((void **)&d_X, warpsNeeded * sizeof(int));
    cudaMalloc((void **)&d_Y, warpsNeeded * sizeof(int));
    cudaMalloc((void **)&d_num_tpe, sizeof(int));
    cudaMemcpy(d_L, L, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, T, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_curr, dist_curr, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_changed_mask, changed_mask, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, warpsNeeded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, warpsNeeded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_tpe, num_tpe, sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < numVertices - 1; i++) {
        bmf_tpe_outcore_kernel<<<blockNum, blockSize>>>(d_T, *num_tpe, d_dist_prev, d_dist_curr, d_changed_mask);
        cudaDeviceSynchronize();

        // Swap dist curr into dist prev
        cudaMemcpy(dist_prev, d_dist_prev, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << "Copied from device:" << std::endl;
        // for (int j = 0; j < 20; j++) {
        //     std::cout << "dist_prev[" << j << "] = " << dist_prev[j] << std::endl;
        // }

        // for (int j = 0; j < 20; j++) {
        //     std::cout << "dist_curr[" << j << "] = " << dist_curr[j] << std::endl;
        // }

        // std::cin.get();

        memcpy(dist_prev, dist_curr, numVertices * sizeof(int));
        cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);

        *num_tpe = 0;
        cudaMemcpy(d_num_tpe, num_tpe, sizeof(int), cudaMemcpyHostToDevice);
        count_edges<<<blockNum, blockSize>>>(d_L, d_changed_mask, d_X, d_num_tpe, numEdges);
        cudaDeviceSynchronize();

        cudaMemcpy(num_tpe, d_num_tpe, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "numtpe: " << *num_tpe << std::endl;
        // std::cin.get();

        if (*num_tpe == 0) {
            std::cout << "I'm done here after " << i << " iterations" << std::endl;
            break;
        }

        get_offset<<<1, warpsNeeded, warpsNeeded * sizeof(int)>>>(d_X, d_Y, warpsNeeded);
        cudaDeviceSynchronize();

        copy_tpe<<<blockNum, blockSize>>>(d_L, d_changed_mask, d_T, numEdges);
        cudaDeviceSynchronize();

        cudaMemcpy(T, d_T, *num_tpe * sizeof(struct edge), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < *num_tpe; i++) {
        //     std::cout << "Edge: [" << T[i].u << ", " << T[i].v << "]" << std::endl;
        // }

        // Reset mask
        for (int j = 0; j < numVertices; j++) {
            changed_mask[j] = 0;
        }
        cudaMemcpy(d_changed_mask, changed_mask, numVertices * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_T, T, *num_tpe * sizeof(struct edge), cudaMemcpyHostToDevice);
    }
}

void work_efficient_in_core(
    const struct edge * L, 
    int * dist,
    const int numVertices, 
    int numEdges, 
    const int blockSize, 
    const int blockNum) {

    // T will store tpe
    struct edge * T = (struct edge *)malloc(sizeof(struct edge) * numEdges);

    // changed_mask will store a mask containing a 1 flag for each change in the dist_curr arr
    int * changed_mask = (int *)calloc(numVertices, sizeof(int));

    // Start out T with all edges
    for (int i = 0; i < numEdges; i++) {
        T[i] = L[i];
    }

    int warpsNeeded = numEdges % WARP_NUM ? numEdges / WARP_NUM + 1 : numEdges / WARP_NUM;
    warpsNeeded = min(64, warpsNeeded);
    int * X = (int *)calloc(warpsNeeded, sizeof(int));
    int * Y = (int *)calloc(warpsNeeded, sizeof(int));

    int * num_tpe = (int *)malloc(sizeof(int));
    *num_tpe = numEdges;

    // Allocate memory on device
    struct edge * d_L;
    struct edge * d_T;
    int * d_dist;
    int * d_changed_mask;
    int * d_X;
    int * d_Y;
    int * d_num_tpe;
    cudaMalloc((void **)&d_L, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_T, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist, numVertices * sizeof(int));
    cudaMalloc((void **)&d_changed_mask, numVertices * sizeof(int));
    cudaMalloc((void **)&d_X, warpsNeeded * sizeof(int));
    cudaMalloc((void **)&d_Y, warpsNeeded * sizeof(int));
    cudaMalloc((void **)&d_num_tpe, sizeof(int));
    cudaMemcpy(d_L, L, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, T, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_changed_mask, changed_mask, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, warpsNeeded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, warpsNeeded * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_tpe, num_tpe, sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < numVertices - 1; i++) {
        bmf_tpe_incore_kernel<<<blockNum, blockSize>>>(d_T, *num_tpe, d_dist, d_changed_mask);
        cudaDeviceSynchronize();

        std::cout << "Copied from device:" << std::endl;
        for (int j = 0; j < 20; j++) {
            std::cout << "dist[" << j << "] = " << dist[j] << std::endl;
        }

        std::cin.get();

        *num_tpe = 0;
        cudaMemcpy(d_num_tpe, num_tpe, sizeof(int), cudaMemcpyHostToDevice);
        count_edges<<<blockNum, blockSize>>>(d_L, d_changed_mask, d_X, d_num_tpe, numEdges);
        cudaDeviceSynchronize();

        cudaMemcpy(num_tpe, d_num_tpe, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "numtpe: " << *num_tpe << std::endl;
        // std::cin.get();

        if (*num_tpe == 0) {
            std::cout << "I'm done here after " << i << " iterations" << std::endl;
            break;
        }

        get_offset<<<1, warpsNeeded, warpsNeeded * sizeof(int)>>>(d_X, d_Y, warpsNeeded);
        cudaDeviceSynchronize();

        copy_tpe<<<blockNum, blockSize>>>(d_L, d_changed_mask, d_T, numEdges);
        cudaDeviceSynchronize();

        cudaMemcpy(T, d_T, *num_tpe * sizeof(struct edge), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < *num_tpe; i++) {
        //     std::cout << "Edge: [" << T[i].u << ", " << T[i].v << "]" << std::endl;
        // }

        // Reset mask
        for (int j = 0; j < numVertices; j++) {
            changed_mask[j] = 0;
        }
        cudaMemcpy(d_changed_mask, changed_mask, numVertices * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_T, T, *num_tpe * sizeof(struct edge), cudaMemcpyHostToDevice);
    }
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
            work_efficient_in_core(L, dist_curr, numVertices, numEdges, blockSize, blockNum);
            break;
        default:
            std::cout << "Invalid processing method" << std::endl;
    }


    std::cout << "Took " << getTime() << "ms.\n";
}