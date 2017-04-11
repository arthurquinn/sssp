#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include "enumerations.hpp"
#include "stdio.h"
#include "user_specified_structures.h"
#include <stdlib.h>
#include <string.h>

#define WARP_NUM 32

// One iteration of segment scan
__device__ void segment_scan(const int lane, const struct edge * L, const int * dist_prev, int * dist_curr) {
    int me;
    int other;

    if (lane >= 1 && L[threadIdx.x].v == L[threadIdx.x - 1].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 1].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 1].u] + L[threadIdx.x - 1].w;
        dist_curr[L[threadIdx.x].v] = min(dist_curr[L[threadIdx.x].v], min(me, other));
    } 
    if (lane >= 2 && L[threadIdx.x].v == L[threadIdx.x - 2].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 2].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 2].u] + L[threadIdx.x - 2].w;     
        dist_curr[L[threadIdx.x].v] = min(dist_curr[L[threadIdx.x].v], min(me, other));
    } 
    if (lane >= 4 && L[threadIdx.x].v == L[threadIdx.x - 4].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 4].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 4].u] + L[threadIdx.x - 4].w;
        dist_curr[L[threadIdx.x].v] = min(dist_curr[L[threadIdx.x].v], min(me, other));
    } 
    if (lane >= 8 && L[threadIdx.x].v == L[threadIdx.x - 8].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 8].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 8].u] + L[threadIdx.x - 8].w;
        dist_curr[L[threadIdx.x].v] = min(dist_curr[L[threadIdx.x].v], min(me, other));
    } 
    if (lane >= 16 && L[threadIdx.x].v == L[threadIdx.x - 16].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 16].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 16].u] + L[threadIdx.x - 16].w;
        dist_curr[L[threadIdx.x].v] = min(dist_curr[L[threadIdx.x].v], min(me, other));
    }
}

__global__ void bellman_ford_outcore_kernel(const struct edge * L, const int * dist_prev, int * dist_curr, const int numEdges, const int numVertices) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / WARP_NUM;
    int laneid = threadIdx.x % WARP_NUM;

    int load = numEdges % WARP_NUM == 0 ? numEdges / WARP_NUM : numEdges / WARP_NUM + 1;
    int beg = load * warp_id;
    int end = min(numEdges, beg + load);
    beg = beg + laneid;

    for (int i = beg; i < end; i += 32) {
        int u = L[i].u;
        int v = L[i].v;
        int w = L[i].w;

        int newDist = dist_prev[u] == SSSP_INF ? SSSP_INF : dist_prev[u] + w;
        if (newDist < dist_prev[v]) {
            atomicMin(&dist_curr[v], newDist);
        }
    }
}

__global__ void bellman_ford_incore_kernel(const struct edge * L, int * dist, const int numEdges, const int numVertices, int * anyChange) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / WARP_NUM;
    int laneid = threadIdx.x % WARP_NUM;

    int load = numEdges % WARP_NUM == 0 ? numEdges / WARP_NUM : numEdges / WARP_NUM + 1;
    int beg = load * warp_id;
    int end = min(numEdges, beg + load);
    beg = beg + laneid;

    for (int i = beg; i < end; i += 32) {
        int u = L[i].u;
        int v = L[i].v;
        int w = L[i].w;

        int temp_dist = dist[u] == SSSP_INF ? SSSP_INF : dist[u] + w;
        if (temp_dist < dist[v]) {
            int old = atomicMin(&dist[v], temp_dist);
            if (old > temp_dist) {
                *anyChange = 1;
            }
        }
    }
}

__global__ void bellman_ford_segment_scan_kernel(const struct edge * L, const int * dist_prev, int * dist_curr, const int numEdges, const int numVertices) {
    // Set up shared mem
    extern __shared__ int a[];
    int * shared_dist_prev = a;
    int * shared_dist_curr = a + numVertices;
    struct edge * shared_L = (struct edge *)(a + 2 * numVertices); 


    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    int iter = numEdges % thread_num ? numEdges / thread_num + 1 : numEdges / thread_num;
    // int warp_iter = blockDim.x / WARP_NUM ? blockDim.x / WARP_NUM + 1 : blockDim.x / WARP_NUM;

    for (int i = 0; i < iter; i++)
    {
        // Assign data into shared memory arrays
        int dataid = thread_id + i * thread_num;
        if (dataid < numEdges)
        {
            shared_L[dataid] = L[dataid];
            shared_dist_prev[L[dataid].u] = dist_prev[shared_L[dataid].u];
            shared_dist_curr[L[dataid].u] = dist_curr[shared_L[dataid].u];
            __syncthreads();

            int lane = threadIdx.x % WARP_NUM;
            segment_scan(lane, shared_L, shared_dist_prev, shared_dist_curr);

            __syncthreads();

            atomicMin(&dist_curr[shared_L[dataid].v], shared_dist_curr[shared_L[dataid].v]);
        }
    }
}

bool arrays_different(const int n, const int * arr1, const int * arr2) {
    for (int i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) {
            return true;
        }
    }
    return false;
}

double bellman_ford_segment_scan(const struct edge * L, int * dist_prev, int * dist_curr, const int numEdges, const int numVertices, const int blockNum, const int blockSize) {
    // Copy host mem to device
    struct edge * d_L;
    int * d_dist_prev;
    int * d_dist_curr;
    cudaMalloc((void **)&d_L, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist_prev, numVertices * sizeof(int));
    cudaMalloc((void **)&d_dist_curr, numVertices * sizeof(int));
    cudaMemcpy(d_L, L, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_curr, dist_curr, numVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Timing Code
    double b = 0.0;
    // Bellman ford algorithm loop
    for (int i = 0; i < numVertices - 1; i++) {
        int smem_size = numEdges * sizeof(struct edge) + 2 * numVertices * sizeof(int);
        setTime();
        bellman_ford_segment_scan_kernel<<<blockNum, blockSize, smem_size>>>(d_L, d_dist_prev, d_dist_curr, numEdges, numVertices);
        cudaDeviceSynchronize();
        b = b + getTime();

        cudaMemcpy(dist_prev, d_dist_prev, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    
        
        if (arrays_different(numVertices, dist_prev, dist_curr)) {


            // swap prev and curr
            memcpy(dist_prev, dist_curr, numVertices * sizeof(int));


            // copy updated mem to device
            cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist_curr, dist_curr, numVertices * sizeof(int), cudaMemcpyHostToDevice);
        } else {
            // std::cout << "Completed after " << i << " iterations" << std::endl;
            break;
        }
    }

    // After all iterations get final result
    cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    //std::cout << "Took " << b << "ms.\n";

    return b;
}

// After algorithm is complete, dist_curr will contain shortest path to all vertices
double bellman_ford_outcore(const struct edge * L, int * dist_prev, int * dist_curr, const int numEdges, const int numVertices, const int blockNum, const int blockSize) {
    // Copy host mem to device mem
    struct edge * d_L;
    int * d_dist_prev;
    int * d_dist_curr;
    cudaMalloc((void **)&d_L, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist_prev, numVertices * sizeof(int));
    cudaMalloc((void **)&d_dist_curr, numVertices * sizeof(int));
    cudaMemcpy(d_L, L, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_curr, dist_curr, numVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Timing code
    double b = 0.0;
    // Bellman Ford algorithm loop
    for (int i = 0; i < numVertices; i++) {
        // Invoke kernel
        setTime();
        bellman_ford_outcore_kernel<<<blockNum, blockSize>>>(d_L, d_dist_prev, d_dist_curr, numEdges, numVertices);
        cudaDeviceSynchronize();
        b = b + getTime();

        // Copy results from device
        cudaMemcpy(dist_prev, d_dist_prev, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

        if (arrays_different(numVertices, dist_prev, dist_curr)) {

            // swap prev and curr
            memcpy(dist_prev, dist_curr, numVertices * sizeof(int));

            // copy updated mem to device
            cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist_curr, dist_curr, numVertices * sizeof(int), cudaMemcpyHostToDevice);
        } else {
            // std::cout << "Completed after " << i << " iterations" << std::endl;
            break;
        }
    }

    // After all iterations get final result
    cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Took " << b << "ms.\n";

    return b;
}

double bellman_ford_incore(const struct edge * L, int * dist, const int numEdges, const int numVertices, const int blockNum, const int blockSize) {
    // Copy host mem to device
    struct edge * d_L;
    int * d_dist;
    int * d_anyChange;
    cudaMalloc((void **)&d_L, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist, numVertices * sizeof(int));
    cudaMalloc((void **)&d_anyChange, sizeof(int));
    cudaMemcpy(d_L, L, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, numVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Bellman ford algorithm loop
    double b = 0.0;
    int * anyChange = (int *)malloc(sizeof(int));
    for (int i = 0; i < numVertices - 1; i++) {

        // Invoke kernel
        setTime();
        bellman_ford_incore_kernel<<<blockNum, blockSize>>>(d_L, d_dist, numEdges, numVertices, d_anyChange);
        cudaDeviceSynchronize();
        b = b + getTime();
        // Check if any value was changed
        cudaMemcpy(anyChange, d_anyChange, sizeof(int), cudaMemcpyDeviceToHost);

        // If a value was changed, set any change back to 0 and continue; else break out of loop
        if (*anyChange == 1) {
            *anyChange = 0;
            cudaMemcpy(d_anyChange, anyChange, sizeof(int), cudaMemcpyHostToDevice);
        } else {
            // std::cout << "Completed after " << i << " iterations" << std::endl;
            break;
        }
         //std::cout << "Took " << getTime() << "ms.\n";
    }
    
    // dist will store results of algorithm
    cudaMemcpy(dist, d_dist, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "Took " << b << "ms.\n";

    return b;
}


struct time_result puller(std::vector<edge> * peeps, int blockSize, int blockNum, int numVertices, enum SyncMode syncMethod, enum SmemMode smemMode, std::ofstream& outputFile) {
    

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

    // Allocate space for dist_prev and dist_curr
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

    double time = 0.0f;
    switch (syncMethod) {
        case InCore:
           // std::cout << "Starting incore" << std::endl;
            time = bellman_ford_incore(L, dist_curr, numEdges, numVertices, blockNum, blockSize);
            break;
        case OutOfCore:
            if (smemMode == UseNoSmem) {
               // std::cout << "Starting outcore" << std::endl;
                time = bellman_ford_outcore(L, dist_prev, dist_curr, numEdges, numVertices, blockNum, blockSize);
            }
            else if (smemMode == UseSmem) {
               // std::cout << "Starting segment scan" << std::endl;
                time = bellman_ford_segment_scan(L, dist_prev, dist_curr, numEdges, numVertices, blockNum, blockSize);
            } else {
                std::cout << "Invalid shared memory specification" << std::endl;
            }
            break;
        default:
            std::cout << "Invalid core processing method" << std::endl;
            break;
    }
    
    int c;
    for ( c = 0 ; c < numVertices ; c++ )
    {   
        outputFile << c <<":"<< dist_curr[c] << "\n";       
    }

    outputFile.close();

    struct time_result ret_time;

    ret_time.comp_time = time;
    ret_time.filter_time = 0.0;

    return ret_time;
}
