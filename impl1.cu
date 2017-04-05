#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"
#include "enumerations.hpp"

#define WARP_NUM 32

struct edge {
    unsigned int u;
    unsigned int v;
    unsigned int w;
};

// One iteration of segment scan
__device__ void segment_scan(const int lane, const struct edge * L, const int * dist_prev, int * dist_curr) {
    int me;
    int other;

    if (lane >= 1 && L[threadIdx.x].v == L[threadIdx.x - 1].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 1].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 1].u] + L[threadIdx.x - 1].w;
        dist_curr[threadIdx.x] = min(me, other);
    } 
    if (lane >= 2 && L[threadIdx.x].v == L[threadIdx.x - 2].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 2].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 2].u] + L[threadIdx.x - 2].w;
        dist_curr[threadIdx.x] = min(me, other);
    } 
    if (lane >= 4 && L[threadIdx.x].v == L[threadIdx.x - 4].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 4].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 4].u] + L[threadIdx.x - 4].w;
        dist_curr[threadIdx.x] = min(me, other);        
    } 
    if (lane >= 8 && L[threadIdx.x].v == L[threadIdx.x - 8].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 8].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 8].u] + L[threadIdx.x - 8].w;
        dist_curr[threadIdx.x] = min(me, other);        
    } 
    if (lane >= 16 && L[threadIdx.x].v == L[threadIdx.x - 16].v) {
        me = dist_prev[L[threadIdx.x].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x].u] + L[threadIdx.x].w;
        other = dist_prev[L[threadIdx.x - 16].u] == SSSP_INF ? SSSP_INF : dist_prev[L[threadIdx.x - 16].u] + L[threadIdx.x - 16].w;
        dist_curr[threadIdx.x] = min(me, other);        
    }
}

__global__ void bellman_ford_outcore_kernel(const struct edge * L, int * dist_prev, int * dist_curr, const int numEdges, const int numVertices) {
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
    extern __shared__ char a[];
    int * s_dist_prev = (int *)a;
    int * s_dist_curr = (int *)(a + numVertices * sizeof(int));
    struct edge * s_L = (struct edge *)(a + numVertices * sizeof(int) * 2);

    // Set up thread identifiers
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = thread_id / WARP_NUM;
    int lane_id = threadIdx.x % WARP_NUM;

    // Determine thread load
    int load = numEdges % WARP_NUM == 0 ? numEdges / WARP_NUM : numEdges / WARP_NUM + 1;
    int beg = load * warp_id;
    int end = min(numEdges, beg + load);
    beg = beg + lane_id;

    if (thread_id < numEdges) {

        s_L[threadIdx.x] = L[thread_id];
        s_dist_prev[threadIdx.x] = dist_prev[thread_id];
        s_dist_curr[threadIdx.x] = dist_curr[thread_id];

        printf("%d\n", threadIdx.x);
        __syncthreads();

        segment_scan(lane_id, s_L, s_dist_prev, s_dist_curr);

        printf("s_dist_curr[%d] = %d\n", threadIdx.x, s_dist_curr[threadIdx.x]);
        __syncthreads();
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

void bellman_ford_segment_scan(const struct edge * L, int * dist_prev, int * dist_curr, const int numEdges, const int numVertices, const int blockNum, const int blockSize) {
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

    // Bellman ford algorithm loop
    for (int i = 0; i < numVertices - 1; i++) {

        std::cout << "Iteration " << i << std::endl;
        int smem_size = numEdges * sizeof(struct edge) + numVertices * sizeof(int) + numVertices * sizeof(int);
        bellman_ford_segment_scan_kernel<<<blockNum, blockSize, smem_size>>>(d_L, d_dist_prev, d_dist_curr, numEdges, numVertices);
        cudaDeviceSynchronize();

        cudaMemcpy(dist_prev, d_dist_prev, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

        if (arrays_different(numVertices, dist_prev, dist_curr)) {

            std::cout << "Copied from device:" << std::endl;
            for (int i = 0; i < numVertices; i++) {
                std::cout << "dist_prev[" << i << "] = " << dist_prev[i] << std::endl;
            }

            for (int i = 0; i < numVertices; i++) {
                std::cout << "dist_curr[" << i << "] = " << dist_curr[i] << std::endl;
            }

            std::cin.get();

            // swap prev and curr
            memcpy(dist_prev, dist_curr, numVertices * sizeof(int));

            std::cout << "Swapped mem: " << std::endl;
            for (int i = 0; i < numVertices; i++) {
                std::cout << "dist_prev[" << i << "] = " << dist_prev[i] << std::endl;
            }

            for (int i = 0; i < numVertices; i++) {
                std::cout << "dist_curr[" << i << "] = " << dist_curr[i] << std::endl;
            }

            std::cin.get();

            // copy updated mem to device
            cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist_curr, dist_curr, numVertices * sizeof(int), cudaMemcpyHostToDevice);
        } else {
            break;
        }
    }

    // After all iterations get final result
    cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
}

// After algorithm is complete, dist_curr will contain shortest path to all vertices
void bellman_ford_outcore(const struct edge * L, int * dist_prev, int * dist_curr, const int numEdges, const int numVertices, const int blockNum, const int blockSize) {
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

    // Bellman Ford algorithm loop
    for (int i = 0; i < numVertices - 1; i++) {
        // Invoke kernel
        bellman_ford_outcore_kernel<<<blockNum, blockSize>>>(d_L, d_dist_prev, d_dist_curr, numEdges, numVertices);
        cudaDeviceSynchronize();

        std::cout << "Iteration: " << i << " complete." << std::endl;

        // Copy results from device
        cudaMemcpy(dist_prev, d_dist_prev, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

        if (arrays_different(numVertices, dist_prev, dist_curr)) {

            // std::cout << "Copied from device:" << std::endl;
            // for (int i = 0; i < numVertices; i++) {
            //     std::cout << "dist_prev[" << i << "] = " << dist_prev[i] << std::endl;
            // }

            // for (int i = 0; i < numVertices; i++) {
            //     std::cout << "dist_curr[" << i << "] = " << dist_curr[i] << std::endl;
            // }

            // std::cin.get();

            // swap prev and curr
            memcpy(dist_prev, dist_curr, numVertices * sizeof(int));

            // std::cout << "Swapped mem: " << std::endl;
            // for (int i = 0; i < numVertices; i++) {
            //     std::cout << "dist_prev[" << i << "] = " << dist_prev[i] << std::endl;
            // }

            // for (int i = 0; i < numVertices; i++) {
            //     std::cout << "dist_curr[" << i << "] = " << dist_curr[i] << std::endl;
            // }

            // std::cin.get();

            // copy updated mem to device
            cudaMemcpy(d_dist_prev, dist_prev, numVertices * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist_curr, dist_curr, numVertices * sizeof(int), cudaMemcpyHostToDevice);
        } else {
            break;
        }
    }

    // After all iterations get final result
    cudaMemcpy(dist_curr, d_dist_curr, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
}

void bellman_ford_incore(const struct edge * L, int * dist, const int numEdges, const int numVertices, const int blockNum, const int blockSize) {
    // Copy host mem to device
    struct edge * d_L;
    int * d_dist;
    int * d_anyChange;
    cudaMalloc((void **)&d_L, numEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist, numVertices * sizeof(int));
    cudaMalloc((void **)&d_anyChange, sizeof(int));
    cudaMemcpy(d_L, L, numEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, numVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Timing Code

    // Bellman ford algorithm loop
    int * anyChange = (int *)malloc(sizeof(int));
    for (int i = 0; i < numVertices - 1; i++) {

        // Invoke kernel
        bellman_ford_incore_kernel<<<blockNum, blockSize>>>(d_L, d_dist, numEdges, numVertices, d_anyChange);
        cudaDeviceSynchronize();

        // Check if any value was changed
        cudaMemcpy(anyChange, d_anyChange, sizeof(int), cudaMemcpyDeviceToHost);

        // If a value was changed, set any change back to 0 and continue; else break out of loop
        if (*anyChange == 1) {
            *anyChange = 0;
            cudaMemcpy(d_anyChange, anyChange, sizeof(int), cudaMemcpyHostToDevice);
        } else {
            break;
        }
    }

    // dist will store results of algorithm
    cudaMemcpy(dist, d_dist, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // // Code to print out result array
    // for (int i = 0; i < numVertices; i++) {
    //     std::cout << dist[i] << " ";
    // }
    // std::cout << std::endl;
}


void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum, int nEdges, enum SyncMode syncMethod, enum SmemMode smemMode) {
    setTime();

    /*
     * Do all the things here!
     **/
    
    // Sorting part
    // std::sort(peeps->begin(), peeps->end(), [] (initial_vertex const& a, initial_vertex const& b) { return a.vertexValue.distance < b.vertexValue.distance; });

    // std::cout << "Size: " << peeps->size() << std::endl;
    

    // In order for parsing the peeps vector into L to work - peeps must be sorted 
    int n = peeps->size();

    struct edge * L = (struct edge *)malloc(sizeof(struct edge) * nEdges);
    int * dist_prev = (int *)malloc(sizeof(int) * n);
    int * dist_curr = (int *)malloc(sizeof(int) * n);

    std::vector<initial_vertex>::iterator itr;
    int vertex_num = 0;
    int edge_num = 0;
    for (itr = peeps->begin(); itr < peeps->end(); ++itr) {
        
        // std::cout << "Distance: " << itr->get_vertex_ref().distance << std::endl;
        
        dist_prev[vertex_num] = itr->get_vertex_ref().distance;
        dist_curr[vertex_num] = itr->get_vertex_ref().distance;

        // std::cout << "dist_prev[" << vertex_num << "] = " << itr->get_vertex_ref().distance << std::endl;
        // std::cout << "dist_curr[" << vertex_num << "] = " << itr->get_vertex_ref().distance << std::endl;

        // std::cout << "Number of neighbors: " << itr->nbrs.size() << std::endl;

        std::vector<neighbor>::iterator neighbor_itr;
        for (neighbor_itr = itr->nbrs.begin(); neighbor_itr < itr->nbrs.end(); ++neighbor_itr) {


            L[edge_num].u = vertex_num;
            L[edge_num].v = neighbor_itr->srcIndex;
            L[edge_num].w = neighbor_itr->edgeValue.weight;

            // std::cout << "L[" << edge_num << "].u = " << vertex_num << std::endl;
            // std::cout << "L[" << edge_num << "].v = " << neighbor_itr->srcIndex << std::endl;
            // std::cout << "L[" << edge_num << "].w = " << neighbor_itr->edgeValue.weight << std::endl;


            // std::cout << "Neighbor: " << neighbor_itr->srcIndex << std::endl;
            // std::cout << "Edge Weight: " << neighbor_itr->edgeValue.weight << std::endl;

            // std::cin.get();

            edge_num++;
        }

        vertex_num++;
        // std::cin.get();
    }

    switch (syncMethod) {
        case InCore:
            std::cout << "Starting incore" << std::endl;
            bellman_ford_incore(L, dist_curr, nEdges, n, blockNum, blockSize);
            break;
        case OutOfCore:
            if (smemMode == UseNoSmem) {
                std::cout << "Starting outcore" << std::endl;
                bellman_ford_outcore(L, dist_prev, dist_curr, nEdges, n, blockNum, blockSize);
            }
            else if (smemMode == UseSmem) {
                std::cout << "Starting segment scan" << std::endl;
                bellman_ford_segment_scan(L, dist_prev, dist_curr, nEdges, n, blockNum, blockSize);
            } else {
                std::cout << "Invalid shared memory specification" << std::endl;
            }
            break;
        default:
            std::cout << "Invalid core processing method" << std::endl;
            break;
    }

    for (int i = 0; i < n; i++) {
        std::cout << "" << dist_curr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Took " << getTime() << "ms.\n";
}
