#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define WARP_NUM 32

struct edge {
    unsigned int u;
    unsigned int v;
    unsigned int w;
};

__global__ void pulling_kernel(struct edge * L, int * dist_prev, int * dist_curr, int numEdges, int numVertices) {

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

bool arrays_different(const int n, const int * arr1, const int * arr2) {
    for (int i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) {
            return true;
        }
    }
    return false;
}

void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum, int nEdges) {
    setTime();

    /*
     * Do all the things here!
     **/

        // Sorting part
    // std::sort(peeps->begin(), peeps->end(), [] (initial_vertex const& a, initial_vertex const& b) { return a.vertexValue.distance < b.vertexValue.distance; });

    std::cout << "Size: " << peeps->size() << std::endl;
    

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

    // Copy host mem to device mem
    struct edge * d_L;
    int * d_dist_prev;
    int * d_dist_curr;
    cudaMalloc((void **)&d_L, nEdges * sizeof(struct edge));
    cudaMalloc((void **)&d_dist_prev, n * sizeof(int));
    cudaMalloc((void **)&d_dist_curr, n * sizeof(int));
    cudaMemcpy(d_L, L, nEdges * sizeof(struct edge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_prev, dist_prev, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist_curr, dist_curr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Timing code

    // Bellman Ford algorithm loop
    for (int i = 0; i < vertex_num - 1; i++) {
        // Invoke kernel
        pulling_kernel<<<blockNum, blockSize>>>(d_L, d_dist_prev, d_dist_curr, nEdges, n);
        cudaDeviceSynchronize();

        std::cout << "Iteration: " << i << " complete." << std::endl;

        // Copy results from device
        cudaMemcpy(dist_prev, d_dist_prev, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(dist_curr, d_dist_curr, n * sizeof(int), cudaMemcpyDeviceToHost);

        if (arrays_different(n, dist_prev, dist_curr)) {

            // std::cout << "Copied from device:" << std::endl;
            // for (int i = 0; i < n; i++) {
            //     std::cout << "dist_prev[" << i << "] = " << dist_prev[i] << std::endl;
            // }

            // for (int i = 0; i < n; i++) {
            //     std::cout << "dist_curr[" << i << "] = " << dist_curr[i] << std::endl;
            // }

            // std::cin.get();

            // swap prev and curr
            memcpy(dist_prev, dist_curr, n * sizeof(int));

            // std::cout << "Swapped mem: " << std::endl;
            // for (int i = 0; i < n; i++) {
            //     std::cout << "dist_prev[" << i << "] = " << dist_prev[i] << std::endl;
            // }

            // for (int i = 0; i < n; i++) {
            //     std::cout << "dist_curr[" << i << "] = " << dist_curr[i] << std::endl;
            // }

            // std::cin.get();

            // copy updated mem to device
            cudaMemcpy(d_dist_prev, dist_prev, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_dist_curr, dist_curr, n * sizeof(int), cudaMemcpyHostToDevice);
        } else {
            break;
        }
    }

    // After all iterations get final result
    cudaMemcpy(dist_curr, d_dist_curr, n * sizeof(int), cudaMemcpyDeviceToHost);


    // for (int i = 0; i < vertex_num; i++) {
    //     std::cout << "Vertex " << i << " dist: " << dist_curr[i] << std::endl;
    //     std::cin.get();
    // }





    // std::cout << "Num vertices: " << vertex_num << std::endl;
    // std::cout << "Num edges: " << edge_num << std::endl;

    std::cout << "Took " << getTime() << "ms.\n";
}
