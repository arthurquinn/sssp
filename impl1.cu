#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

struct edge {
    unsigned int u;
    unsigned int v;
    unsigned int w;
};

__global__ void pulling_kernel(std::vector<initial_vertex> * peeps, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //offset will tell you who I am.
}

void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum, int nEdges){
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

        // std::cout << "Number of neighbors: " << itr->nbrs.size();

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

            edge_num++;
        }

        vertex_num++;
        // std::cin.get();
    }

    // struct edge * edgeList = malloc(sizeof(struct edge) * peeps.size());

    // // allocate device mem
    // for (std::vector<initial_vertex>::size_type i = 0; i != peeps.size(); i++) {

    // }

    // std::vector<initial_vertex> * d_peeps;

    // int * d_anyChange;

    std::cout << "Num vertices: " << vertex_num << std::endl;
    std::cout << "Num edges: " << edge_num << std::endl;

    std::cout << "Took " << getTime() << "ms.\n";
}
