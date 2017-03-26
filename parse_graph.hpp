#ifndef PARSE_GRAPH_HPP
#define PARSE_GRAPH_HPP

#include <fstream>

#include "initial_graph.hpp"

#define SSSP_INF 1073741824

namespace parse_graph {
	uint parse(
		std::ifstream& inFile,
		std::vector<initial_vertex>& initGraph,
		const long long arbparam,
		const bool nondirected );
}

#endif	//	PARSE_GRAPH_HPP
