#ifndef PARSE_GRAPH_HPP
#define PARSE_GRAPH_HPP


#include <vector>
#include <fstream>
#include "user_specified_structures.h"

#define SSSP_INF 1073741824

namespace parse_graph {
	uint parse(
		std::ifstream& inFile,
		std::vector<struct edge>& initGraph,
		const long long arbparam,
		const bool nondirected );
}

#endif	//	PARSE_GRAPH_HPP
