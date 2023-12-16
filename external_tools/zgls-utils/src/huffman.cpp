#include "huffman.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::unordered_map<unsigned, std::string>
huffman_table_from_indices_frequencies(
    std::unordered_map<unsigned, double> indices_freqs, bool larger_as_zero) {
    auto tree = Tree<unsigned, double>::from_freqs(indices_freqs);

    tree.build_code_book(larger_as_zero);

    return tree.get_code_book();
}

PYBIND11_MODULE(_huffman, m) {
    m.doc() =
        "pybind11 plugin for huffman encoding"; // optional module docstring
    m.def("huffman_table_from_indices_frequencies",
          &huffman_table_from_indices_frequencies,
          "Build a huffman table from indices frequencies");
}
