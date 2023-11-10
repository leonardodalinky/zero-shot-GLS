#pragma once
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

/// @brief Huffman tree node. ONLY should be used by Huffman Tree.
/// @tparam K Key type.
/// @tparam F Frequency type.
template <typename K, typename F> struct Node {
    K key;
    F freq;
    int idx;
    int left_idx;
    int right_idx;

    Node(K key, F freq, int idx)
        : key(key), freq(freq), idx(idx), left_idx(-1), right_idx(-1) {}
    Node(F freq, int idx, int left_idx, int right_idx)
        : freq(freq), idx(idx), left_idx(left_idx), right_idx(right_idx) {}

    bool is_leaf() const { return left_idx < 0 && right_idx < 0; }
    bool operator<(const Node &other) const { return freq < other.freq; }
    bool operator>(const Node &other) const { return freq > other.freq; }
    bool operator<=(const Node &other) const { return freq <= other.freq; }
    bool operator>=(const Node &other) const { return freq >= other.freq; }
    bool operator==(const Node &other) const { return freq == other.freq; }
};

template <typename K, typename F> struct Tree {
    std::vector<Node<K, F>> nodes;

    /// @brief Get clone of the code book
    std::unordered_map<K, std::string> get_code_book() const {
        return std::unordered_map<K, std::string>(code_book);
    }

    static Tree from_freqs(const std::unordered_map<K, F> &freqs);

    void build_code_book(bool larger_as_zero) {
        if (nodes.size() == 0) {
            return;
        }
        std::string code = "";
        _build_code_book(root_idx, code, larger_as_zero);
    }

  private:
    Tree(std::vector<Node<K, F>> nodes, size_t root_idx)
        : nodes(nodes), root_idx(root_idx) {}
    size_t root_idx;
    std::unordered_map<K, std::string> code_book;

    struct NodeCompare {
        bool operator()(const Node<K, F> &lhs, const Node<K, F> &rhs) const {
            return lhs.freq > rhs.freq;
        }
    };

    void _build_code_book(int node_idx, std::string code, bool larger_as_zero) {
        if (node_idx < 0) {
            return;
        }
        const auto &node = nodes[node_idx];
        if (node.is_leaf()) {
            code_book[node.key] = code;
        } else {
            if (larger_as_zero) {
                _build_code_book(node.left_idx, code + "0", larger_as_zero);
                _build_code_book(node.right_idx, code + "1", larger_as_zero);
            } else {
                _build_code_book(node.left_idx, code + "1", larger_as_zero);
                _build_code_book(node.right_idx, code + "0", larger_as_zero);
            }
        }
    }
};

template <typename K, typename F>
Tree<K, F> Tree<K, F>::from_freqs(const std::unordered_map<K, F> &freqs) {
    if (freqs.size() == 0) {
        return Tree(std::vector<Node<K, F>>(), 0);
    } else if (freqs.size() == 1) {
        std::vector<Node<K, F>> new_nodes;
        new_nodes.reserve(2);
        Node<K, F> node(freqs.cbegin()->first, freqs.cbegin()->second, 0);
        new_nodes.push_back(node);
        new_nodes.push_back(Node<K, F>(node.freq, 1, 0, -1));
        return Tree(new_nodes, 1);
    } else {
        // construct basic nodes
        std::vector<Node<K, F>> new_nodes;
        new_nodes.reserve(freqs.size() * 2 - 1);
        for (const auto &kv : freqs) {
            new_nodes.push_back(
                Node<K, F>(kv.first, kv.second, new_nodes.size()));
        }

        std::priority_queue<Node<K, F>, std::vector<Node<K, F>>, NodeCompare>
            pq;
        for (const auto &node : new_nodes) {
            pq.push(node);
        }

        while (pq.size() > 1) {
            const auto right_node = pq.top();
            pq.pop();
            const auto left_node = pq.top();
            pq.pop();
            const auto new_node =
                Node<K, F>(left_node.freq + right_node.freq, new_nodes.size(),
                           left_node.idx, right_node.idx);
            new_nodes.push_back(new_node);
            pq.push(new_node);
        }

        return Tree(new_nodes, new_nodes.size() - 1);
    }
};
