#pragma once
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

template <typename I, typename F> struct Node {
    using NodePtr = typename std::shared_ptr<Node<I, F>>;

    bool is_leaf;
    I index;
    F freq;
    NodePtr left;
    NodePtr right;

    Node(I index, F freq, bool is_leaf)
        : is_leaf(is_leaf), index(index), freq(freq) {}
    Node(F freq, NodePtr left, NodePtr right)
        : is_leaf(false), freq(freq), left(left), right(right) {}

    bool operator<(const Node &other) const { return freq < other.freq; }
    bool operator>(const Node &other) const { return freq > other.freq; }
    bool operator<=(const Node &other) const { return freq <= other.freq; }
    bool operator>=(const Node &other) const { return freq >= other.freq; }
    bool operator==(const Node &other) const { return freq == other.freq; }
};

template <typename I, typename F> struct Tree {
    using NodePtr = typename Node<I, F>::NodePtr;

    NodePtr root = nullptr;

    /// @brief Get clone of the code book
    std::unordered_map<I, std::string> get_code_book() const {
        return std::unordered_map<I, std::string>(code_book);
    }

    static Tree from_freqs(const std::unordered_map<I, F> &freqs);

    void build_code_book(bool larger_as_zero) {
        if (root == nullptr) {
            return;
        }
        std::string code = "";
        _build_code_book(root, code, larger_as_zero);
    }

  private:
    Tree(NodePtr root) : root(root) {}
    std::unordered_map<I, std::string> code_book;

    struct NodeCompare {
        bool operator()(const NodePtr &lhs, const NodePtr &rhs) const {
            return lhs->freq > rhs->freq;
        }
    };

    void _build_code_book(NodePtr node, std::string code, bool larger_as_zero) {
        if (node == nullptr) {
            return;
        } else if (node->is_leaf) {
            code_book[node->index] = code;
        } else {
            if (larger_as_zero) {
                _build_code_book(node->left, code + "0", larger_as_zero);
                _build_code_book(node->right, code + "1", larger_as_zero);
            } else {
                _build_code_book(node->left, code + "1", larger_as_zero);
                _build_code_book(node->right, code + "0", larger_as_zero);
            }
        }
    }
};

template <typename I, typename F>
Tree<I, F> Tree<I, F>::from_freqs(const std::unordered_map<I, F> &freqs) {
    if (freqs.size() == 0) {
        return Tree(nullptr);
    } else if (freqs.size() == 1) {
        NodePtr left = std::make_shared<Node<I, F>>(
            freqs.begin()->first, freqs.begin()->second, true);
        auto parent = std::make_shared<Node<I, F>>(left->freq, left, nullptr);
        return Tree(parent);
    } else {
        std::priority_queue<NodePtr, std::vector<NodePtr>, NodeCompare> pq;
        for (const auto &kv : freqs) {
            pq.push(std::make_shared<Node<I, F>>(kv.first, kv.second, true));
        }

        while (pq.size() > 1) {
            NodePtr right = pq.top();
            pq.pop();
            NodePtr left = pq.top();
            pq.pop();
            auto parent = std::make_shared<Node<I, F>>(left->freq + right->freq,
                                                       left, right);
            pq.push(parent);
        }
        return Tree(pq.top());
    }
};
