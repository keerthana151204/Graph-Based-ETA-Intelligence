"""
GraphSAGE (mean aggregator) implemented from scratch in NumPy.

Why from scratch instead of torch_geometric:
  * zero heavy dependencies -> the benchmark reproduces anywhere, including CI;
  * it makes the mechanism explicit (neighbourhood mean -> concat -> linear ->
    ReLU), which is the point of putting "GraphSAGE" on a CV — being able to
    explain it, not just import it.

Algorithm (Hamilton, Ying & Leskovec 2017), mean variant:
    h_v^k = ReLU( W_k . CONCAT( h_v^{k-1},  MEAN_{u in N(v)} h_u^{k-1} ) )

We train it as a supervised node regression (predict each hub's mean outgoing
delay ratio, computed on TRAIN legs only). The penultimate layer activations are
taken as the node embeddings and fed to the downstream ETA model. Everything is
fit on the training graph; unseen test nodes get a zero embedding.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import scipy.sparse as sp

from . import config as C


def _normalised_adj(G: nx.DiGraph, nodes):
    """Row-normalised adjacency of the undirected projection (mean aggregation),
    with self-loops so a node also sees its own previous representation."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    Gu = G.to_undirected()
    rows, cols = [], []
    for u, v in Gu.edges():
        rows += [idx[u], idx[v]]
        cols += [idx[v], idx[u]]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = A + sp.eye(n)
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    Dinv = sp.diags(1.0 / deg)
    return (Dinv @ A).tocsr(), idx


def _node_input_features(G: nx.DiGraph, nodes):
    """Cheap structural input features for each node (standardised)."""
    indeg = np.array([G.in_degree(n) for n in nodes], float)
    outdeg = np.array([G.out_degree(n) for n in nodes], float)
    vol = np.array([sum(d.get("trips", 0) for _, _, d in G.out_edges(n, data=True)) for n in nodes], float)
    X = np.vstack([indeg, outdeg, np.log1p(vol)]).T
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X


def _node_target(G: nx.DiGraph, nodes):
    """Supervised target: mean outgoing corridor delay ratio (train graph)."""
    y = []
    for n in nodes:
        w = [d["weight"] for _, _, d in G.out_edges(n, data=True)]
        y.append(np.mean(w) if w else 1.0)
    return np.array(y, float)


class NumpyGraphSAGE:
    """2-layer mean-aggregator GraphSAGE with a linear regression head."""

    def __init__(self, in_dim, hidden, embed, lr, epochs, seed):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.normal(0, np.sqrt(2.0 / a), (a, b))
        # layer 1 takes concat(self, neigh) = 2*in_dim
        self.W1 = s(2 * in_dim, hidden)
        # layer 2 takes concat(self, neigh) = 2*hidden
        self.W2 = s(2 * hidden, embed)
        self.Wout = s(embed, 1)
        self.lr, self.epochs = lr, epochs

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0)

    def _forward(self, A, X):
        # Layer 1
        neigh1 = A @ X
        z1 = np.hstack([X, neigh1]) @ self.W1
        h1 = self._relu(z1)
        # Layer 2
        neigh2 = A @ h1
        z2 = np.hstack([h1, neigh2]) @ self.W2
        h2 = self._relu(z2)            # node embeddings
        yhat = (h2 @ self.Wout).ravel()
        cache = (X, neigh1, z1, h1, neigh2, z2, h2)
        return yhat, h2, cache

    def fit(self, A, X, y):
        n = len(y)
        for ep in range(self.epochs):
            yhat, h2, (X_, neigh1, z1, h1, neigh2, z2, h2_) = self._forward(A, X)
            err = (yhat - y)
            loss = np.mean(err ** 2)
            # ---- backprop (full-batch) ----
            dy = (2.0 / n) * err            # dL/dyhat
            dWout = h2.T @ dy[:, None]
            dh2 = dy[:, None] @ self.Wout.T
            dz2 = dh2 * (z2 > 0)
            cat1 = np.hstack([h1, neigh2])
            dW2 = cat1.T @ dz2
            dcat1 = dz2 @ self.W2.T
            dh1_self = dcat1[:, :h1.shape[1]]
            dh1_neigh = dcat1[:, h1.shape[1]:]
            dh1 = dh1_self + A.T @ dh1_neigh   # neigh path flows back through A
            dz1 = dh1 * (z1 > 0)
            cat0 = np.hstack([X_, neigh1])
            dW1 = cat0.T @ dz1
            # ---- SGD step ----
            self.Wout -= self.lr * dWout
            self.W2 -= self.lr * dW2
            self.W1 -= self.lr * dW1
            if ep == 0 or (ep + 1) % 40 == 0:
                print(f"      sage epoch {ep+1:>3}/{self.epochs}  mse={loss:.4f}")
        return self

    def embeddings(self, A, X):
        _, h2, _ = self._forward(A, X)
        return h2


def train_sage(G: nx.DiGraph):
    """Train SAGE on the train graph; return {node: embedding} dict."""
    nodes = list(G.nodes())
    A, idx = _normalised_adj(G, nodes)
    X = _node_input_features(G, nodes)
    y = _node_target(G, nodes)
    s = C.SETTINGS
    model = NumpyGraphSAGE(X.shape[1], s.sage_hidden_dim, s.sage_embed_dim,
                           s.sage_lr, s.sage_epochs, s.sage_seed).fit(A, X, y)
    emb = model.embeddings(A, X)
    return {n: emb[idx[n]] for n in nodes}, s.sage_embed_dim


def attach_sage(df, emb: dict, dim: int):
    """Concatenate source + destination embeddings as features; zero for unseen."""
    df = df.copy()
    zero = np.zeros(dim)
    src = np.vstack([emb.get(n, zero) for n in df[C.SOURCE]])
    dst = np.vstack([emb.get(n, zero) for n in df[C.DEST]])
    cols = []
    for i in range(dim):
        df[f"src_sage_{i}"] = src[:, i]; cols.append(f"src_sage_{i}")
    for i in range(dim):
        df[f"dst_sage_{i}"] = dst[:, i]; cols.append(f"dst_sage_{i}")
    return df, cols
