import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import pq
import networkx as nx
import matplotlib.pyplot as plt

# Make Graph
SIZE, BLOCK_SIZE = 2056, 128
INF = np.array(1.0e5)
nodes = np.zeros((SIZE + BLOCK_SIZE, 2))
nodes[:SIZE, 1] = -(np.arange(SIZE) / SIZE)
parabola = (np.arange(SIZE) / SIZE) - 0.5
nodes[:SIZE, 0] = ((np.random.rand(SIZE) - 0.5) / 4.0) + (0.5 - (parabola * parabola))
nodes[0] = np.array([0.25, 0.1])
nodes[SIZE - 1] = np.array([0.25, -1.1])
distance = nodes[:, None, :] - nodes[None, :, :]
edges = np.sqrt((distance * distance).sum(-1)) + np.eye(SIZE + BLOCK_SIZE) * INF
edges[:, -BLOCK_SIZE:] = INF
for i in range(SIZE + BLOCK_SIZE):
    edges[i, :i], edges[i, i + BLOCK_SIZE :] = INF, INF
edges[0, SIZE - 1] = INF
INF = jnp.array(1.0e5)

# Data structures
def shortest_path(edges, msize):
    heap = pq.make_heap(BLOCK_SIZE, 2**msize)
    B, L = jnp.full((BLOCK_SIZE), INF), jnp.full((BLOCK_SIZE), -1, dtype=int)
    B = B.at[0].set(0)
    L = L.at[0].set(0)
    Q = jnp.zeros((SIZE + BLOCK_SIZE))
    D = jnp.full((SIZE + BLOCK_SIZE), INF)

    # Dijkstra
    final = None
    def inner_loop(_, args):
        heap, B, L, Q, D = args
        # Find next node
        def find_pos(_, args):
            s, heap, B, L = args
            start = jnp.argwhere(Q[L] == 0, size=1, fill_value=-1)
            heap, B, L = jax.lax.cond(
                pred=(start[0] == -1).all(),
                true_fun = lambda : pq.delete_min(heap, msize),
                false_fun = lambda : (heap, B, L))
            return start.reshape(1)[0], heap, B, L
        pos, heap, B, L = jax.lax.fori_loop(0, 10, find_pos,
                                            (-1, heap, B, L))

        def update(args):
            heap, B, L, Q, D = args
            l, b = L[pos], B[pos]
            Q = Q.at[l].set(1.0)
            D = D.at[l].set(b)
            # Expand
            bs = jnp.arange(BLOCK_SIZE)
            scores = (
                Q[bs + l] * INF
                + b
                + edges[l, bs + l]
            )
            scores = jnp.where(scores < D[bs + l], scores, INF)
            order = np.argsort(scores, kind='stable')
            heap = pq.insert(*heap, msize, scores[order], (bs+l)[order])

            # Refresh buffer
            heap, B2, L2 = pq.delete_min(heap, msize)
            i = jnp.arange(BLOCK_SIZE)
            B = jnp.where(i < (BLOCK_SIZE - (pos+1)), jnp.roll(B, -(pos+1)), INF)
            L = jnp.where(i < (BLOCK_SIZE - (pos+1)), jnp.roll(L, -(pos+1)), -1) 
            B, B2, L, L2 = pq.merge(B, B2, L, L2)
            heap = pq.insert(*heap, msize, B2, L2)
            return heap, B, L, Q, D
        return jax.lax.cond((pos!=-1).all(), update, lambda x: x, (heap, B, L, Q, D))
    heap, B, L, Q, D = jax.lax.fori_loop(
        0, 2*SIZE,
        inner_loop,
        (heap, B, L, Q, D))
    return D[SIZE-1]
# path = jax.grad(shortest_path)
# from jax.config import config

# config.update('jax_disable_jit', True)

# shortest_path(jnp.array(edges), 8)
path = jax.jit(jax.grad(lambda x: shortest_path(x, 8)))
out = path(jnp.array(edges))
print(out)
# print("done1")
# out = path(jnp.array(edges))
# Plot
plt.figure(figsize=(20, 30))
edge = np.array(edges[:SIZE, :SIZE])
G = nx.from_numpy_matrix(edge * (edge < 100))
pos = {i: nodes[i].tolist() for i in range(SIZE)}
nx.draw(
    G,
    pos,
    node_size=150,
    labels={0: "START", SIZE - 1: "END"},
    width=0.02,
    node_color="yellow",
    font_size=60,
    font_weight="bold",
)

print(np.array(out).nonzero())
d = {}
nz = np.array(out).nonzero()
for i in range(nz[0].shape[0]):
    d[nz[0][i], nz[1][i]] = "x"
nx.draw_networkx_edges(G, pos, width=10.0, edgelist=d.keys(), edge_color="red")
plt.tight_layout()
plt.savefig("Graph.png", format="PNG")

