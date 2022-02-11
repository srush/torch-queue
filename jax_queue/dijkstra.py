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
nodes[:SIZE, 0] = ((np.rand(SIZE) - 0.5) / 4.0) + (0.5 - (parabola * parabola))
nodes[0] = np.tensor([0.25, 0.1])
nodes[SIZE - 1] = np.tensor([0.25, -1.1])
distance = nodes[:, None, :] - nodes[None, :, :]
edges = (distance * distance).sum(-1).sqrt() + torch.eye(SIZE + BLOCK_SIZE) * INF
edges[:, -BLOCK_SIZE:] = INF
for i in range(SIZE + BLOCK_SIZE):
    edges[i, :i], edges[i, i + BLOCK_SIZE :] = INF, INF
edges[0, SIZE - 1] = INF


# Data structures
def shortest_path(edges):
    heap, msize = pq.make_heap(BLOCK_SIZE, 2**8)
    B, L = np.full((BLOCK_SIZE), INF), np.full((1, BLOCK_SIZE), -1)
    B = B.at[0].set(0)
    L = L.at[0].set(0)
    Q = jnp.zeros((SIZE + BLOCK_SIZE))
    D = jnp.full((SIZE + BLOCK_SIZE), INF)

    # Dijkstra
    final = None
    def inner_loop(heap, B, L, Q, D):
        # Find next node
        def find_pos(s, heap, B, L):
            start = np.argwhere(Q[L[0]] == 0, size=1, fill_value=-1)
            heap, B, L = jax.lax.cond(
                start[0] == -1,
                lambda: pq.delete_min(heap, msize),
                lambda: heap, B, L)
            return start[0], heap, B, L
        pos, heap, B, L = jax.while(lambda s: s[0], find_pos, (-1, heap, B, L))
        l, b = L[pos], B[pos]
        Q = Q.at[l].set(1.0)
        D = D.at[l].set(b)

        # Expand
        scores = (
            Q[l : l + BLOCK_SIZE] * INF
            + b
            + edges[l, l : l + BLOCK_SIZE]
        )
        scores = torch.where(scores < D[l : l + BLOCK_SIZE], scores, INF)
        indices = torch.arange(l, l + BLOCK_SIZE)
        heap = pq.insert(heap, msize, scores, indices, sorted=False)

        # Refresh buffer
        heap, B2, L2 = pq.delete_min(heap, msize)
        B = B.at[ : -(pos + 1)].set(B[0, pos + 1 :]).at[-(pos + 1) :].set(INF)
        L = L.at[ : -(pos + 1)].set(L[0, pos + 1 :]).at[-(pos + 1) :].set(-1)
        B, B2, L, L2 = pq.merge(B, B2, L, L2)
        heap = pq.insert(heap, msize, B2, L2, sorted=True)
        return heap, B, L, Q, D
    heap, B, L, Q, D = jax.while(lambda s: s[3][SIZE-1] == 1,
                                 inner_loop,
                                 (heap, B, L, Q, D))
    return D[SIZE-1]
path = jax.grad(shortest_path)
path(jnp.array(edges))

# Plot
plt.figure(figsize=(20, 30))
edge = edges[:SIZE, :SIZE].cpu().detach().numpy()
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

d = {}
for x in edges.grad.nonzero():
    d[x[0].item(), x[1].item()] = "x"
nx.draw_networkx_edges(G, pos, width=10.0, edgelist=d.keys(), edge_color="red")
plt.tight_layout()
plt.savefig("Graph.png", format="PNG")
