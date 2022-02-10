import torch
import pq
import networkx as nx
import matplotlib.pyplot as plt

torch.set_default_tensor_type("torch.cuda.FloatTensor")

# Make Graph
SIZE, BLOCK_SIZE = 2056, 64
INF = torch.tensor(1.0e5).float()
nodes = torch.zeros((SIZE + BLOCK_SIZE, 2))
nodes[:SIZE, 1] = -(torch.arange(SIZE) / SIZE)
parabola = (torch.arange(SIZE) / SIZE) - 0.5
nodes[:SIZE, 0] = ((torch.rand(SIZE) - 0.5) / 4.0) + (0.5 - (parabola * parabola))
nodes[0] = torch.tensor([0.25, 0.1])
nodes[SIZE - 1] = torch.tensor([0.25, -1.1])
distance = nodes[:, None, :] - nodes[None, :, :]
edges = (distance * distance).sum(-1).sqrt() + torch.eye(SIZE + BLOCK_SIZE).cuda() * INF
edges[:, -BLOCK_SIZE:] = INF
for i in range(SIZE + BLOCK_SIZE):
    edges[i, :i], edges[i, i + BLOCK_SIZE :] = INF, INF
edges[0, SIZE - 1] = INF
edges.requires_grad_(True)

# Data-structures
heap = pq.Heap(BLOCK_SIZE, 10000, 1)
B, L = torch.full((1, BLOCK_SIZE), INF), torch.full((1, BLOCK_SIZE), -1)
B[0, 0], L[0, 0] = 0, 0
Q = torch.zeros((1, SIZE + BLOCK_SIZE)).cuda()
D = torch.full((1, SIZE + BLOCK_SIZE), INF).cuda()

# Dijkstra
final = None
while True:
    # Find next node
    while True:
        start = (Q[0, L[0]] == 0).nonzero()
        if start.shape[0] == 0:
            B[:], L[:] = heap.delete_min()
        else:
            pos = start[0, 0]
            break
    l, b = L[0, pos], B[0, pos]
    Q[0, l], D[0, l] = 1.0, b

    if l == SIZE - 1:
        final = b
        break

    # Expand
    scores = (
        Q[:, l : l + BLOCK_SIZE] * INF
        + b
        + edges[l, l : l + BLOCK_SIZE].view(1, BLOCK_SIZE)
    )
    scores = torch.where(scores < D[:, l : l + BLOCK_SIZE], scores, INF)
    indices = torch.arange(l, l + BLOCK_SIZE).view(1, -1)
    heap.insert(scores, indices, sorted=False)

    # Refresh buffer
    B2, L2 = heap.delete_min()
    B[0, : -(pos + 1)], B[0, -(pos + 1) :] = B[0, pos + 1 :].clone(), INF
    L[0, : -(pos + 1)], L[0, -(pos + 1) :] = L[0, pos + 1 :].clone(), -1
    B, B2, L, L2 = pq.merge(B, B2, L, L2)
    heap.insert(B2, L2, sorted=True)
final.backward()
print(edges.grad)

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


