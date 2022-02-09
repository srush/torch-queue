import torch
import pq
import networkx as nx
import matplotlib.pyplot as plt
size = 256
INF = torch.tensor(1.e5).float()
nodes = torch.rand((size, 2))
nodes[0] = 0
nodes[-1] = 1.0
distance = nodes[:, None, :] - nodes[None, :, :]
edges = (distance * distance).sum(-1).sqrt() + torch.eye(size) * INF + (torch.rand((size, size)) > 0.3) * INF
edges.requires_grad_(True)

start = 0
heap = pq.Heap(size, 256, 1)
heap.insert(edges[0].view(1, size), torch.arange(size))
best, locations = torch.full((1, size), INF), torch.full((1, size), -1)
final = None

Q = torch.zeros((1, size))
D = torch.full((1, size), INF)
while True:
    best2, locations2 = heap.delete_min()
    best, best2, locations, locations2 = pq.merge(best, best2, locations, locations2)
    # print(best, locations)
    if locations[0, 0] == size-1:
        final = best[0, 0]
        break
    Q[0, locations[0, 0]] = 1.
    D[0, locations[0, 0]] = best[0, 0]
    scores = Q * INF + best[0, 0] + edges[locations[0, 0]].view(1, size)
    scores = torch.where(scores < D, scores, INF)
    heap.insert(scores, torch.arange(size))
    best[0, :-1] = best[0, 1:].clone()
    locations[0, :-1] = locations[0, 1:].clone()
    best[0, -1] = INF
    locations[0, -1] = -1
plt.figure(figsize=(50, 50))
edge = edges.detach().numpy()
G = nx.from_numpy_matrix(edge * (edge < 1e5))
pos = {i : nodes[i].tolist() for i in range(size)}
nx.draw(G, pos, node_size=500,
        labels = {0: 'START', 
                  size-1: 'END'},
        edge_width=0.2,
        node_color='yellow', font_size=20, font_weight='bold')
final.backward()

d = {}
for x in edges.grad.nonzero():
    d[x[0].item(), x[1].item()] = "x"

nx.draw_networkx_edges(
    G, pos,
    width=10.0,
    edgelist=d.keys(),
    edge_color='red'
)
plt.tight_layout()
plt.savefig("Graph.png", format="PNG")

print(edges.grad)
