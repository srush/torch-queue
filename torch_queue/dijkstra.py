import torch
import pq
import networkx as nx
import matplotlib.pyplot as plt
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Make Graph
SIZE, BLOCK_SIZE  = 2056, 128
INF = torch.tensor(1.e5).float()
nodes = torch.zeros((SIZE + BLOCK_SIZE, 2))
nodes[:SIZE, 1] = -(torch.arange(SIZE) / SIZE)
parabola = ((torch.arange(SIZE) / SIZE) -0.5) 
nodes[:SIZE, 0] = ((torch.rand(SIZE)-0.5) / 500.0) / (parabola * parabola + 0.05)
nodes[0] = torch.tensor([0.0, 0.1])
nodes[SIZE-1] = torch.tensor([0.0, -1.1] )
distance = nodes[:, None, :] - nodes[None, :, :]
edges = (distance * distance).sum(-1).sqrt() + torch.eye(SIZE+BLOCK_SIZE).cuda() * INF
edges[:, -BLOCK_SIZE:] = INF
for i in range(SIZE-BLOCK_SIZE):
    edges[i, :i], edges[i, i+BLOCK_SIZE:] = INF, INF
edges[0, SIZE-1] = INF
edges.requires_grad_(True)

# Data-structures
heap = pq.Heap(BLOCK_SIZE, 10000, 1)
best, locations = torch.full((1, BLOCK_SIZE), INF), torch.full((1, BLOCK_SIZE), -1)
best[0, 0], locations[0, 0] = 0, 0
Q = torch.zeros((1, SIZE+BLOCK_SIZE)).cuda()
D = torch.full((1, SIZE+BLOCK_SIZE), INF).cuda()

# Dijkstra
final = None
while True:
    # Find next node
    while True:
        start = (Q[0, locations[0]] == 0).nonzero()
        if start.shape[0] == 0:
            best[:], locations[:] = heap.delete_min()
        else:                  
            pos = start[0, 0]
            break
    l, b = locations[0, pos], best[0, pos]
    Q[0, l], D[0, l] = 1., b

    if l == SIZE-1:
        final = b
        break
    
    # Expand
    scores = Q[:, l:l+BLOCK_SIZE] * INF + b + edges[l, l:l+BLOCK_SIZE].view(1, BLOCK_SIZE)
    scores = torch.where(scores < D[:, l:l+BLOCK_SIZE], scores, INF)
    indices = torch.arange(l, l + BLOCK_SIZE).view(1, -1)
    heap.insert(scores, indices, sorted=False)

    # Refresh buffer
    best2, locations2 = heap.delete_min()
    best[0, :-(pos+1)], best[0, -(pos+1):] = best[0, pos+1:].clone(), INF
    locations[0, :-(pos+1)], locations[0, -(pos+1):] = locations[0, pos+1:].clone(), -1
    best, best2, locations, locations2 = pq.merge(best, best2, locations, locations2)
    heap.insert(best2, locations2, sorted=True)

# Plot
plt.figure(figsize=(20, 30))
edge = edges[:SIZE, :SIZE].cpu().detach().numpy()
G = nx.from_numpy_matrix(edge * (edge < 1e5))
pos = {i : nodes[i].tolist() for i in range(SIZE)}
nx.draw(G, pos, node_size=300,
        labels = {0: 'START', 
                  SIZE-1: 'END'},
        width=0.02,
        node_color='yellow', font_size=60, font_weight='bold')
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
