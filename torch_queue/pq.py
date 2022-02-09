import torch

def merge(a, b):
    n = a.shape[-1]
    ordera = torch.searchsorted(a, b) + torch.arange(n)
    orderb = torch.searchsorted(b, a, right=True) + torch.arange(n)
    out = torch.zeros(a.shape[:-1] + (a.shape[-1] + b.shape[-1], ))
    out[..., ordera] = b
    out[..., orderb] = a
    return out[..., :a.shape[-1]], out[..., a.shape[-1]:]

def test_merge():
    out_a, out_b = merge(torch.tensor([1., 2, 5]), torch.tensor([1., 4, 6]))
    assert out_a.tolist() == [1, 1, 2]
    assert out_b.tolist() == [4, 5, 6]
    out_a, out_b = merge(torch.tensor([[1., 2, 5], [1., 4, 6]]),
                         torch.tensor([[1., 4, 6], [1., 2, 5]]))
    assert out_a[1].tolist() == [1, 1, 2]
    assert out_b[1].tolist() == [4, 5, 6]

def make_path(index):
    order = "{0:b}".format(index + 1)
    order = torch.tensor(list(map(int, order)))
    out = [1]
    for i in range(1, len(order)):
        out.append(order[i] + 2 * out[-1])
    return torch.tensor(out) - 1

def test_path():
    # assert make_path(0) == []
    assert make_path(1).tolist() == [0, 1]
    assert make_path(2).tolist() == [0, 2]
    assert make_path(3).tolist() == [0, 1, 3]
    assert make_path(4).tolist() == [0, 1, 4]
    assert make_path(5).tolist() == [0, 2, 5]
        
# def insert_heapify(items, pos, order, node):
#     if node is None:
#         return [items, [None, None]]
#     head, c = node
#     new_head, new_items = merge(head, items)
#     if order[pos] == "0":
#         c[0] = insert_heapify(new_items, pos+1, order, c[0])
#     else:
#         c[1] = insert_heapify(new_items, pos+1, order, c[1])
#     return [new_head, c]




def children(index):
    return (index * 2, index * 2 + 1)


class Heap:
    def __init__(self, group_size, total_size):
        self.size = 0
        self.storage = torch.zeros(1, total_size, group_size)
        self.lengths = torch.zeros(1, total_size)
        
    def insert_heapify(self, items, pos, order):
        node_id = order[pos]
        head = self.storage[:, node_id]
        if self.lengths[0, node_id] == 0:
            assert(pos == order.shape[0] - 1)
            self.lengths[0, node_id] = items.shape[0]
            head[:] = items
        else:
            head[:], items[:] = merge(head, items)
            self.lengths[0, node_id] = items.shape[0]
            self.insert_heapify(items, pos+1, order)

        
    def insert(self, items):
        if self.lengths[0, 0] == 0:
            self.storage[0, 0, :] = items
            self.lengths[0, 0] = items.shape[0]
            self.size = self.size + 1
        else:
            path = make_path(self.size)
            self.insert_heapify(items, 0, path)
            self.size = self.size + 1

    def delete_heapify(self, node_id):
        c = children(node_id)
        if self.lengths[0, c[0]] == 0 or self.lengths[0, node_id] == 0:
            return
        
        head = self.storage[0, node_id]
        c_l = self.storage[0, c[0]]
        if self.lengths[0, c[1]] == 0:
            head[:], c_l[:] = merge(head, c_l)
        else:
            c_r = self.storage[0, c[1]]
            s, l = (c[0], c[1]) if c_l[-1] < c_r[-1] else (c[1], c[0])
            small, self.storage[0, l, :] = merge(c_l, c_r)
            head[:], self.storage[0, s, :] = merge(head, small)
            self.delete_heapify(s)

    def delete_min(self):
        items = self.storage[:, 0].clone()
        if self.size == 1:
            self.storage[:, 0] = 0
            self.lengths[:, 0] = 0
            return items
        path = make_path(self.size-1)
        self.storage[:, 0, :] = self.storage[:, path[-1]]
        self.storage[:, path[-1]] = 0.0
        self.lengths[:, path[-1]] = 0
        self.delete_heapify(0)
        self.size = self.size - 1
        return items
        
    # def insert2(self, items):
    #     if self.top is None:
    #         self.top = (items, [None, None])
    #         return
    #     order = "{0:b}".format(self.size)
    #     self.top = insert_heapify(items, 0, order, self.top)
    #     self.size = self.size + 1
        
    # def delete_min(self):
    #     if self.size == 0:
    #         return self.top[0]
    #     ret, _ = self.top
    #     order = "{0:b}".format(self.size-1)
    #     cur = self.top
    #     for i in range(len(order)):
    #         _, c = cur
    #         parent = cur
    #         cur = c[int(order[i])]

    #     head, c = self.top
    #     self.top = [cur[0], c]
    #     parent[1][int(order[i])] = None
    #     self.size = self.size - 1
    #     self.top = delete_heapify(self.top)
    #     return ret


def test_head():
 
    heap = Heap(4, 16)
    x = torch.arange(4)
    heap.insert(torch.tensor([[1, 2, 3, 4]]))
    for i in range(5):
        heap.insert(x * 5 + float(i))
    print(heap.storage)
    for i in range(5):
        print(heap.delete_min())
    assert(False)
