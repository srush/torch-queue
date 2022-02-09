import torch

def merge(a, b):
    ordera = torch.searchsorted(a, b) + torch.arange(3)
    orderb = torch.searchsorted(b, a) + torch.arange(3)
    out = torch.zeros(a.shape[0] + b.shape[0])
    out[ordera] = b
    out[orderb] = a
    return out[:a.shape[0]], out[a.shape[0]:]


merge(torch.tensor([1., 2, 5]), torch.tensor([3., 4, 6]))

def insert_heap(heap, items):
    (top, size) = heap
    head, children = top
    new_head, new_items = merge(head, items)
    pass
    return new_head, new_children
