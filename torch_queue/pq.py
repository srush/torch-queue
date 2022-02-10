import torch


def merge(a, b, av, bv):
    """
    Merge two sorted key tensors `a` and `b` as well as corresponding
    int value tensors `av` and `bv`
    """
    n = a.shape[-1]
    B = a.shape[0]
    Bs = torch.arange(B).view(-1, 1)
    ordera = torch.searchsorted(a, b) + torch.arange(n)
    orderb = torch.searchsorted(b, a, right=True) + torch.arange(n)
    out = torch.zeros(a.shape[:-1] + (a.shape[-1] + b.shape[-1],))
    out[Bs, ordera] = b
    out[Bs, orderb] = a
    outv = torch.zeros(a.shape[:-1] + (a.shape[-1] + b.shape[-1],)).long()
    outv[Bs, ordera] = bv
    outv[Bs, orderb] = av
    return (
        out[..., : a.shape[-1]].contiguous(),
        out[..., a.shape[-1] :].contiguous(),
        outv[..., : a.shape[-1]],
        outv[..., a.shape[-1] :],
    )


def make_path(index):
    """
    Given a size of a node to add get the binary path to reach it.
    """
    order = "{0:b}".format(index + 1)
    order = torch.tensor(list(map(int, order)))
    out = [1]
    for i in range(1, len(order)):
        out.append(order[i] + 2 * out[-1])
    return torch.tensor(out) - 1


class Heap:
    def __init__(self, group_size, total_size, batch=1):
        """
        Create a heap over vectors of `group_size` that
        can expand to `group_size * total_size` nodes with
        independent `batch`es.
        """
        self.size = 0
        self.storage = torch.zeros(batch, total_size, group_size)
        self.storage.fill_(1.0e9)
        self.values = torch.zeros(batch, total_size, group_size).long()
        self.lengths = torch.zeros(batch, total_size)
        self.batch = batch

    def insert_heapify(self, items, values, pos, order):
        "Internal"
        node_id = order[pos]
        head = self.storage[:, node_id]
        hvalues = self.values[:, node_id]
        # self.lengths[:, node_id] = items.shape[0]
        if pos == order.shape[0] - 1:
            head[:] = items
            self.values[:, node_id, :] = values
        else:

            head[:], items[:], hvalues[:], values[:] = merge(
                head, items, hvalues, values
            )
            self.insert_heapify(items, values, pos + 1, order)

    def insert(self, keys, values, sorted=False):
        """
        Insert a batch of group_size keys `keys`  with corresponding
        integer `values`.
        """
        items = keys
        if not sorted:
            items, order = torch.sort(items, dim=-1)
            values = values[:, order]
        self.insert_heapify(items, values, 0, make_path(self.size))
        self.size = self.size + 1

    def delete_heapify(self, node_ids):
        if (node_ids >= self.size).any():
            return
        Bs = torch.arange(self.batch)
        c = torch.stack(((node_ids + 1) * 2 - 1, (node_ids + 1) * 2), dim=1)
        top = self.storage[Bs, node_ids]
        topv = self.values[Bs, node_ids]
        c_l = self.storage[Bs, c[:, 0]]
        c_r = self.storage[Bs, c[:, 1]]
        c_lv = self.values[Bs, c[:, 0]]
        c_rv = self.values[Bs, c[:, 1]]
        ins = torch.where(c_l[..., -1] < c_r[..., -1], 0, 1)
        s, l = c[Bs, ins], c[Bs, 1 - ins]
        small, self.storage[Bs, l, :], smallv, self.values[Bs, l, :] = merge(
            c_l, c_r, c_lv, c_rv
        )
        (
            self.storage[Bs, node_ids, :],
            self.storage[Bs, s, :],
            self.values[Bs, node_ids, :],
            self.values[Bs, s, :],
        ) = merge(top, small.contiguous(), topv, smallv)
        self.delete_heapify(s)

    def delete_min(self):
        """
        Pop and delete a batch of group_size keys and values.
        """
        items = self.storage[:, 0].clone()
        values = self.values[:, 0].clone()
        if self.size == 1:
            self.storage[:, 0] = 1.0e9
            self.values[:, 0] = -1
            return items, values
        path = make_path(self.size - 1)
        self.storage[:, 0, :] = self.storage[:, path[-1]]
        self.values[:, 0, :] = self.values[:, path[-1]]
        self.storage[:, path[-1]] = 1.0e9
        self.values[:, path[-1]] = -1
        self.delete_heapify(torch.zeros(self.batch).long())
        self.size = self.size - 1
        return items, values


# TESTS

import hypothesis
from hypothesis import example, given, strategies as st


@given(
    st.lists(
        st.lists(st.integers(min_value=-50, max_value=50), min_size=32, max_size=32),
        min_size=1,
        max_size=2,
    )
)
@example([[1, 2, 3, 4], [4, 5, 3, 4]])
def test_sort(ls):
    ls2 = [list(sorted(l)) for l in ls]
    size = len(ls[0])
    group = 2
    batch = len(ls)
    heap = Heap(group, size, batch)
    for i in range(size // group):
        x = torch.tensor([l[i * group : (i + 1) * group] for l in ls]).float()
        x, _ = x.sort(dim=-1)
        x = x.view(batch, group)
        print(x)
        heap.insert(x, x.long())
        print(heap.storage)
    ks, vs = [], []
    for j in range(size // group):
        k, v = heap.delete_min()
        ks.append(k)
        vs.append(v)
    print(ks)
    for b in range(batch):
        ls = []
        lsv = []
        for j in range(size // group):
            ls += ks[j][b].tolist()
            lsv += vs[j][b].tolist()

        for j in range(1, len(ls)):
            assert ls[j] >= ls[j - 1]
            assert ls2[b][j] == ls[j]
            assert ls[j] == lsv[j]


def test_head():
    heap = Heap(4, 16, batch=1)
    x = torch.arange(4).view(1, 4)


def test_merge():
    out_a, out_b, av, bv = merge(
        torch.tensor([[1.0, 2, 5]]),
        torch.tensor([[1.0, 4, 6]]),
        torch.tensor([[1, 2, 5]]),
        torch.tensor([[1, 4, 6]]),
    )

    assert out_a.tolist() == [[1, 1, 2]]
    assert out_b.tolist() == [[4, 5, 6]]
    assert av.tolist() == [[1, 1, 2]]
    assert bv.tolist() == [[4, 5, 6]]
    out_a, out_b, av, bv = merge(
        torch.tensor([[1.0, 2, 5], [1.0, 4, 6]]),
        torch.tensor([[1.0, 4, 6], [1.0, 2, 5]]),
        torch.tensor([[1, 2, 5], [1, 4, 6]]),
        torch.tensor([[1, 4, 6], [1, 2, 5]]),
    )
    assert out_a[1].tolist() == [1, 1, 2]
    assert out_b[1].tolist() == [4, 5, 6]
    assert av[1].tolist() == [1, 1, 2]
    assert bv[1].tolist() == [4, 5, 6]


def test_path():
    # assert make_path(0) == []
    assert make_path(1).tolist() == [0, 1]
    assert make_path(2).tolist() == [0, 2]
    assert make_path(3).tolist() == [0, 1, 3]
    assert make_path(4).tolist() == [0, 1, 4]
    assert make_path(5).tolist() == [0, 2, 5]
