import jax.numpy as np
import jax
from functools import partial

@partial(np.vectorize, signature="(a),(a),(a),(a)->(a),(a),(a),(a)")
def merge(a, b, av, bv):
    """
    Merge two sorted key tensors `a` and `b` as well as corresponding
    int value tensors `av` and `bv`
    """
    n = a.shape[-1]
    ordera = np.searchsorted(a, b) + np.arange(n)
    orderb = np.searchsorted(b, a, side='right') + np.arange(n)
    out = np.zeros((a.shape[-1] + b.shape[-1],))
    out = out.at[ordera].set(b)
    out = out.at[orderb].set(a)
    outv = np.zeros((a.shape[-1] + b.shape[-1],))
    outv = outv.at[ordera].set(bv)
    outv = outv.at[orderb].set(av)
    return (
        out[: a.shape[-1]],
        out[a.shape[-1] :],
        outv[: a.shape[-1]],
        outv[a.shape[-1] :]
    )

@partial(jax.jit, static_argnums=(1,))
def make_path(index, bits):
    """
    Given a size of a node to add get the binary path to reach it.
    """
    mask = 2**np.arange(bits-1, -1, -1)
    x = np.bitwise_and(np.array(index+1).ravel(), mask) != 0
    _, x = jax.lax.scan(lambda c, a : (a + 2 * c, a + 2 * c), np.array(1), x[1:])
    return np.concatenate((np.array([0]), x - 1))


def make_heap(group_size, total_size):
    """
    Create a heap over vectors of `group_size` that
    can expand to `group_size * total_size` nodes with
    independent `batch`es.
    """
    size = np.zeros(1, dtype=int)
    key_store = np.full((total_size, group_size), 1.e5)
    val_store = np.zeros((total_size, group_size), dtype=int)
    return (key_store, val_store, size), int(np.log2(key_store.shape[0]))

@partial(jax.jit, static_argnums=3)
def insert(key_store, val_store, size, max_size, keys, values, sorted=False):
    """
    Insert a batch of group_size keys `keys`  with corresponding
    integer `values`.
    """
    if not sorted:
        order = np.argsort(keys)
        keys = keys[order]
        values = values[order]
    path = make_path(size, max_size)
    key_store, val_store = insert_heapify(key_store, val_store, keys, values, 0, path)
    size = size + 1
    return key_store, val_store, size

def insert_heapify(key_store, val_store, keys, values, pos, order):
    "Internal"
    n = order[pos]
    if pos == order.shape[0] - 1:
        key_store = key_store.at[n].set(keys)
        val_store = val_store.at[n].set(values)
    else:
        head, keys, hvalues, values = merge(
            key_store[n], keys, val_store[n], values
        )
        key_store = key_store.at[n].set(head)
        val_store = val_store.at[n].set(hvalues)
        key_store, val_store = insert_heapify(key_store, val_store, keys, values, pos + 1, order)
    return key_store, val_store

@partial(jax.jit, static_argnums=1)
def delete_min(heap, msize):
    """
    Pop and delete a batch of group_size keys and values.
    """
    key_store, val_store, size = heap
    keys = key_store[0]
    values = val_store[0]

    def one():
        return key_store.at[0].set(1.0e9), val_store.at[0].set(-1)
    def two():
        path = make_path(size - 1, msize)
        key_store2 = key_store.at[0].set(key_store[path[-1]])
        val_store2 = val_store.at[0].set(val_store[path[-1]])
        key_store2 = key_store2.at[path[-1]].set(1.0e9)
        val_store2 = val_store2.at[path[-1]].set(-1)
        return delete_heapify(key_store2, val_store2, size, msize, 0)


    key_store, val_store = jax.lax.cond((size == 1).all(), one, two)
    size = size - 1
    return (key_store, val_store, size), keys, values

def delete_heapify(key_store, val_store, size, max_size, n):
    if (n >= max_size):
        return key_store, val_store
    c = np.stack(((n + 1) * 2 - 1, (n + 1) * 2))
    top = key_store[n]
    topv = val_store[n]
    c_l = key_store[c[0]]
    c_r = key_store[c[1]]
    c_lv = val_store[c[0]]
    c_rv = val_store[c[1]]
    ins = np.where(c_l[-1] < c_r[-1], 0, 1)
    s, l = c[ins], c[1 - ins]

    small, k2, smallv, v2 = merge(c_l, c_r, c_lv, c_rv)
    key_store = key_store.at[l].set(k2)
    val_store = val_store.at[l].set(v2)

    k1, k2, v1, v2 = merge(top, small, topv, smallv)
    key_store = key_store.at[n].set(k1)
    val_store = val_store.at[n].set(v1)
    key_store = key_store.at[s].set(k2)
    val_store = val_store.at[s].set(v2)    
    key_store, val_store = delete_heapify(key_store, val_store, size, max_size, s)
    return key_store, val_store


# TESTS

import hypothesis
from hypothesis import example, given, strategies as st


@given(
    st.lists(st.integers(min_value=-50, max_value=50), min_size=32, max_size=32)
)
def test_sort(ls):
    ls2 = sorted(ls)
    size = len(ls)
    group = 4
    heap = make_heap(group, size)
    for i in range(size // group):
        x = np.array(ls[i * group : (i + 1) * group])
        x = x.reshape(group)
        heap = insert(heap, x, x)
        
    ks, vs = [], []
    for j in range(size // group):
        heap, k, v = delete_min(heap)
        ks.append(k)
        vs.append(v)
    ls = []
    lsv = []
    for j in range(size // group):
        ls += ks[j].tolist()
        lsv += vs[j].tolist()

    for j in range(1, len(ls)):
        assert ls[j] >= ls[j - 1]
        assert ls2[j] == ls[j]
        assert ls[j] == lsv[j]


def test_head():
    
    
    heap, msize = make_heap(4, 16)
    x = np.array([3, 2,3, 1])
    heap = insert(*heap, msize, x, x)
    x = np.array([4, 2,4, 1])
    heap = insert(*heap, msize, x, x)
    x = np.array([5, 6,7, 8])
    heap = insert(*heap, msize, x, x)
    x = np.array([5, 6,7, 8])
    heap = insert(*heap, msize, x, x)
    heap, k, v = delete_min(heap, msize)
    assert (k == np.array([1, 1, 2, 2])).all()
    heap, k, v = delete_min(heap, msize)
    assert (k == np.array([3, 3, 4, 4])).all()


def test_merge():
    out_a, out_b, av, bv = merge(
        np.array([[1.0, 2, 5]]),
        np.array([[1.0, 4, 6]]),
        np.array([[1, 2, 5]]),
        np.array([[1, 4, 6]]),
    )

    assert out_a.tolist() == [[1, 1, 2]]
    assert out_b.tolist() == [[4, 5, 6]]
    assert av.tolist() == [[1, 1, 2]]
    assert bv.tolist() == [[4, 5, 6]]
    out_a, out_b, av, bv = merge(
        np.array([[1.0, 2, 5], [1.0, 4, 6]]),
        np.array([[1.0, 4, 6], [1.0, 2, 5]]),
        np.array([[1, 2, 5], [1, 4, 6]]),
        np.array([[1, 4, 6], [1, 2, 5]]),
    )
    assert out_a[1].tolist() == [1, 1, 2]
    assert out_b[1].tolist() == [4, 5, 6]
    assert av[1].tolist() == [1, 1, 2]
    assert bv[1].tolist() == [4, 5, 6]


def test_path():
    # assert make_path(0) == []
    assert make_path(1, 2).tolist() == [0, 1]
    assert make_path(2, 2).tolist() == [0, 2]
    assert make_path(3, 3).tolist() == [0, 1, 3]
    assert make_path(4, 3).tolist() == [0, 1, 4]
    assert make_path(5, 3).tolist() == [0, 2, 5]
