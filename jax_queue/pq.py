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
    outv = np.zeros((a.shape[-1] + b.shape[-1],), dtype=int)
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
    def path(c, a):
        x = a + 2 * c
        x = np.minimum(x, index+1)
        return x, x

    _, x = jax.lax.scan(path, np.array([1]), x[1:])
    return np.concatenate((np.array([0]), (x - 1).reshape(-1)))


def make_heap(group_size, total_size):
    """
    Create a heap over vectors of `group_size` that
    can expand to `group_size * total_size` nodes with
    independent `batch`es.
    """
    size = np.zeros(1, dtype=int)
    key_store = np.full((total_size, group_size), 1.e5)
    val_store = np.zeros((total_size, group_size), dtype=int)
    return (key_store, val_store, size)

INF = 1.e9

@partial(jax.jit, static_argnums=3)
def insert(key_store, val_store, size, max_size, keys, values):
    """
    Insert a batch of group_size keys `keys`  with corresponding
    integer `values`.
    """
    path = make_path(size, max_size)
    def insert_heapify(state, n):
        key_store, val_store, keys, values = state
        head, keys, hvalues, values = merge(
            key_store[n], keys, val_store[n], values
        )
        return (key_store.at[n].set(head), val_store.at[n].set(hvalues),
                keys, values), None

    (key_store, val_store, keys, values), _ = \
        jax.lax.scan(insert_heapify, (key_store, val_store, keys, values), path)
    return key_store, val_store, size + 1

@partial(jax.jit, static_argnums=1)
def delete_min(heap, msize):
    key_store, val_store, size = heap
    keys = key_store[0]
    values = val_store[0]
    def one():
        return key_store.at[0].set(INF), val_store.at[0].set(-1)
    def two():
        path = make_path(size - 1, msize)
        key_store2 = key_store.at[0].set(key_store[path[-1]]).at[path[-1]].set(INF)
        val_store2 = val_store.at[0].set(val_store[path[-1]]).at[path[-1]].set(-1)
        key_store3, val_store3, n = \
            jax.lax.fori_loop(0, msize, delete_heapify, (key_store2, val_store2, 0))
        return key_store3, val_store3
    key_store, val_store = jax.lax.cond((size == 1).all(), one, two)
    size = size - 1
    return (key_store, val_store, size), keys, values

def delete_heapify(_, state):
    key_store, val_store, n = state
    c = np.stack(((n + 1) * 2 - 1, (n + 1) * 2))
    c_l,c_r = key_store[c[0]], key_store[c[1]]
    c_lv, c_rv = val_store[c[0]], val_store[c[1]]
    ins = np.where(c_l[-1] < c_r[-1], 0, 1)
    s, l = c[ins], c[1 - ins]
    small, k2, smallv, v2 = merge(c_l, c_r, c_lv, c_rv)
    k1, k2, v1, v2 = merge(key_store[n], small, val_store[n], smallv)
    key_store = key_store.at[l].set(k2).at[n].set(k1).at[s].set(k2)
    val_store = val_store.at[l].set(v2).at[n].set(v1).at[s].set(v2)
    return key_store, val_store, s

# TESTS

import hypothesis
from hypothesis import example, given, strategies as st
from jax.config import config




@given(
    st.lists(st.integers(min_value=-50, max_value=50), min_size=32, max_size=32)
)
def test_sort(ls):
    config.update('jax_disable_jit', True)
    ls2 = sorted(ls)
    size = len(ls)
    group = 4
    heap = make_heap(group, size)
    msize = 8
    for i in range(size // group):
        x = np.array(ls[i * group : (i + 1) * group])
        x = x.reshape(group)
        x = np.sort(x)
        heap = insert(*heap, msize, x, x)

    ks, vs = [], []
    for j in range(size // group):
        heap, k, v = delete_min(heap, msize)
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
    msize = 10
    heap = make_heap(4, 2**4)
    x = np.array([1, 2, 3, 3])
    heap = insert(*heap, msize, x.astype(float), x)
    x = np.array([1, 2, 4, 4])
    heap = insert(*heap, msize, x.astype(float), x)
    x = np.array([5, 6, 7, 8])
    heap = insert(*heap, msize, x.astype(float), x)
    x = np.array([5, 6, 7, 8])
    heap = insert(*heap, msize, x.astype(float), x)
    heap, k, v = delete_min(heap, msize)
    assert (k == np.array([1, 1, 2, 2])).all()
    assert (k == v).all()
    heap, k, v = delete_min(heap, msize)
    assert (k == np.array([3, 3, 4, 4])).all()
    assert (k == v).all()
    heap, k, v = delete_min(heap, msize)
    print(k)
    assert (k == np.array([5, 5, 6, 6])).all()
    assert (k == v).all()
    heap, k, v = delete_min(heap, msize)
    assert (k == np.array([7, 7, 8, 8])).all()
    assert (k == v).all()

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
    assert make_path(1, 3).tolist() == [0, 1, 1]
    assert make_path(2, 2).tolist() == [0, 2]
    assert make_path(3, 3).tolist() == [0, 1, 3]
    assert make_path(4, 3).tolist() == [0, 1, 4]
    assert make_path(5, 3).tolist() == [0, 2, 5]
