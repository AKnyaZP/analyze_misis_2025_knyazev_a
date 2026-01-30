import json
import numpy as np
import os
import re
from collections import deque
from typing import Any, Dict, List, Set, Tuple

_TRAILING_COMMAS = re.compile(r",\s*([\]\}])")


def _loads_relaxed(s: str) -> Any:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s = _TRAILING_COMMAS.sub(r"\1", s)
        return json.loads(s)


def _to_clusters(x: Any) -> List[List[str]]:
    if not isinstance(x, list):
        raise ValueError("Ranking must be a list")
    out: List[List[str]] = []
    for item in x:
        if isinstance(item, list):
            cl = [str(v) for v in item]
        else:
            cl = [str(item)]
        if cl:
            out.append(cl)
    return out


def _universe(a: List[List[str]], b: List[List[str]]) -> List[str]:
    s: Set[str] = set()
    for cl in a:
        s.update(cl)
    for cl in b:
        s.update(cl)

    def key(z: str):
        return (0, int(z)) if z.isdigit() else (1, z)

    return sorted(s, key=key)


def _levels(clusters: List[List[str]], items: List[str]) -> List[int]:
    n = len(items)
    pos: Dict[str, int] = {items[i]: i for i in range(n)}
    lvl = [-1] * n
    for r, cl in enumerate(clusters):
        for v in cl:
            lvl[pos[v]] = r
    return lvl


def _build_Y_strict(lvl: List[int]) -> np.ndarray:
    """Build Y matrix using numpy for optimization.
    Matrix where y[i,j] = 1 if pos[i] >= pos[j] (like build_matrix in new code).
    Note: smaller level means earlier position, so lvl[i] <= lvl[j] means pos[i] >= pos[j]."""
    n = len(lvl)
    lvl_arr = np.array(lvl)
    valid_mask = lvl_arr >= 0
    
    i_indices = np.arange(n)[:, np.newaxis]
    j_indices = np.arange(n)[np.newaxis, :]
    
    condition = (lvl_arr[i_indices] <= lvl_arr[j_indices]) & valid_mask[i_indices] & valid_mask[j_indices]
    y = condition.astype(int)
    
    return y


def _transpose(m: np.ndarray) -> np.ndarray:
    """Transpose matrix using numpy."""
    return m.T


def _and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise AND using numpy."""
    return (a & b).astype(int)


def _or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise OR using numpy."""
    return (a | b).astype(int)


def _warshall(m: np.ndarray) -> np.ndarray:
    """Floyd-Warshall algorithm for transitive closure using numpy."""
    closure = m.copy()
    n = len(closure)
    
    for k in range(n):
        closure = closure | (closure[:, k:k+1] & closure[k:k+1, :])
    
    return closure


def _components_undirected(adj: np.ndarray) -> List[List[int]]:
    """Find connected components using numpy-optimized BFS."""
    n = len(adj)
    used = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []

    for s in range(n):
        if used[s]:
            continue
        q = deque([s])
        used[s] = True
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            neighbors = np.where(adj[v, :] & ~used)[0]
            for u in neighbors:
                used[u] = True
                q.append(u)
        comps.append(sorted(comp))
    return comps


def _cluster_order(YA: np.ndarray, YB: np.ndarray, clusters: List[List[int]]) -> List[List[int]]:
    """Order clusters using topological sort (recursive DFS)."""
    m = len(clusters)
    g = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            ok = False
            for a in clusters[i]:
                for b in clusters[j]:
                    if YA[a, b] and YB[a, b]:
                        ok = True
                        break
                if ok:
                    break
            g[i, j] = 1 if ok else 0

    visited = np.zeros(m, dtype=bool)
    result_order = []
    
    def topological_sort(v):
        visited[v] = True
        for u in range(m):
            if g[v, u] == 1 and not visited[u]:
                topological_sort(u)
        result_order.append(v)
    
    for i in range(m):
        if not visited[i]:
            topological_sort(i)
    
    result_order.reverse()
    return [clusters[i] for i in result_order]


def _encode(items: List[str], clusters: List[List[int]]) -> List[Any]:
    def cast(v: str):
        return int(v) if v.isdigit() else v

    out: List[Any] = []
    for cl in clusters:
        vals = [cast(items[i]) for i in cl]
        if len(vals) == 1:
            out.append(vals[0])
        else:
            out.append(vals)
    return out


def main(a_json: str, b_json: str) -> Dict[str, Any]:
    """Main function that returns kernel and consistent_ranking like the new code."""
    A = _to_clusters(_loads_relaxed(a_json))
    B = _to_clusters(_loads_relaxed(b_json))

    items = _universe(A, B)

    lvlA = _levels(A, items)
    lvlB = _levels(B, items)

    YA = _build_Y_strict(lvlA)
    YB = _build_Y_strict(lvlB)

    n = len(items)

    YAB = YA * YB
    YA_T = _transpose(YA)
    YB_T = _transpose(YB)
    YAB_prime = YA_T * YB_T

    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
    kernel_mask = (YAB == 0) & (YAB_prime == 0) & upper_triangle
    kernel_pairs = np.argwhere(kernel_mask)
    kernel = [[int(i + 1), int(j + 1)] for i, j in kernel_pairs]

    C = YA * YB

    if kernel:
        kernel_indices = np.array(kernel) - 1
        C[kernel_indices[:, 0], kernel_indices[:, 1]] = 1
        C[kernel_indices[:, 1], kernel_indices[:, 0]] = 1

    E = C * C.T

    E_star = _warshall(E)

    und = _or(E_star, _transpose(E_star))
    
    clusters = _components_undirected(und)
    
    if not clusters:
        return {"kernel": kernel, "consistent_ranking": []}

    clusters = _cluster_order(YA, YB, clusters)

    item_to_value = {i: int(items[i]) if items[i].isdigit() else items[i] for i in range(len(items))}
    
    consistent_ranking = []
    for cluster in clusters:
        cluster_values = [item_to_value[i] for i in cluster]
        if len(cluster_values) == 1:
            consistent_ranking.append(cluster_values[0])
        else:
            consistent_ranking.append(cluster_values)

    return {
        "kernel": kernel,
        "consistent_ranking": consistent_ranking
    }


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data/task3")
    
    json_a = _read_text(os.path.join(data_dir, "Ranzhirovka-A.json"))
    json_b = _read_text(os.path.join(data_dir, "Ranzhirovka-B.json"))
    json_c = _read_text(os.path.join(data_dir, "Ranzhirovka-С.json"))
    
    print("СРАВНЕНИЕ РАНЖИРОВОК")
    
    print("\nrange_a.json vs range_b.json")
    result_ab = main(json_a, json_b)
    print(f"Ядро противоречий: {result_ab['kernel']}")
    print(f"Согласованная ранжировка: {result_ab['consistent_ranking']}")
    
    print("\nrange_a.json vs range_c.json")
    result_ac = main(json_a, json_c)
    print(f"Ядро противоречий: {result_ac['kernel']}")
    print(f"Согласованная ранжировка: {result_ac['consistent_ranking']}")
    
    print("\nrange_b.json vs range_c.json")
    result_bc = main(json_b, json_c)
    print(f"Ядро противоречий: {result_bc['kernel']}")
    print(f"Согласованная ранжировка: {result_bc['consistent_ranking']}")
