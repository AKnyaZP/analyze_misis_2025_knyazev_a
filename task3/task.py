import json
import os
import re
import sys
from collections import defaultdict, deque
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


def _build_Y_strict(lvl: List[int]) -> List[List[int]]:
    n = len(lvl)
    y = [[0] * n for _ in range(n)]
    for i in range(n):
        li = lvl[i]
        if li < 0:
            continue
        row = y[i]
        for j in range(n):
            if i == j:
                continue
            lj = lvl[j]
            if lj < 0:
                continue
            if li < lj:
                row[j] = 1
    return y


def _transpose(m: List[List[int]]) -> List[List[int]]:
    n = len(m)
    return [[m[j][i] for j in range(n)] for i in range(n)]


def _and(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    n = len(a)
    return [[1 if (a[i][j] and b[i][j]) else 0 for j in range(n)] for i in range(n)]


def _or(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    n = len(a)
    return [[1 if (a[i][j] or b[i][j]) else 0 for j in range(n)] for i in range(n)]


def _warshall(m: List[List[int]]) -> List[List[int]]:
    n = len(m)
    w = [row[:] for row in m]
    for k in range(n):
        wk = w[k]
        for i in range(n):
            if not w[i][k]:
                continue
            wi = w[i]
            for j in range(n):
                if wk[j]:
                    wi[j] = 1
    return w


def _components_undirected(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    used = [False] * n
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
            row = adj[v]
            for u in range(n):
                if row[u] and not used[u]:
                    used[u] = True
                    q.append(u)
        comps.append(sorted(comp))
    return comps


def _cluster_order(YA: List[List[int]], YB: List[List[int]], clusters: List[List[int]]) -> List[List[int]]:
    m = len(clusters)
    g = [[0] * m for _ in range(m)]

    # Use "agreement" edges: cluster i precedes cluster j if exists a in i, b in j
    # such that both experts say a < b.
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            ok = False
            for a in clusters[i]:
                for b in clusters[j]:
                    if YA[a][b] and YB[a][b]:
                        ok = True
                        break
                if ok:
                    break
            g[i][j] = 1 if ok else 0

    indeg = [0] * m
    for i in range(m):
        for j in range(m):
            if g[i][j]:
                indeg[j] += 1

    q = deque([i for i in range(m) if indeg[i] == 0])
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for u in range(m):
            if g[v][u]:
                indeg[u] -= 1
                if indeg[u] == 0:
                    q.append(u)

    if len(order) != m:
        return clusters
    return [clusters[i] for i in order]


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


def main(a_json: str, b_json: str) -> str:
    A = _to_clusters(_loads_relaxed(a_json))
    B = _to_clusters(_loads_relaxed(b_json))

    items = _universe(A, B)

    lvlA = _levels(A, items)
    lvlB = _levels(B, items)

    YA = _build_Y_strict(lvlA)
    YB = _build_Y_strict(lvlB)

    n = len(items)

    # Conflicts: A says i<j while B says j<i (or vice versa).
    core_pairs: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            a_ij = YA[i][j]
            a_ji = YA[j][i]
            b_ij = YB[i][j]
            b_ji = YB[j][i]
            if (a_ij and b_ji) or (a_ji and b_ij):
                core_pairs.add((i, j))

    # Build equivalence: start with identity, then connect conflicting pairs.
    E = [[0] * n for _ in range(n)]
    for i in range(n):
        E[i][i] = 1
    for i, j in core_pairs:
        E[i][j] = 1
        E[j][i] = 1

    E_star = _warshall(E)
    und = [[1 if (E_star[i][j] or E_star[j][i]) else 0 for j in range(n)] for i in range(n)]
    clusters = _components_undirected(und)
    clusters = _cluster_order(YA, YB, clusters)

    return json.dumps(_encode(items, clusters), ensure_ascii=False)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "../data/task3")
    
    a_json_path = os.path.join(data_dir, "Ranzhirovka-A.json")
    b_json_path = os.path.join(data_dir, "Ranzhirovka-B.json")
    
    a_json_content = _read_text(a_json_path)
    b_json_content = _read_text(b_json_path)
    
    result = main(a_json_content, b_json_content)
    print(result)
