import math
import os
from collections import defaultdict, deque
from typing import List, Tuple


def _parse_graph(s: str, root_id: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Парсит CSV-строку с рёбрами (parent,child), возвращает список узлов (root первым) и рёбра по индексам."""
    root_id = (root_id or "").strip()
    if not root_id:
        raise ValueError("Empty root id")

    text = (s or "").strip()
    edges_raw: List[Tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Bad edge line: {line!r}")
        edges_raw.append((parts[0], parts[1]))

    children = defaultdict(list)
    parent: dict = {}
    for u, v in edges_raw:
        children[u].append(v)
        if v in parent and parent[v] != u:
            raise ValueError(f"Node {v!r} has multiple parents")
        parent[v] = u

    reachable = {root_id}
    q = deque([root_id])
    while q:
        x = q.popleft()
        for y in children.get(x, []):
            if y not in reachable:
                reachable.add(y)
                q.append(y)

    children2: dict = {}
    parent2: dict = {}
    for u in reachable:
        kids = sorted(v for v in children.get(u, []) if v in reachable)
        if kids:
            children2[u] = kids
            for v in kids:
                parent2[v] = u

    nodes: List[str] = []
    q = deque([root_id])
    seen = {root_id}
    while q:
        x = q.popleft()
        nodes.append(x)
        for y in children2.get(x, []):
            if y not in seen:
                seen.add(y)
                q.append(y)
    for v in sorted(reachable):
        if v not in seen:
            nodes.append(v)

    n = len(nodes)
    pos = {v: i for i, v in enumerate(nodes)}
    edges_idx = [(pos[u], pos[v]) for u, v in edges_raw if u in pos and v in pos]
    return nodes, edges_idx


def _build_relation_matrices(
    n: int, edges_idx: List[Tuple[int, int]]
) -> Tuple[List[List[int]], ...]:
    """Строит матрицы отношений r1..r5 (n×n, 0/1)."""
    r1 = [[0] * n for _ in range(n)]
    for i, j in edges_idx:
        r1[i][j] = 1

    r2 = [[r1[j][i] for j in range(n)] for i in range(n)]

    # r3: опосредованное управление — пути длины >= 2 (транзитивное замыкание минус r1)
    closure = [row[:] for row in r1]
    for _ in range(1, n):
        new = [[0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                if closure[i][k]:
                    for j in range(n):
                        if r1[k][j]:
                            new[i][j] = 1
        for i in range(n):
            for j in range(n):
                closure[i][j] = min(1, closure[i][j] + new[i][j])
    r3 = [[1 if closure[i][j] and not r1[i][j] else 0 for j in range(n)] for i in range(n)]

    r4 = [[r3[j][i] for j in range(n)] for i in range(n)]

    # r5: сотрудничество на одном уровне — общий родитель (r2[i][j]=1 ⇒ j родитель i)
    r5 = [[0] * n for _ in range(n)]
    by_parent: dict = defaultdict(list)
    for j in range(n):
        for i in range(n):
            if r2[i][j]:
                by_parent[j].append(i)
    for _parent, siblings in by_parent.items():
        for a in siblings:
            for b in siblings:
                if a != b:
                    r5[a][b] = 1

    return r1, r2, r3, r4, r5


def _entropy_and_normalized(
    matrices: List[List[List[int]]], n: int, k: int
) -> Tuple[float, float]:
    """
    H(M,R) = sum over all nodes j and relation types i of H(m_j, r_i),
    где l_ij — число исходящих связей узла j в отношении i,
    P = l_ij / (n-1), H(m_j, r_i) = -P*log2(P), 0*log2(0)=0.
    H_ref = c * n * k, c = 1/(e*ln(2)).
    h = H / H_ref.
    """
    if n <= 1:
        return 0.0, 0.0
    denom = n - 1
    H = 0.0
    for mat in matrices:
        for j in range(n):
            l_j = sum(mat[j][i] for i in range(n) if i != j)
            if l_j == 0:
                continue
            p = l_j / denom
            if p > 0:
                H += -p * math.log2(p)
    c = 1.0 / (math.e * math.log(2))
    H_ref = c * n * k
    h_val = H / H_ref if H_ref > 0 else 0.0
    return H, h_val


def task(s: str, e: str) -> Tuple[float, float]:
    """
    Принимает CSV-строку с рёбрами графа (parent,child) и идентификатор корневого узла.
    Возвращает (энтропия структуры H(M,R), нормированная структурная сложность h).
    Округление до одного знака после запятой.
    """
    nodes, edges_idx = _parse_graph(s, e)
    n = len(nodes)
    k = 5
    r1, r2, r3, r4, r5 = _build_relation_matrices(n, edges_idx)
    matrices = [r1, r2, r3, r4, r5]
    H, h_val = _entropy_and_normalized(matrices, n, k)
    return (round(H, 1), round(h_val, 1))


# Сигнатура из задания: main(s: str, e: str) -> Tuple[float, float]
def main(s: str, e: str) -> Tuple[float, float]:
    return task(s, e)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "../data/task2/test.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        input_data = f.read()
    root = input("Введите значение корневой вершины: ").strip() or "1"
    H, h = main(input_data, root)
    print(f"H(M,R) = {H}")
    print(f"h(M,R) = {h}")
