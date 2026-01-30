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


# --- Шаг 1: Построение матриц отношений Y (y_ij = 1 <=> объект i не строго позже j в ранжировке) ---
def _build_Y(lvl: List[int]) -> np.ndarray:
    """Матрица отношений: Y[i,j] = 1 если уровень i <= уровень j (т.е. i не позже j)."""
    n = len(lvl)
    lvl_arr = np.array(lvl)
    valid = lvl_arr >= 0
    i_ = np.arange(n)[:, np.newaxis]
    j_ = np.arange(n)[np.newaxis, :]
    y = ((lvl_arr[i_] <= lvl_arr[j_]) & valid[i_] & valid[j_]).astype(np.int64)
    return y


def _compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Композиция отношений: (A ◦ B)[i,j] = 1 если существует k: A[i,k]=1 и B[k,j]=1."""
    return (A @ B > 0).astype(np.int64)


def _or_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Побитовое ИЛИ для 0/1 матриц."""
    return (A | B).astype(np.int64)


# --- Алгоритм Уоршелла для транзитивного замыкания ---
def _warshall(E: np.ndarray) -> np.ndarray:
    """Транзитивное замыкание E* алгоритмом Уоршелла."""
    closure = E.copy()
    n = closure.shape[0]
    for k in range(n):
        closure = closure | (closure[:, k : k + 1] @ closure[k : k + 1, :] > 0).astype(
            np.int64
        )
    return closure


def _connected_components(adj: np.ndarray) -> List[List[int]]:
    """Компоненты связности неориентированного графа (adj симметрична)."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []

    for s in range(n):
        if visited[s]:
            continue
        q = deque([s])
        visited[s] = True
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            for u in range(n):
                if adj[v, u] and not visited[u]:
                    visited[u] = True
                    q.append(u)
        comps.append(sorted(comp))
    return comps


def _order_clusters_by_C(
    C: np.ndarray, clusters: List[List[int]]
) -> List[List[int]]:
    """Упорядочивание кластеров по матрице согласованного порядка C (топологическая сортировка)."""
    m = len(clusters)
    # Граф порядка: кластер i < кластер j, если для каких-то a из i, b из j: C[a,b]=1 и не C[b,a]=1 или просто C[a,b]=1 и кластеры различаются
    # Отношение: кластер i предшествует кластеру j, если для любых/некоторых представителей a из i, b из j выполняется C[a,b]=1 (согласованный порядок).
    order_adj = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            # Есть ли дуга i -> j: существует a in clusters[i], b in clusters[j] с C[a,b]=1
            for a in clusters[i]:
                for b in clusters[j]:
                    if C[a, b] == 1:
                        order_adj[i, j] = 1
                        break
                if order_adj[i, j]:
                    break

    visited = np.zeros(m, dtype=bool)
    result_order: List[int] = []

    def dfs(v: int) -> None:
        visited[v] = True
        for u in range(m):
            if order_adj[v, u] and not visited[u]:
                dfs(u)
        result_order.append(v)

    for i in range(m):
        if not visited[i]:
            dfs(i)

    result_order.reverse()
    return [clusters[i] for i in result_order]


def main(a_json: str, b_json: str) -> Dict[str, Any]:
    """
    Построение согласованной кластерной ранжировки f(A, B) по алгоритму.
    Возвращает: kernel (ядро противоречий S(A,B)), consistent_ranking (f(A,B)).
    """
    A = _to_clusters(_loads_relaxed(a_json))
    B = _to_clusters(_loads_relaxed(b_json))
    items = _universe(A, B)
    n = len(items)

    lvlA = _levels(A, items)
    lvlB = _levels(B, items)

    # --- Шаг 1: Матрицы YA, YB и транспонированные ---
    YA = _build_Y(lvlA)
    YB = _build_Y(lvlB)
    YA_T = YA.T
    YB_T = YB.T

    # --- Шаг 2: Выявление противоречий ---
    # Противоречие: эксперты расходятся — один считает i≼j, другой j≼i.
    # YAB[i,j]=YA*YB (поэлементно): 1 если оба считают i≼j. YAB'[i,j]=Y'A*Y'B: 1 если оба j≼i.
    # Пара (i,j) в ядре противоречий, если ни оба i≼j, ни оба j≼i — т.е. (YAB[i,j]==0 и YAB'[i,j]==0).
    YAB = (YA * YB).astype(np.int64)
    YAB_prime = (YA_T * YB_T).astype(np.int64)
    # Матрица противоречий: P[i,j]=1 для пар, где есть противоречие (i≠j и оба произведения 0)
    P = ((YAB == 0) & (YAB_prime == 0) & (np.arange(n)[:, None] != np.arange(n)[None, :])).astype(
        np.int64
    )

    # Ядро противоречий S(A,B) — пары объектов с p_ij = 1 (противоречивая пара), выводим как [i+1, j+1]
    kernel: List[List[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if P[i, j] == 1:
                kernel.append([i + 1, j + 1])

    # --- Шаг 3: Согласованный порядок C = YA ◦ YB (согласованное отношение: оба говорят i≼j);
    # в реализации: C = (YA * YB) поэлементно — оба эксперта согласны, что i не позже j.
    # Для (i,j) ∈ S(A,B): c_ij = c_ji = 1 (объединяем противоречивые пары в один кластер). ---
    C = (YA * YB).astype(np.int64)
    for (i1, j1) in kernel:
        i0, j0 = i1 - 1, j1 - 1
        C[i0, j0] = 1
        C[j0, i0] = 1

    # --- Шаг 4: Кластеры ---
    # Матрица эквивалентности: E = C ◦ C^T — симметричная часть (i~j если C[i,j]=1 и C[j,i]=1)
    E = (C * C.T).astype(np.int64)
    E_star = _warshall(E)
    # Неориентированный граф для компонент связности: E* симметрична по смыслу (эквивалентность)
    sym = E_star | E_star.T
    clusters_indices = _connected_components(sym)

    # --- Шаг 5–6: Упорядочивание кластеров и формирование f(A, B) ---
    ordered_clusters = _order_clusters_by_C(C, clusters_indices)

    def to_value(idx: int):
        s = items[idx]
        return int(s) if s.isdigit() else s

    consistent_ranking: List[Any] = []
    for cl in ordered_clusters:
        vals = [to_value(i) for i in cl]
        if len(vals) == 1:
            consistent_ranking.append(vals[0])
        else:
            consistent_ranking.append(vals)

    return {
        "kernel": kernel,
        "consistent_ranking": consistent_ranking,
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

    print("СРАВНЕНИЕ РАНЖИРОВОК (алгоритм согласованной кластерной ранжировки)\n")

    print("Ranzhirovka-A.json vs Ranzhirovka-B.json")
    result_ab = main(json_a, json_b)
    print(f"Ядро противоречий S(A,B): {result_ab['kernel']}")
    print(f"Согласованная кластерная ранжировка f(A,B): {result_ab['consistent_ranking']}")

    print("\nRanzhirovka-A.json vs Ranzhirovka-С.json")
    result_ac = main(json_a, json_c)
    print(f"Ядро противоречий S(A,C): {result_ac['kernel']}")
    print(f"Согласованная кластерная ранжировка f(A,C): {result_ac['consistent_ranking']}")

    print("\nRanzhirovka-B.json vs Ranzhirovka-С.json")
    result_bc = main(json_b, json_c)
    print(f"Ядро противоречий S(B,C): {result_bc['kernel']}")
    print(f"Согласованная кластерная ранжировка f(B,C): {result_bc['consistent_ranking']}")
