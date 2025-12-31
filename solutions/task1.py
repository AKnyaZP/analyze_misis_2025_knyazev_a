import argparse
import os
import sys
from collections import defaultdict, deque
from typing import List, Tuple


def main(s: str, e: str) -> Tuple[
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
]:
    root = (e or "").strip()
    if not root:
        raise ValueError("Empty root id")

    text = (s or "").strip()
    edges = []
    if text:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Bad edge line: {line!r}")
            edges.append((parts[0], parts[1]))

    children = defaultdict(list)
    parent = {}

    for u, v in edges:
        children[u].append(v)
        if v in parent and parent[v] != u:
            raise ValueError(f"Node {v!r} has multiple parents")
        parent[v] = u

    reachable = {root}
    q = deque([root])
    while q:
        x = q.popleft()
        for y in children.get(x, []):
            if y not in reachable:
                reachable.add(y)
                q.append(y)

    children2 = {}
    parent2 = {}
    for u in reachable:
        kids = [v for v in children.get(u, []) if v in reachable]
        if kids:
            kids.sort()
            children2[u] = kids
            for v in kids:
                parent2[v] = u

    nodes = []
    q = deque([root])
    seen = {root}
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

    def mat():
        return [[False] * n for _ in range(n)]

    r1, r2, r3, r4, r5 = mat(), mat(), mat(), mat(), mat()

    for u, kids in children2.items():
        iu = pos[u]
        for v in kids:
            iv = pos[v]
            r1[iu][iv] = True
            r2[iv][iu] = True

    for v in nodes:
        iv = pos[v]
        cur = v
        dist = 0
        while cur in parent2:
            cur = parent2[cur]
            dist += 1
            if dist >= 2:
                ia = pos[cur]
                r3[ia][iv] = True
                r4[iv][ia] = True

    by_parent = defaultdict(list)
    for v in nodes:
        if v in parent2:
            by_parent[parent2[v]].append(v)

    for kids in by_parent.values():
        kids.sort()
        for i in range(len(kids)):
            ai = pos[kids[i]]
            for j in range(len(kids)):
                if i != j:
                    r5[ai][pos[kids[j]]] = True

    return r1, r2, r3, r4, r5


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", dest="csv_path", default=None)
    p.add_argument("--root", dest="root", default=None)
    p.add_argument("--print-nodes", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(script_dir, "../data/task1/test.csv")
    root = args.root or "1"

    csv_content = _read_text(csv_path)
    out = main(csv_content, root)

    if args.print_nodes:
        print("OK")

    print(out)