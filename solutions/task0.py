def main(csv_graph: str) -> list[list[int]]:
    """
    Конвертирует CSV строку с описанием рёбер ориентированного ациклического графа в матрицу смежности.
    
    Args:
        csv_graph: строка в формате CSV, где каждая строка содержит ребро в формате "начальная_вершина,конечная_вершина"
    
    Returns:
        Матрица смежности в виде list[list[int]], где matrix[i][j] = 1 означает наличие ребра из вершины i в вершину j
    """
    # Обрабатываем пустой ввод
    if not csv_graph or not csv_graph.strip():
        return []
    
    # Парсим строки и извлекаем рёбра
    lines = csv_graph.strip().split('\n')
    edges = []
    vertices = set()
    
    for line in lines:
        line = line.strip()
        if line:  # Пропускаем пустые строки
            try:
                parts = line.split()
                if len(parts) != 2:
                    continue  # Пропускаем некорректные строки
                
                start, end = int(parts[0].strip()), int(parts[1].strip())
                edges.append((start, end))
                vertices.add(start)
                vertices.add(end)
            except ValueError:
                continue  # Пропускаем строки с некорректными данными
    
    # Если нет валидных рёбер
    if not vertices:
        return []
    
    # Определяем количество вершин и создаём отображение вершин к индексам
    # Сортируем вершины для стабильного порядка
    sorted_vertices = sorted(vertices)
    n = len(sorted_vertices)
    vertex_to_index = {vertex: i for i, vertex in enumerate(sorted_vertices)}
    
    # Инициализируем матрицу смежности нулями
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    # Заполняем матрицу смежности
    for start, end in edges:
        i = vertex_to_index[start]
        j = vertex_to_index[end]
        matrix[i][j] = 1
    
    return matrix


# Чтение CSV файла
with open('data/task0/graph.csv', 'r') as file:
    csv_content = file.read()
    print(csv_content)

# Получение матрицы смежности
result = main(csv_content)
print(result)