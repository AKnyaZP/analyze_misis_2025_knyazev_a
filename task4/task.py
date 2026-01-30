import json
from typing import List, Tuple, Dict, Union, Any


def load_json_data(data_input: Union[str, dict, list]) -> Union[dict, list]:
    """
    Загружает и парсит JSON данные из строки или возвращает готовую структуру.
    
    Args:
        data_input: JSON строка, словарь или список
        
    Returns:
        Распарсенная структура данных (dict или list)
    """
    if isinstance(data_input, (dict, list)):
        return data_input
    
    try:
        return json.loads(data_input.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON: {e}")


def normalize_term_name(term: str) -> str:
    """
    Нормализует название терма для унификации вариантов написания.
    
    Args:
        term: Название терма
        
    Returns:
        Нормализованное название
    """
    if term is None:
        return None
    
    normalized = str(term).strip().lower()
    
    # Словарь синонимов для нормализации
    synonyms = {
        "нормально": "комфортно",
        "комф": "комфортно",
        "слабо": "слабый",
        "слаб": "слабый",
        "умеренно": "умеренный",
        "умерен": "умеренный",
        "интенсивно": "интенсивный",
        "интенс": "интенсивный",
    }
    
    return synonyms.get(normalized, normalized)


def calculate_membership(x: float, points: List[Tuple[float, float]]) -> float:
    """
    Вычисляет степень принадлежности значения x к нечеткому множеству.
    Использует линейную интерполяцию между точками.
    
    Args:
        x: Входное значение
        points: Список точек [(x1, y1), (x2, y2), ...] определяющих функцию принадлежности
        
    Returns:
        Степень принадлежности в диапазоне [0, 1]
    """
    if not points:
        return 0.0
    
    # Сортируем точки по x-координате
    sorted_points = sorted([(float(px), float(py)) for px, py in points])
    
    # Граничные случаи
    first_x, first_y = sorted_points[0]
    last_x, last_y = sorted_points[-1]
    
    if x <= first_x:
        return max(0.0, min(1.0, first_y))
    if x >= last_x:
        return max(0.0, min(1.0, last_y))
    
    # Проверяем точное совпадение
    for point_x, point_y in sorted_points:
        if point_x == x:
            return max(0.0, min(1.0, point_y))
    
    # Линейная интерполяция между соседними точками
    for i in range(len(sorted_points) - 1):
        x1, y1 = sorted_points[i]
        x2, y2 = sorted_points[i + 1]
        
        if x1 <= x <= x2:
            if x2 == x1:
                return max(0.0, min(1.0, max(y1, y2)))
            
            # Линейная интерполяция
            ratio = (x - x1) / (x2 - x1)
            interpolated_y = y1 + ratio * (y2 - y1)
            return max(0.0, min(1.0, interpolated_y))
    
    return 0.0


def build_terms_dictionary(data: dict, variable_key: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Извлекает термы и их функции принадлежности из данных.
    
    Args:
        data: Словарь с данными переменной
        variable_key: Ключ переменной (например, "temperature" или "heating")
        
    Returns:
        Словарь {название_терма: список_точек}
    """
    terms = {}
    
    if variable_key in data and "terms" in data[variable_key]:
        for term_data in data[variable_key]["terms"]:
            term_id = normalize_term_name(term_data.get("id", ""))
            term_points = term_data.get("points", [])
            terms[term_id] = term_points
    
    return terms


def fuzzify_input(value: float, terms: Dict[str, List[Tuple[float, float]]]) -> Dict[str, float]:
    """
    Фаззификация: вычисляет степени принадлежности входного значения ко всем термам.
    
    Args:
        value: Входное значение
        terms: Словарь термов с их функциями принадлежности
        
    Returns:
        Словарь {название_терма: степень_принадлежности}
    """
    memberships = {}
    for term_name, term_points in terms.items():
        memberships[term_name] = calculate_membership(value, term_points)
    
    return memberships


def apply_fuzzy_rules(
    input_memberships: Dict[str, float],
    rules: List[Tuple[str, str]],
    output_terms: Dict[str, List[Tuple[float, float]]],
    grid_size: int = 10000
) -> List[float]:
    """
    Применяет нечеткие правила и агрегирует результат.
    
    Args:
        input_memberships: Степени принадлежности входа к термам
        rules: Список правил вида [(антецедент, консеквент), ...]
        output_terms: Термы выходной переменной
        grid_size: Размер сетки для дискретизации
        
    Returns:
        Агрегированная функция принадлежности (список значений)
    """
    # Определяем диапазон значений выходной переменной
    all_x_values = []
    for points in output_terms.values():
        for x, _ in points:
            all_x_values.append(float(x))
    
    if not all_x_values:
        return [0.0]
    
    min_x = min(all_x_values)
    max_x = max(all_x_values)
    
    if max_x <= min_x:
        return [0.0]
    
    step = (max_x - min_x) / grid_size
    aggregated = [0.0] * (grid_size + 1)
    
    # Применяем каждое правило
    for antecedent, consequent in rules:
        # Нормализуем названия термов
        norm_antecedent = normalize_term_name(antecedent)
        norm_consequent = normalize_term_name(consequent)
        
        # Степень активации правила (минимум из антецедентов)
        activation_level = input_memberships.get(norm_antecedent, 0.0)
        
        if activation_level <= 0.0:
            continue
        
        # Получаем функцию принадлежности консеквента
        consequent_points = output_terms.get(norm_consequent, [])
        
        # Применяем операцию min (метод Мамдани)
        for i in range(grid_size + 1):
            x = min_x + step * i
            consequent_membership = calculate_membership(x, consequent_points)
            rule_output = min(activation_level, consequent_membership)
            
            # Агрегация по максимуму
            aggregated[i] = max(aggregated[i], rule_output)
    
    return aggregated


def defuzzify_centroid(
    aggregated: List[float],
    min_x: float,
    max_x: float
) -> float:
    """
    Дефаззификация методом центра тяжести (центроида).
    
    Args:
        aggregated: Агрегированная функция принадлежности
        min_x: Минимальное значение диапазона
        max_x: Максимальное значение диапазона
        
    Returns:
        Четкое выходное значение
    """
    if not aggregated or max_x <= min_x:
        return min_x
    
    grid_size = len(aggregated) - 1
    step = (max_x - min_x) / grid_size if grid_size > 0 else 0
    
    numerator = 0.0
    denominator = 0.0
    
    for i, membership in enumerate(aggregated):
        x = min_x + step * i
        numerator += x * membership
        denominator += membership
    
    if denominator == 0:
        return min_x
    
    return numerator / denominator


def defuzzify_max(
    aggregated: List[float],
    min_x: float,
    max_x: float
) -> float:
    """
    Дефаззификация методом максимума (выбирается первое значение с максимальной принадлежностью).
    
    Args:
        aggregated: Агрегированная функция принадлежности
        min_x: Минимальное значение диапазона
        max_x: Максимальное значение диапазона
        
    Returns:
        Четкое выходное значение
    """
    if not aggregated:
        return min_x
    
    max_membership = max(aggregated)
    
    if max_membership <= 0:
        return min_x
    
    grid_size = len(aggregated) - 1
    step = (max_x - min_x) / grid_size if grid_size > 0 else 0
    
    # Находим первое значение с максимальной принадлежностью
    epsilon = 1e-12
    for i, membership in enumerate(aggregated):
        if membership >= max_membership - epsilon:
            return min_x + step * i
    
    return min_x


def main(
    temperature_data: Union[str, dict],
    heating_data: Union[str, dict],
    rules_data: Union[str, list],
    current_temperature: float
) -> float:
    """
    Главная функция нечеткого вывода для системы управления отоплением.
    
    Args:
        temperature_data: JSON с данными о температуре (входная переменная)
        heating_data: JSON с данными об уровне нагрева (выходная переменная)
        rules_data: JSON с правилами вида [["холодно", "интенсивно"], ...]
        current_temperature: Текущая температура (четкое входное значение)
        
    Returns:
        Рекомендуемый уровень нагрева (четкое выходное значение)
    """
    # 1. Загрузка данных
    temp_data = load_json_data(temperature_data)
    heat_data = load_json_data(heating_data)
    rules = load_json_data(rules_data)
    
    # 2. Извлечение термов
    temperature_terms = build_terms_dictionary(temp_data, "temperature")
    heating_terms = build_terms_dictionary(heat_data, "heating")
    
    # 3. Фаззификация входа
    temp_value = float(current_temperature)
    input_memberships = fuzzify_input(temp_value, temperature_terms)
    
    # 4. Применение правил и агрегация
    aggregated_output = apply_fuzzy_rules(
        input_memberships,
        rules,
        heating_terms,
        grid_size=10000
    )
    
    # 5. Определение диапазона выходной переменной
    all_heating_x = []
    for points in heating_terms.values():
        for x, _ in points:
            all_heating_x.append(float(x))
    
    min_heating = min(all_heating_x) if all_heating_x else 0.0
    max_heating = max(all_heating_x) if all_heating_x else 0.0
    
    # 6. Дефаззификация (метод максимума)
    result = defuzzify_max(aggregated_output, min_heating, max_heating)
    
    return float(result)


if __name__ == "__main__":
    import os
    
    # Пути к файлам данных
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/task4/data.json")
    rules_path = os.path.join(script_dir, "../data/task4/rules.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    
    with open(rules_path, "r", encoding="utf-8") as f:
        fuzzy_rules = json.load(f)
    
    input_temperature = int(input("Введите температуру в °C: "))
    
    optimal_s = main(
        temperature_data=full_data,
        heating_data=full_data,
        rules_data=fuzzy_rules,
        current_temperature=input_temperature
    )
    
    print(f"Для температуры {input_temperature}°C оптимальное управление: {optimal_s:.2f}")
