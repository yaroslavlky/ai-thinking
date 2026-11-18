"""Сообщества графа и их сводки. Требование FR-3."""

from collections import defaultdict

import networkx as nx

from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected.edges[source, target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    groups = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: tuple(nodes))
    return [Community(number, nodes) for number, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    members = set(community.nodes)
    entities = [f"{graph.nodes[node].get('type')}: {graph.nodes[node].get('name')}" for node in community.nodes]
    facts: list[str] = []
    relation_totals: defaultdict[str, int] = defaultdict(int)
    failure_totals: defaultdict[str, int] = defaultdict(int)
    for source, target, data in graph.edges(data=True):
        if source not in members or target not in members:
            continue
        relation = data.get("type", "")
        weight = int(data.get("weight", 1))
        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        facts.append(f"{source_name} -{relation}-> {target_name} (подтверждений: {weight})")
        relation_totals[relation] += weight
        if relation == "FAILED":
            failure_totals[target_name] += weight

    counts = [f"{kind}: {count}" for kind, count in sorted(relation_totals.items())]
    failures = [
        f"{name}: {count}"
        for name, count in sorted(failure_totals.items(), key=lambda item: (-item[1], item[0]))
    ]
    prompt = "\n".join(
        [
            "Напиши по-русски фактическую сводку сообщества графа знаний объёмом не более 150 слов.",
            "Не придумывай числа: используй только уже посчитанные значения ниже.",
            "Сущности:",
            *entities,
            "Связи:",
            *(facts or ["нет внутренних связей"]),
            "Суммарное число подтверждений по типам связей:",
            *(counts or ["нет"]),
            "Число отказов по узлам (веса FAILED):",
            *(failures or ["нет"]),
        ]
    )
    response = llm.chat([{"role": "user", "content": prompt}], max_tokens=300).content.strip()
    return " ".join(response.split()[:150])
