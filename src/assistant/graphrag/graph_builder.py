"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)

    for result in results:
        for entity in result.entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
            sources = set(graph.nodes[entity.node_id]["sources"])
            sources.add(result.chunk_id)
            graph.nodes[entity.node_id]["sources"] = sorted(sources)

        for relation in result.relations:
            if not relation.matches_schema():
                continue
            for entity in (relation.subject, relation.object):
                if entity.node_id not in graph:
                    graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
                sources = set(graph.nodes[entity.node_id]["sources"])
                sources.add(result.chunk_id)
                graph.nodes[entity.node_id]["sources"] = sorted(sources)

            source, target = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, weight=0, sources=[])
            edge = graph.edges[source, target]
            edge_sources = set(edge["sources"])
            edge_sources.add(result.chunk_id)
            edge["sources"] = sorted(edge_sources)
            edge["weight"] = len(edge_sources)

    seen: set[frozenset[str]] = set()
    for source, target, data in list(graph.edges(data=True)):
        pair = frozenset((source, target))
        if data["type"] != "REPLACED_BY" or pair in seen or not graph.has_edge(target, source):
            continue
        reverse = graph.edges[target, source]
        if reverse["type"] != "REPLACED_BY":
            continue
        seen.add(pair)
        graph.graph["conflicts"] += 1
        if data["weight"] > reverse["weight"]:
            graph.remove_edge(target, source)
        elif data["weight"] < reverse["weight"]:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)

    return graph
