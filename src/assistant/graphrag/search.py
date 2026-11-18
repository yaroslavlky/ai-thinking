"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
import re
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

_NOT_FOUND = "В базе знаний не найдено."
CommunitySources = list[str]


def _document_sources(chunk_sources) -> list[str]:
    return sorted({str(source).split("#", 1)[0] for source in chunk_sources})


def _edge_fact(graph: nx.DiGraph, source: str, target: str) -> tuple[str, list[str]]:
    data = graph.edges[source, target]
    left = graph.nodes[source].get("name", source)
    right = graph.nodes[target].get("name", target)
    return f"{left} -{data['type']}-> {right}", data.get("sources", [])


def _directed_edge(graph: nx.DiGraph, left: str, right: str) -> tuple[str, str] | None:
    if graph.has_edge(left, right):
        return left, right
    if graph.has_edge(right, left):
        return right, left
    return None


def _replacement_edges(graph: nx.DiGraph, starts: set[str]) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    queue = [node for node in starts if graph.nodes[node].get("type") == "Part"]
    visited = set(queue)
    while queue:
        node = queue.pop(0)
        for target in graph.successors(node):
            if graph.edges[node, target].get("type") != "REPLACED_BY":
                continue
            selected.add((node, target))
            if target not in visited:
                visited.add(target)
                queue.append(target)
    return selected


def _supplier_edges(graph: nx.DiGraph, parts: set[str]) -> set[tuple[str, str]]:
    return {
        (part, target)
        for part in parts
        if graph.nodes[part].get("type") == "Part"
        for target in graph.successors(part)
        if graph.edges[part, target].get("type") == "SUPPLIED_BY"
    }


def _context_from_edges(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> tuple[list[str], list[str]]:
    rows: list[tuple[str, list[str]]] = [_edge_fact(graph, *edge) for edge in edges]
    rows.sort(key=lambda row: row[0])
    facts: list[str] = []
    sources: set[str] = set()
    used = 0
    for number, (fact, chunks) in enumerate(rows, start=1):
        line = f"{number}. {fact}"
        addition = len(line) + (1 if facts else 0)
        if used + addition > 8000:
            break
        facts.append(fact)
        sources.update(chunks)
        used += addition
    return facts, _document_sources(sources)


def _remove_ungrounded_code_sentences(text: str, allowed_codes: set[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    grounded = [sentence for sentence in sentences if set(find_codes(sentence)) <= allowed_codes]
    return " ".join(sentence.strip() for sentence in grounded if sentence.strip())


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    codes = find_codes(question)
    code_nodes = [f"{code_type(code)}:{code}" for code in codes]
    if any(node not in graph for node in code_nodes):
        return Answer(_NOT_FOUND, refused=True)

    name_nodes = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    anchors = list(dict.fromkeys([*code_nodes, *(node for node in name_nodes if node in graph)]))
    if not anchors:
        return Answer(_NOT_FOUND, refused=True)

    undirected = graph.to_undirected(as_view=True)
    edges: set[tuple[str, str]] = set()
    path_nodes: set[str] = set()
    if len(anchors) >= 2:
        for left, right in combinations(anchors, 2):
            try:
                paths = nx.all_simple_paths(undirected, left, right, cutoff=3)
                for path in paths:
                    path_nodes.update(path)
                    for a, b in zip(path, path[1:], strict=False):
                        edge = _directed_edge(graph, a, b)
                        if edge:
                            edges.add(edge)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    else:
        lengths = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        path_nodes.update(lengths)
        for source, target in graph.edges:
            if source in lengths and target in lengths:
                edges.add((source, target))

    replacements = _replacement_edges(graph, path_nodes)
    edges.update(replacements)
    replacement_parts = {node for edge in replacements for node in edge}
    edges.update(_supplier_edges(graph, replacement_parts))
    facts, sources = _context_from_edges(graph, edges)
    if not facts:
        return Answer(_NOT_FOUND, refused=True)

    numbered = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    prompt = (
        "Ответь на вопрос только по пронумерованным фактам графа. Не добавляй коды или числа, "
        "которых нет в фактах или вопросе. Называй коды деталей, а не только названия узлов. "
        "Если спрашивают, какая деталь стоит в модели, соедини HAS_PART и IS_A. Если в фактах есть цепочка "
        "REPLACED_BY, актуальной деталью всегда считай самый последний код всей цепочки и используй его "
        "в ответе, если пользователь прямо не спрашивает о старой детали. В A -REPLACED_BY-> B новой "
        "деталью является B; в A -REPLACED_BY-> B -REPLACED_BY-> C актуальной является только C. В связи "
        "E-18 -FIXED_BY-> SP-108 регламентом является объект справа, SP-108, а не код ошибки E-18. "
        "Не ссылайся на номера фактов: назови сам код регламента, детали или поставщика. "
        "Не повторяй количество заказа, цену или остаток: этих данных нет в графе, их проверяет ERP.\n\n"
        f"Вопрос: {question}\n\nФакты:\n{numbered}"
    )
    text = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    terminals = {target for _, target in replacements} - {source for source, _ in replacements}
    asks_for_part = any(word in question.lower() for word in ("детал", "заказать", "ставится", "поставляет"))
    if asks_for_part and len(terminals) == 1:
        terminal = next(iter(terminals))
        terminal_code = graph.nodes[terminal].get("name", terminal)
        if terminal_code not in find_codes(text):
            text = f"{text} Актуальная деталь: {terminal_code}."
    allowed_codes = set(find_codes(question)) | {code for fact in facts for code in find_codes(fact)}
    text = _remove_ungrounded_code_sentences(text, allowed_codes)
    if not text:
        return Answer(_NOT_FOUND, sources, facts, refused=True)
    return Answer(text, sources, facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    useful: list[tuple[str, CommunitySources]] = []
    for community in kb.communities:
        prompt = (
            "Ответь на вопрос только по сводке сообщества. Верни JSON с полями answer и score от 0 до 10. "
            "score=0, если сводка не помогает ответить.\n\n"
            f"Вопрос: {question}\n\nСводка:\n{community.summary}"
        )
        raw = llm.chat([{"role": "user", "content": prompt}], json_schema=MAP_SCHEMA).content
        try:
            partial = json.loads(raw)
            answer = partial["answer"]
            score = partial["score"]
            if not isinstance(answer, str) or not isinstance(score, int):
                continue
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if score > 0:
            useful.append((answer, _community_sources(community.nodes, kb.graph)))

    if not useful:
        return Answer(_NOT_FOUND, refused=True)

    partials = "\n".join(f"{number}. {answer}" for number, (answer, _) in enumerate(useful, start=1))
    prompt = (
        "Объедини полезные частичные ответы в один краткий ответ на вопрос. "
        "Не добавляй фактов, кодов или чисел, которых нет в частичных ответах.\n\n"
        f"Вопрос: {question}\n\nЧастичные ответы:\n{partials}"
    )
    text = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    sources = sorted({source for _, selected in useful for source in selected})
    return Answer(text, sources, [answer for answer, _ in useful])


def _community_sources(nodes: list[str], graph: nx.DiGraph) -> CommunitySources:
    members = set(nodes)
    chunks: set[str] = set()
    for node in members:
        if node in graph:
            chunks.update(graph.nodes[node].get("sources", []))
    for source, target, data in graph.edges(data=True):
        if source in members and target in members:
            chunks.update(data.get("sources", []))
    return _document_sources(chunks)
