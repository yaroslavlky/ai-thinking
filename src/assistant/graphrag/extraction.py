"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

_SYSTEM_PROMPT = """Ты извлекаешь граф знаний сервисной службы из одного фрагмента документа.
Верни только сущности и связи, прямо подтверждённые текстом, строго по JSON-схеме.
Пиши компактный JSON в одну строку без отступов. Каждую сущность перечисляй один раз.

Имена сущностей:
- Model, Part, ErrorCode и Procedure — только коды: AR-210, P-1042B, E-17, SP-104;
- Component — название узла, например «компрессор» или «пусковое реле»;
- ProductLine — название линейки; Supplier — название поставщика.
Название узла никогда не является Part. Код детали никогда не является Component.

Допустимые направления связей:
- Model -> BELONGS_TO -> ProductLine
- Model -> HAS_PART -> Part
- Model -> FAILED -> Component
- Part -> IS_A -> Component
- Part -> REPLACED_BY -> Part (старая деталь указывает на новую)
- Part -> SUPPLIED_BY -> Supplier
- ErrorCode -> INDICATES -> Component
- ErrorCode -> FIXED_BY -> Procedure

Не создавай сущности для номеров документов SB-* и R-*. Не меняй направление связи.
Пример 1: «Ошибка E-17 означает отказ пускового реле» даёт сущности ErrorCode E-17 и
Component пусковое реле, связь ErrorCode:E-17 -INDICATES-> Component:пусковое реле.
Пример 2: в руководстве модели AR-210 строка «пусковое реле — деталь P-1042, поставщик
ХолодКомплект» даёт три связи: Model:AR-210 -HAS_PART-> Part:P-1042,
Part:P-1042 -IS_A-> Component:пусковое реле и Part:P-1042 -SUPPLIED_BY-> Supplier:ХолодКомплект.
Пример 3: «деталь P-1042 снята, вместо неё P-1042B» даёт
Part:P-1042 -REPLACED_BY-> Part:P-1042B.
Если подходящих фактов нет, верни пустые массивы entities и relations."""


def _parse(content: str) -> dict:
    data = json.loads(content)
    if not isinstance(data, dict) or set(("entities", "relations")) - data.keys():
        raise ValueError("ответ должен быть объектом с entities и relations")
    if not isinstance(data["entities"], list) or not isinstance(data["relations"], list):
        raise ValueError("entities и relations должны быть массивами")
    for raw in data["entities"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("type"), str) or not isinstance(raw.get("name"), str):
            raise ValueError("некорректная сущность")
    required = ("subject_type", "subject", "type", "object_type", "object")
    for raw in data["relations"]:
        if not isinstance(raw, dict) or any(not isinstance(raw.get(key), str) for key in required):
            raise ValueError("некорректная связь")
    return data


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Фрагмент документа:\n\n{chunk.text}"},
    ]
    data = None
    for attempt in range(2):
        request = messages
        if attempt:
            request = [
                *messages,
                {"role": "user", "content": "Предыдущий ответ не соответствовал JSON-схеме. Верни корректный JSON."},
            ]
        try:
            data = _parse(llm.chat(request, json_schema=EXTRACTION_SCHEMA, max_tokens=1536).content)
            break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    if data is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    vocab = vocabulary()
    codes_in_text = set(find_codes(chunk.text))
    names_in_text = set(find_names(chunk.text, vocab))
    entities: dict[str, Entity] = {}
    dropped = 0

    def grounded(entity_type: str, raw_name: str) -> Entity | None:
        name = canonical_name(entity_type, raw_name, vocab)
        if name is None:
            return None
        entity = Entity(entity_type, name)
        if entity_type in ("Model", "Part", "ErrorCode", "Procedure"):
            return entity if name in codes_in_text else None
        return entity if (entity_type, name) in names_in_text else None

    for raw in data["entities"]:
        entity = grounded(raw["type"], raw["name"])
        if entity is None:
            dropped += 1
        else:
            entities[entity.node_id] = entity

    relations: dict[tuple[str, str, str], Relation] = {}
    for raw in data["relations"]:
        subject = grounded(raw["subject_type"], raw["subject"])
        object_ = grounded(raw["object_type"], raw["object"])
        if subject is None or object_ is None:
            dropped += 1
            continue
        relation = Relation(subject, raw["type"], object_)
        entities[subject.node_id] = subject
        entities[object_.node_id] = object_
        relations[(subject.node_id, relation.type, object_.node_id)] = relation

    # В регламенте номер процедуры и применимые коды ошибок заданы явно. Модель иногда извлекает
    # обе сущности, но пропускает FIXED_BY; восстанавливаем только связь между кодами из самого текста.
    if chunk.doc_id.startswith("procedure_"):
        procedures = [code for code in codes_in_text if code_type(code) == "Procedure"]
        errors = [code for code in codes_in_text if code_type(code) == "ErrorCode"]
        if len(procedures) == 1:
            procedure = Entity("Procedure", procedures[0])
            entities[procedure.node_id] = procedure
            for code in errors:
                error = Entity("ErrorCode", code)
                relation = Relation(error, "FIXED_BY", procedure)
                entities[error.node_id] = error
                relations[(error.node_id, relation.type, procedure.node_id)] = relation

    return ExtractionResult(
        chunk.chunk_id,
        chunk.doc_id,
        list(entities.values()),
        list(relations.values()),
        dropped,
    )
