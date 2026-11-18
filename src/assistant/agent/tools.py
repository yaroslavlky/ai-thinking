"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
TOOL_SCHEMAS: list[dict] = [
    _function(
        "search_knowledge",
        "Найти точные факты о модели, ошибке, узле, детали, замене или регламенте в графе знаний.",
        {"question": {"type": "string", "description": "Точный вопрос к базе знаний"}},
        ["question"],
    ),
    _function(
        "summarize_topic",
        "Ответить на обзорный вопрос по всему корпусу через сводки сообществ.",
        {"question": {"type": "string", "description": "Обзорный вопрос"}},
        ["question"],
    ),
    _function(
        "check_stock",
        "Проверить остаток известной детали на конкретном складе ERP.",
        {
            "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
            "warehouse": {"type": "string", "enum": WAREHOUSES},
        },
        ["part_id", "warehouse"],
    ),
    _function(
        "get_price",
        "Получить актуальную цену известной детали из ERP.",
        {"part_id": {"type": "string", "description": "Номер детали, например P-1042B"}},
        ["part_id"],
    ),
    _function(
        "create_service_order",
        "Создать заказ детали после явного подтверждения пользователя.",
        {
            "part_id": {"type": "string", "description": "Номер детали"},
            "warehouse": {"type": "string", "enum": WAREHOUSES},
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "description": "Код модели, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    def search_knowledge(question: str = "") -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Вопрос должен быть непустой строкой"}
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(question: str = "") -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Вопрос должен быть непустой строкой"}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def valid_part(part_id: object) -> str | None:
        if not isinstance(part_id, str):
            return None
        normalized = normalize_code(part_id)
        return normalized if normalized and code_type(normalized) == "Part" else None

    def checked_erp(call: Callable[[], dict]) -> dict:
        try:
            return call()
        except ErpNotFound as exc:
            return {"error": f"Данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    def check_stock(part_id: object = None, warehouse: object = None) -> dict:
        part = valid_part(part_id)
        if part is None:
            return {"error": "Некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад"}
        return checked_erp(lambda: erp.get_stock(part, warehouse))

    def get_price(part_id: object = None) -> dict:
        part = valid_part(part_id)
        if part is None:
            return {"error": "Некорректный номер детали"}
        return checked_erp(lambda: erp.get_price(part))

    def create_service_order(
        part_id: object = None, warehouse: object = None, qty: object = None, model: object = None
    ) -> dict:
        part = valid_part(part_id)
        if part is None:
            return {"error": "Некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад"}
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {"error": "Количество должно быть целым положительным числом"}
        normalized_model = normalize_code(model) if isinstance(model, str) else None
        if normalized_model is None or code_type(normalized_model) != "Model":
            return {"error": "Некорректный код модели"}
        return checked_erp(lambda: erp.create_order(part, warehouse, qty, normalized_model))

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
