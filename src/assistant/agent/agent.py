"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_SYSTEM_PROMPT = """Ты ассистент диспетчера сервисной службы «ПолярТех».
Никогда не отвечай по памяти: для каждого вопроса сначала вызови подходящий инструмент.
Используй search_knowledge для точных вопросов по документам, summarize_topic для обзорных,
а ERP-инструменты только после того, как номер детали появился в вопросе или результате поиска.
Не выдумывай коды, цены и остатки. Нулевой остаток не означает отсутствие детали в ERP.
Если пользователь просит оформить заказ, обязательно вызови create_service_order после определения детали;
код сам остановит действие для подтверждения. Заказ готовь только с количеством, явно названным пользователем.
Отвечай кратко по-русски."""
_WITH_UNIT = re.compile(r"(?<![\w-])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)\s*(руб|₽|шт)", re.IGNORECASE)
_REPLACEMENT = re.compile(r"\b(P-\d{4}[A-Z]?)\s+-REPLACED_BY->\s+(P-\d{4}[A-Z]?)\b")


def _cache_key(name: str, arguments: dict) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def _quantity_was_asked(question: str, quantity: object) -> bool:
    if not isinstance(quantity, int) or isinstance(quantity, bool):
        return False
    return re.search(rf"(?<!\d){quantity}(?!\d)", question) is not None


def _grounded_part(part_id: object, known_text: str) -> bool:
    if not isinstance(part_id, str):
        return False
    normalized = normalize_code(part_id)
    return normalized is not None and normalized in find_codes(known_text)


def _latest_replacement(results: list[dict]) -> str | None:
    edges = {
        match.groups()
        for result in results
        if isinstance(result, dict)
        for fact in result.get("facts", [])
        if isinstance(fact, str)
        for match in _REPLACEMENT.finditer(fact)
    }
    terminals = {target for _, target in edges} - {source for source, _ in edges}
    return next(iter(terminals)) if len(terminals) == 1 else None


def _answer_is_grounded(answer: str, question: str, results: list[dict]) -> bool:
    result_text = json.dumps(results, ensure_ascii=False)
    allowed_codes = set(find_codes(question)) | set(find_codes(result_text))
    if any(code not in allowed_codes for code in find_codes(answer)):
        return False

    allowed_prices = {
        str(result["price_rub"])
        for result in results
        if isinstance(result, dict) and isinstance(result.get("price_rub"), int)
    }
    allowed_quantities = {
        str(result["qty"])
        for result in results
        if isinstance(result, dict) and isinstance(result.get("qty"), int)
    }
    for raw, unit in _WITH_UNIT.findall(answer):
        number = re.sub(r"\D", "", raw)
        if unit.lower() in ("руб", "₽") and number not in allowed_prices:
            return False
        if unit.lower() == "шт" and number not in allowed_quantities:
            return False
    return True


def _workflow_incomplete(question: str, steps: list[ToolStep]) -> str | None:
    called = {step.tool for step in steps}
    lowered = question.lower()
    if not steps:
        return "Сначала вызови подходящий инструмент и только затем отвечай."
    asks_for_order = any(word in lowered for word in ("оформи заказ", "создай заказ", "закажи"))
    if asks_for_order and "create_service_order" not in called:
        return "Пользователь просит заказ: вызови create_service_order с подтверждёнными аргументами."
    if any(word in lowered for word in ("сколько стоит", "цена", "стоимость")) and "get_price" not in called:
        return "Для цены обязательно вызови get_price."
    if any(word in lowered for word in ("сколько штук", "есть ли", "остаток")) and "check_stock" not in called:
        return "Для остатка обязательно вызови check_stock."
    return None


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    results: list[dict] = []
    cached: dict[str, tuple[dict | None, str | None]] = {}
    knowledge_refused = False
    missing_data = False
    nudges = 0

    while len(steps) < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            answer = response.content.strip()
            incomplete = _workflow_incomplete(question, steps)
            if incomplete and nudges < 2:
                nudges += 1
                messages.append({"role": "user", "content": incomplete})
                continue
            if _answer_is_grounded(answer, question, results):
                refused = knowledge_refused or missing_data
                return AgentResult(answer, steps, sorted(sources), refused)

            correction = (
                "В ответе есть код, цена или остаток, которых нет в вопросе или результатах инструментов. "
                "Исправь ответ, используя только подтверждённые данные."
            )
            messages.append({"role": "user", "content": correction})
            fixed = llm.chat(messages, tools=TOOL_SCHEMAS).content.strip()
            if _answer_is_grounded(fixed, question, results):
                refused = knowledge_refused or missing_data
                return AgentResult(fixed, steps, sorted(sources), refused)
            return AgentResult("Не удалось подготовить ответ, подтверждённый данными.", steps, sorted(sources), True)

        for call in response.tool_calls:
            if len(steps) >= max_steps:
                break
            arguments = call.arguments
            error = None
            result = None
            if call.name not in tools:
                error = f"Неизвестный инструмент: {call.name}"
            elif not isinstance(arguments, dict) or "_raw" in arguments:
                error = "Аргументы инструмента не удалось разобрать"
            elif "part_id" in arguments and not _grounded_part(
                arguments["part_id"], f"{question}\n{json.dumps(results, ensure_ascii=False)}"
            ):
                error = "Номер детали не встречался в вопросе или результатах. Сначала вызови search_knowledge."
            elif (
                call.name in ("check_stock", "get_price", "create_service_order")
                and (latest := _latest_replacement(results)) is not None
                and normalize_code(str(arguments.get("part_id", ""))) != latest
            ):
                error = f"Указанная деталь заменена. Используй последнюю актуальную деталь цепочки: {latest}."
            elif call.name == "create_service_order" and not _quantity_was_asked(question, arguments.get("qty")):
                error = "Количество не названо пользователем. Уточни, сколько штук заказать."
            elif call.name == "create_service_order" and not confirm:
                part = arguments.get("part_id", "")
                quantity = arguments.get("qty", "")
                warehouse = arguments.get("warehouse", "")
                return AgentResult(
                    f"Требуется подтверждение заказа детали {part} на склад {warehouse}; "
                    f"количество указано пользователем ({quantity}).",
                    steps,
                    sorted(sources),
                    needs_confirmation=True,
                    pending_action={"tool": call.name, "args": arguments},
                )
            else:
                key = _cache_key(call.name, arguments)
                if key in cached:
                    result, error = cached[key]
                else:
                    for attempt in range(3):
                        try:
                            result = tools[call.name](**arguments)
                            if isinstance(result, dict) and "error" in result:
                                error = str(result["error"])
                            break
                        except ErpUnavailable as exc:
                            if attempt == 2:
                                error = str(exc)
                        except (TypeError, ValueError) as exc:
                            error = f"Некорректные аргументы: {exc}"
                            break
                    cached[key] = (result, error)

            step = ToolStep(call.name, arguments, result if error is None else None, error)
            steps.append(step)
            if error and error.lower().startswith("данных нет"):
                missing_data = True
            if result is not None:
                results.append(result)
                if isinstance(result.get("sources"), list):
                    sources.update(result["sources"])
                if call.name in ("search_knowledge", "summarize_topic") and result.get("refused") is True:
                    knowledge_refused = True
            observation = result if error is None and result is not None else {"error": error or "Нет результата"}
            messages.append(tool_message(call, observation))

    learned = json.dumps(results, ensure_ascii=False)
    return AgentResult(f"Работа прервана по лимиту шагов. Полученные данные: {learned}", steps, sorted(sources))
