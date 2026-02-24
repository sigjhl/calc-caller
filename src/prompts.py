"""System prompt and tool schema definitions."""
from __future__ import annotations

from typing import Any


def build_system_prompt(schemas: dict[str, Any]) -> str:
    calc_lines = "\n".join(
        f"- {cid} - {s['title']}"
        for cid, s in sorted(schemas.items())
    )
    return (
        "You are OmniCalc, a clinical calculator assistant.\n\n"
        "## Available Calculators\n"
        f"{calc_lines}\n\n"
        "## Workflow\n"
        "1. Identify which calculator to use from the clinical data\n"
        "2. Call `calc_info` to get the exact input field names and units\n"
        "3. Call `execute_calc` with extracted variables using the exact field IDs from the schema\n"
        '4. **CRITICAL**: Once `execute_calc` returns `"success": true`, you MUST stop and respond '
        'with just "Done.". DO NOT call `calc_info` or `execute_calc` again.\n\n'
        "## Rules\n"
        "- If the user explicitly provides a unit, use it.\n"
        "- If no unit is provided, assume the User's Locale Default below is in effect.\n"
        "- **CRITICAL**: If the User's Locale Default differs from the calculator's `canonical_unit` "
        '(or if you are unsure), you MUST pass the unit along with the value as '
        '`{"value": number, "unit": "string"}`.\n'
        "- If required variables are missing, ask for the specific missing values.\n"
        "- If calculation fails, state the error briefly.\n"
        "- Never perform arithmetic yourself - always use `execute_calc`.\n"
        "- NEVER call `execute_calc` again if you already have a successful result.\n"
        "- Be concise. No interpretation or explanation unless asked.\n"
        "- **CRITICAL**: Do NOT loop. Call execute_calc exactly ONCE per request.\n\n"
        "## Locale Defaults\n"
        "- U.S. Conventional Units\n"
    )


# Passed to tokenizer.apply_chat_template(tools=TOOL_SCHEMAS)
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calc_info",
            "description": (
                "Get the schema for a calculator: input field IDs, types, "
                "canonical units, and synonyms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calc_id": {
                        "type": "string",
                        "description": "Calculator identifier (e.g. 'meld_na').",
                    }
                },
                "required": ["calc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_calc",
            "description": "Execute a calculator with the supplied variable values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "calc_id": {
                        "type": "string",
                        "description": "Calculator identifier.",
                    },
                    "variables": {
                        "type": "object",
                        "description": (
                            "Input variables keyed by field ID. "
                            'Each value is {"value": <number|string|bool>} '
                            'or {"value": <number>, "unit": "<string>"} '
                            "when a unit is required or differs from the locale default."
                        ),
                    },
                },
                "required": ["calc_id", "variables"],
            },
        },
    },
]
