# Card Knowledge Population Guidelines

## Objective

Populate `features/knowledge/card_knowledge.json` for every Clash Royale card.

The existing entries (Knight, Valkyrie, Firecracker, etc.) are the source of truth for structure and style.

## Rules

- Do NOT change the schema.
- Do NOT rename fields.
- Do NOT add new fields.
- Do NOT remove fields.
- Do NOT modify `enums.md`.
- Preserve JSON formatting.
- Every card must follow the exact same schema.

## Sources of Truth

1. Existing cards in `card_knowledge.json`
2. `features/knowledge/enums.md`
3. Official Clash Royale mechanics
4. Existing Gemini research
5. Official Clash Royale Wiki only when necessary

## Unknown Values

If a value cannot be determined confidently:

- Use `null`
- Never invent values.

## Validation

Every card must satisfy:

- Valid JSON
- Valid enum values
- Required fields present
- No duplicate tags
- No schema deviations