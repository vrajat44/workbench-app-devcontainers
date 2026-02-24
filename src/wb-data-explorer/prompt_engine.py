"""
Step 2: Prompt Engine
Builds the system prompt that instructs the LLM to generate SQL from natural
language, using the metadata context from Step 1.

Also provides the function to call Gemini via Vertex AI.

v2 Changes (from benchmark v1 findings):
  - Dynamic join rules derived from loaded metadata (not hardcoded study names)
  - Dynamic large-table awareness from column counts
  - Cross-study prompting: auto-detects multi-project data and generates rules
  - Always-execute policy with graceful fallback on access errors
  - Response format now requires Data Mapping explanation and Limitations
  - Verbose answers must include sample SQL queries

Usage:
    from metadata_loader import load_metadata_from_json_dir, format_schemas_for_prompt
    from prompt_engine import build_system_prompt, call_gemini

    schemas = load_metadata_from_json_dir("/path/to/json/metadata/")
    system_prompt = build_system_prompt(schemas)
    response = call_gemini(system_prompt, "Show me all BHS participants with high cardiac risk")
"""

from __future__ import annotations

from typing import Optional

from metadata_loader import TableSchema, format_schemas_for_prompt


# ── System Prompt Template ────────────────────────────────────────────────────
# Placeholders: {schema_context}, {join_rules}, {large_table_rules}

_SYSTEM_PROMPT_TEMPLATE = """You are a SQL expert and biomedical data analyst working inside Verily Workbench.
Your job is to help researchers explore data by converting their natural language questions into
BigQuery SQL queries. You have access to the following datasets.

{schema_context}

## RULES — You MUST follow these when generating SQL:

1. **Always use fully-qualified table names** in backticks: `project.dataset.table`
   - The table names above are already fully-qualified (e.g., `project.dataset.TABLE`)
   - Copy them exactly as shown — do NOT modify the project or dataset prefix

2. **Use the exact column names** shown above. NEVER invent column names.

{join_rules}

4. **Always add LIMIT** to prevent runaway queries:
   - Use `LIMIT 100` by default for exploration queries
   - Use `LIMIT 1000` for aggregation source data
   - Only omit LIMIT when the user explicitly asks for all rows

5. **For cohort creation queries**, follow this pattern:
   ```sql
   -- Step 1: Define the cohort with filters
   SELECT DISTINCT primary_key_column
   FROM table
   WHERE conditions
   ```
   Then join the cohort to other tables as needed.

6. **For count/summary queries**, always include the total alongside breakdowns.

7. **Handle NULL values** — use IFNULL, COALESCE, or explicit IS NOT NULL as appropriate.

8. **Sensitive columns** (marked [UID] or [PHI]) — note their sensitivity in your response
   but still include them in queries when necessary for the user's task.

9. **Date handling** — use BigQuery date functions (DATE, TIMESTAMP, EXTRACT, DATE_DIFF).

{large_table_rules}

## CRITICAL — ALWAYS EXECUTE QUERIES:

You have tools available. When the user asks a data question:
1. Write the SQL query
2. **ALWAYS call the `query_bigquery` tool to execute it** — do NOT just show the SQL
3. Present the results to the user along with your interpretation
4. **If the query fails due to access permissions, quota, or other infrastructure errors:**
   - Still present your SQL query and explain what it does
   - Explain the error in plain language
   - Suggest how to resolve the access issue
   - Do NOT treat infrastructure errors as a failure of the analysis — the researcher
     still gets value from seeing the query logic and your explanation

NEVER just show SQL without attempting to execute it first.

**When answering questions about data availability or suggesting tables:**
ALWAYS include a sample SQL query the user can run to explore the data, and execute it
using the `query_bigquery` tool. Do not give verbose-only answers without a runnable example.

## RESPONSE FORMAT:

When the user asks a data question:

1. **Understanding**: Brief restatement of what you think the user wants (1-2 sentences)
2. **Data Mapping**: Explain which tables and columns you chose and why
   (e.g., "I mapped 'depression' → PHQ9 table because it contains the PHQ-9 depression questionnaire scores")
3. **Execute**: Call `query_bigquery` with the SQL to run it against BigQuery
4. **Results**: Summarize what the query returned
5. **Limitations**: Note any limitations — missing data, assumptions about clinical thresholds,
   tables not available, or potential biases in the data
6. **Follow-up suggestions**: 1-2 suggestions for what they might want to explore next

When the user asks a general question (what tables exist, what does a column mean, etc.),
answer in natural language using the metadata above. ALSO include a sample SQL query
that the user can start with, and execute it.

## IMPORTANT:
- If a question is ambiguous, state your assumptions clearly and ask if they're correct.
- If you're not sure which table or column to use, say so and list the candidates.
- If a query requires data that doesn't exist in the available tables, say so clearly.
- Always note any limitations, assumptions, or caveats in your analysis.
- NEVER assume the researcher knows which studies, tables, or datasets exist — always explain.
"""

_BQ_PROJECT_NOTE = """
## BigQuery Project Configuration:
- Project ID: {project_id}
- When writing SQL, prefix table names with the project: `{project_id}.dataset.table`
"""


# ── Dynamic Rule Builders ────────────────────────────────────────────────────

def _analyze_data_landscape(schemas: dict[str, TableSchema]) -> dict:
    """
    Analyze loaded schemas to understand the multi-study data landscape.
    Returns a dict with structured info about projects, primary keys,
    large tables, and omic tables — used to generate dynamic prompt rules.
    """
    # Group tables by GCP project (= study)
    projects: dict[str, list[tuple[str, TableSchema]]] = {}
    for fq_name, schema in schemas.items():
        parts = fq_name.split(".")
        project = parts[0] if len(parts) >= 3 else "default"
        projects.setdefault(project, []).append((fq_name, schema))

    # Extract study names from metadata (verily-study-name extension).
    # NOTE: Currently each table JSON carries its own study name (L1 metadata).
    # As the number of studies grows, this should migrate to L2 (study-level)
    # StructureDefinitions that tables reference via canonical URL, providing
    # a single source of truth for study name, description, PI, and governance.
    # See Decision Log DL-1 in metadata_json_creation_playbook.md.
    study_names: dict[str, set[str]] = {}
    for project, tables in projects.items():
        names: set[str] = set()
        for _, schema in tables:
            if schema.study_name:
                names.add(schema.study_name)
        study_names[project] = names

    # Group tables by primary key
    pk_groups: dict[str, list[str]] = {}
    for fq_name, schema in schemas.items():
        pk = schema.primary_key
        if pk:
            pk_groups.setdefault(pk, []).append(fq_name)

    # Identify large tables (>50 columns)
    large_tables = [
        (n, s) for n, s in schemas.items() if len(s.columns) > 50
    ]

    # Identify high-volume tables from MeasureReport data profiles (row counts).
    # Threshold: tables with >100,000 rows get an explicit "always filter" warning.
    # This replaces the old hardcoded keyword list; volume is now metadata-driven.
    HIGH_VOLUME_THRESHOLD = 100_000
    high_volume_tables = [
        (n, s) for n, s in schemas.items()
        if s.row_count is not None and s.row_count > HIGH_VOLUME_THRESHOLD
    ]

    return {
        "projects": projects,
        "study_names": study_names,
        "pk_groups": pk_groups,
        "large_tables": large_tables,
        "high_volume_tables": high_volume_tables,
    }


def _build_join_rules(
    schemas: dict[str, TableSchema],
    landscape: dict,
) -> str:
    """
    Build dynamic join rules (Rule 3) from the loaded metadata.

    Instead of hardcoding "For BHS tables: join on USUBJID", this function
    discovers primary keys, table groupings, and cross-study boundaries
    from the metadata itself.
    """
    pk_groups = landscape["pk_groups"]
    projects = landscape["projects"]
    study_names = landscape["study_names"]

    lines = []
    lines.append("3. **Join tables using the Primary Key** listed for each table.")

    # Primary key groups
    for pk, tables in sorted(pk_groups.items(), key=lambda x: -len(x[1])):
        short_names = [t.split(".")[-1] for t in tables]
        if len(tables) > 1:
            # Check if tables in this PK group share a VISIT column
            has_visit = []
            for t in tables:
                schema = schemas.get(t)
                if schema and any(
                    c.name.upper() in ("VISIT", "VISITNUM") for c in schema.columns
                ):
                    has_visit.append(t.split(".")[-1])

            lines.append(
                f"   - Tables sharing key `{pk}` ({len(tables)} tables): "
                f"{', '.join(short_names)}"
            )
            if has_visit:
                lines.append(
                    f"     → Join on `{pk}` (and `VISIT` when both tables have it "
                    f"for visit-level precision)"
                )
            else:
                lines.append(f"     → Join on `{pk}`")
        else:
            lines.append(f"   - `{short_names[0]}` uses key `{pk}`")

    lines.append(
        '   - Only join tables that share a primary key or have a '
        '"Joins To" relationship listed above.'
    )

    # Cross-study awareness (auto-detected from multiple projects)
    if len(projects) > 1:
        lines.append("")
        lines.append("   **CROSS-STUDY DATA — IMPORTANT:**")
        lines.append(
            "   The data spans multiple independent studies stored in "
            "separate GCP projects:"
        )

        for proj, tbls in sorted(projects.items()):
            pks = sorted(set(s.primary_key for _, s in tbls if s.primary_key))
            short_names = [t.split(".")[-1] for t, _ in tbls]
            names = study_names.get(proj, set())
            label = " / ".join(sorted(names)) if names else proj

            lines.append(
                f"   - **{label}** (project: `{proj}`, {len(tbls)} tables)"
            )
            lines.append(
                f"     Keys: `{'`, `'.join(pks)}`  |  "
                f"Tables: {', '.join(short_names)}"
            )

        lines.append("")
        lines.append(
            "   These studies use **different participant identifiers**. "
            "Participants cannot be directly linked across studies."
        )
        lines.append(
            "   When the user asks about 'all studies', 'across studies', "
            "'both studies', or 'combined':"
        )
        lines.append(
            "   - You MUST include data from ALL available studies in your query"
        )
        lines.append(
            "   - Use UNION ALL to combine per-study results where possible"
        )
        lines.append(
            "   - If data domains don't overlap between studies, explain what "
            "each study offers and why a combined query isn't feasible"
        )
        lines.append(
            "   - NEVER assume the researcher knows which studies or projects "
            "exist — always name and describe them"
        )

    return "\n".join(lines)


def _build_large_table_rules(landscape: dict) -> str:
    """
    Build dynamic large-table awareness rules (Rule 10) from metadata.

    Instead of hardcoding "bhs.analysis.DIAGNOSES has 130 columns", this
    discovers large and omic tables automatically.
    """
    large_tables = landscape["large_tables"]
    high_volume_tables = landscape["high_volume_tables"]

    if not large_tables and not high_volume_tables:
        return ""

    lines = ["10. **Large table awareness**:"]
    seen = set()

    for fq_name, schema in large_tables:
        short = fq_name.split(".")[-1]
        lines.append(
            f"    - `{short}` has {len(schema.columns)} columns — "
            f"only SELECT the columns the user needs"
        )
        seen.add(fq_name)

    for fq_name, schema in high_volume_tables:
        if fq_name not in seen:
            short = fq_name.split(".")[-1]
            row_str = f"{schema.row_count:,}" if schema.row_count else "many"
            lines.append(
                f"    - `{short}` ({row_str} rows, high-volume data) — always "
                f"apply filters; never SELECT * without WHERE clause"
            )

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────

def build_system_prompt(
    schemas: dict[str, TableSchema],
    bq_project_id: Optional[str] = None,
) -> str:
    """
    Build the complete system prompt with metadata context.

    All rules about join keys, study names, and table sizes are derived
    dynamically from the loaded schemas — nothing is hardcoded.

    Args:
        schemas: Table schemas from metadata_loader.
        bq_project_id: Optional GCP project ID to prefix table names.

    Returns:
        Complete system prompt string.
    """
    schema_context = format_schemas_for_prompt(schemas)

    # Analyze the loaded schemas to build dynamic rules
    landscape = _analyze_data_landscape(schemas)
    join_rules = _build_join_rules(schemas, landscape)
    large_table_rules = _build_large_table_rules(landscape)

    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        schema_context=schema_context,
        join_rules=join_rules,
        large_table_rules=large_table_rules,
    )

    if bq_project_id:
        prompt += _BQ_PROJECT_NOTE.format(project_id=bq_project_id)

    return prompt


def build_error_fix_prompt(
    original_query: str,
    error_message: str,
    schemas: dict[str, TableSchema],
) -> str:
    """
    Build a prompt asking the LLM to fix a failed SQL query.
    Used in the query-retry loop (Step 3).

    Args:
        original_query: The SQL that failed.
        error_message: The BigQuery error message.
        schemas: Table schemas for context.

    Returns:
        Prompt string for the fix request.
    """
    schema_context = format_schemas_for_prompt(schemas)

    return f"""The following BigQuery SQL query failed with an error. 
Please fix the query and return ONLY the corrected SQL in a ```sql code block.

## Failed Query:
```sql
{original_query}
```

## Error Message:
{error_message}

## Available Tables and Columns:
{schema_context}

## Instructions:
- Fix the specific error mentioned above
- Use only the exact column names and table names listed in the schema
- Return ONLY the corrected SQL query in a ```sql block — no explanation needed
"""


# ── Gemini Client ─────────────────────────────────────────────────────────────

def call_gemini(
    system_prompt: str,
    user_message: str,
    model_name: str = "gemini-2.5-pro",
    project_id: Optional[str] = None,
    location: str = "us-central1",
    temperature: float = 0.1,
    max_output_tokens: int = 4096,
) -> str:
    """
    Call Gemini via the google-genai SDK (Vertex AI backend).

    When running inside a Workbench cloud app, ADC handles auth automatically.
    For local testing, run `gcloud auth application-default login` first.

    Args:
        system_prompt: The system instructions.
        user_message: The user's natural language query.
        model_name: Gemini model to use.
        project_id: GCP project ID. If None, uses ADC default.
        location: GCP region.
        temperature: LLM temperature (lower = more deterministic).
        max_output_tokens: Max response length.

    Returns:
        The LLM's text response.
    """
    from google import genai
    from google.genai.types import GenerateContentConfig

    # Initialize client — uses ADC automatically in Workbench
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=config,
    )

    return response.text


def extract_sql_from_response(response: str) -> Optional[str]:
    """
    Extract the SQL query from an LLM response that contains a ```sql block.

    Returns None if no SQL block is found.
    """
    import re

    # Match ```sql ... ``` blocks
    pattern = r"```sql\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try ``` ... ``` without sql tag
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        text = match.group(1).strip()
        # Only return if it looks like SQL
        if any(kw in text.upper() for kw in ["SELECT", "WITH", "INSERT", "CREATE"]):
            return text

    return None


# ── Main (for quick testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from metadata_loader import load_metadata_from_json_dir

    # Default metadata path
    default_json_dir = "/Users/rajat/Verily1/product_mgmnt/Metadata/Metadata JSON for Demo/JSON Metadata/"

    json_dir = sys.argv[1] if len(sys.argv) > 1 else default_json_dir
    print(f"Loading metadata from: {json_dir}")

    schemas = load_metadata_from_json_dir(json_dir)
    system_prompt = build_system_prompt(schemas)

    print(f"\nSystem prompt length: {len(system_prompt)} characters")
    print(f"System prompt lines: {system_prompt.count(chr(10))} lines")
    print()

    # Test questions (no LLM call — just show what would be sent)
    test_questions = [
        "How many BHS participants are eligible for the study?",
        "Show me PRESCO participants with PASC who are also progressors",
        "What are the top 10 ICD-10 codes in billing claims?",
        "Join BHS cohort eligibility with ASCVD risk scores for year one visits",
        "What tables are available?",
    ]

    print("=" * 70)
    print("TEST QUESTIONS (will be sent as user messages):")
    print("=" * 70)
    for i, q in enumerate(test_questions, 1):
        print(f"  Q{i}: {q}")

    print()
    print("To test with Gemini, set GCP project and run:")
    print("  from prompt_engine import call_gemini, build_system_prompt")
    print("  response = call_gemini(system_prompt, 'your question here', project_id='your-project')")

    # If --call flag is passed, actually call Gemini
    if "--call" in sys.argv:
        project_id = None
        for arg in sys.argv:
            if arg.startswith("--project="):
                project_id = arg.split("=")[1]

        if not project_id:
            print("\nERROR: Pass --project=YOUR_PROJECT_ID to call Gemini")
            sys.exit(1)

        question = test_questions[0]
        for arg in sys.argv:
            if arg.startswith("--question="):
                question = arg.split("=", 1)[1]

        print(f"\n🤖 Calling Gemini with: '{question}'")
        print("-" * 70)
        response = call_gemini(system_prompt, question, project_id=project_id)
        print(response)

        sql = extract_sql_from_response(response)
        if sql:
            print("\n📋 Extracted SQL:")
            print(sql)
