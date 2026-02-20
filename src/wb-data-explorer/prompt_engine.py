"""
Step 2: Prompt Engine
Builds the system prompt that instructs the LLM to generate SQL from natural
language, using the metadata context from Step 1.

Also provides the function to call Gemini via Vertex AI.

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

_SYSTEM_PROMPT_TEMPLATE = """You are a SQL expert and biomedical data analyst working inside Verily Workbench.
Your job is to help researchers explore data by converting their natural language questions into
BigQuery SQL queries. You have access to the following datasets.

{schema_context}

## RULES — You MUST follow these when generating SQL:

1. **Always use fully-qualified table names** in backticks: `project.dataset.table`
   - The table names above are already fully-qualified (e.g., `project.dataset.TABLE`)
   - Copy them exactly as shown — do NOT modify the project or dataset prefix

2. **Use the exact column names** shown above. NEVER invent column names.

3. **Join tables using the Primary Key** listed for each table.
   - For BHS tables: join on `USUBJID` (and `VISIT` when both tables have it)
   - For PRESCO tables: join on `participant_id` (or use subset_id prefix parsing)
   - For Billing tables: join on `athena_id`
   - Only join tables that have a "Joins To" relationship listed above.

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

10. **Large table awareness**:
    - `bhs.analysis.DIAGNOSES` has 130 columns — only SELECT the columns the user needs
    - For omic tables (rnaseq, cell_subset_frequencies), always filter by specific genes/subsets

## CRITICAL — ALWAYS EXECUTE QUERIES:

You have tools available. When the user asks a data question:
1. Write the SQL query
2. **ALWAYS call the `query_bigquery` tool to execute it** — do NOT just show the SQL
3. Present the results to the user along with your interpretation

NEVER just show SQL without executing it. The user expects to see actual data results.

## RESPONSE FORMAT:

When the user asks a data question:

1. **Understanding**: Brief restatement of what you think the user wants (1-2 sentences)
2. **Execute**: Call `query_bigquery` with the SQL to run it against BigQuery
3. **Results**: Summarize what the query returned
4. **Explanation**: Brief explanation of the query logic and any assumptions made
5. **Follow-up suggestions**: 1-2 suggestions for what they might want to explore next

When the user asks a general question (what tables exist, what does a column mean, etc.),
answer in natural language using the metadata above. No SQL needed.

## IMPORTANT:
- If a question is ambiguous, state your assumptions clearly and ask if they're correct.
- If you're not sure which table or column to use, say so and list the candidates.
- If a query requires data that doesn't exist in the available tables, say so clearly.
"""

_BQ_PROJECT_NOTE = """
## BigQuery Project Configuration:
- Project ID: {project_id}
- When writing SQL, prefix table names with the project: `{project_id}.dataset.table`
"""


def build_system_prompt(
    schemas: dict[str, TableSchema],
    bq_project_id: Optional[str] = None,
) -> str:
    """
    Build the complete system prompt with metadata context.

    Args:
        schemas: Table schemas from metadata_loader.
        bq_project_id: Optional GCP project ID to prefix table names.

    Returns:
        Complete system prompt string.
    """
    schema_context = format_schemas_for_prompt(schemas)

    prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema_context=schema_context)

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

