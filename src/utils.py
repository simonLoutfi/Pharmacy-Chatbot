

import re
from typing import Tuple, Optional, Iterable
import pandas as pd


def extract_schema(df: pd.DataFrame) -> str:
    """
    Extract DataFrame schema as a formatted string for LLM context.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Formatted schema description string
    """
    schema_parts = []
    
    # Basic info
    schema_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Column details
    schema_parts.append("Columns:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        unique = df[col].nunique()
        
        # Sample values (for categorical or string columns)
        if df[col].dtype == 'object' or unique <= 10:
            samples = df[col].dropna().unique()[:5]
            sample_str = f" | Examples: {list(samples)}"
        else:
            sample_str = f" | Range: [{df[col].min()}, {df[col].max()}]"
        
        schema_parts.append(
            f"  - {col}: {dtype} ({non_null} non-null, {unique} unique){sample_str}"
        )
    
    return "\n".join(schema_parts)


def get_column_descriptions() -> str:
    """
    Get human-readable column descriptions for the pharmacy stock movement dataset.
    
    Returns:
        Formatted column descriptions
    """
    return """
Column Descriptions:
- ARTICLE: Product/medicine name (string)
  Example: A.S.L PANPHARMA 900M, 5-FLUOROURACIL EBEWE
  Identifies what product the movement or transaction is related to.

- MOV DES (Movement Description): Type of stock movement in Arabic (string)
  Describes the operation/reason why quantity changed.
  Examples:
    • تـحـويـل مــن قـســم الـى → Transfer from one department to another
    • دخول مخزون (إضافة) → Stock entry (addition)
    • إلغاء تحويل من قسم الى قسم → Cancelled transfer from one department to another
    • إلغاء بضائع مجانية → Cancellation of free goods

- QTY (Quantity): Number of units moved (numeric)
  Positive values = incoming stock
  Negative values = outgoing stock
  Example: 8, 50, 550, -120

- U.P (Unit Price): Price per single unit of the article (numeric, float)
  Example: 798, 795.428571, 649310
  Used to calculate total transaction value.

- T.P (Total Price): Total value of the transaction (numeric, float)
  Formula: T.P = QTY × U.P
  Example: 39,900 = 50 × 798
           437,000 ≈ 550 × 795.428571

- SOURCE_FILE: Excel filename the row came from (string)
  Example: STORE 2024.xlsx
  Useful for filtering by the original source file.

- SOURCE_YEAR: Year extracted from the filename (integer, may be missing)
  Example: 2021, 2024
  Useful for cross-year comparisons.
""".strip()


def build_code_generation_prompt(question: str, schema: str) -> str:
    """
    Build the prompt for LLM code generation.
    
    Args:
        question: User's natural language question
        schema: DataFrame schema description
        
    Returns:
        Complete prompt for code generation
    """
    return build_code_generation_prompt_with_history(question, schema, history="")


def build_code_generation_prompt_with_history(
    question: str,
    schema: str,
    history: str = ""
) -> str:
    """
    Build the prompt for LLM code generation with optional chat history.
    """
    history_block = f"CONVERSATION CONTEXT:\n{history}\n\n" if history else ""
    return f"""You are a pandas expert. Generate ONLY executable Python pandas code to answer the user's question.

DATAFRAME SCHEMA:
{schema}

{get_column_descriptions()}

{history_block}

EXAMPLE QUERIES:
1. "Total quantity moved by article" → df.groupby('ARTICLE')['QTY'].sum().sort_values(ascending=False)
2. "Average unit price per article" → df.groupby('ARTICLE')['U.P'].mean()
3. "Total value by movement type" → df.groupby('MOV DES')['T.P'].sum()
4. "Top 5 articles by total transaction value" → df.groupby('ARTICLE')['T.P'].sum().nlargest(5)
5. "Movement types and their frequency" → df['MOV DES'].value_counts()
6. "Articles with negative quantity movements" → df[df['QTY'] < 0]['ARTICLE'].unique()
7. "Average QTY per movement type" → df.groupby('MOV DES')['QTY'].mean()

USER QUESTION: "{question}"

INSTRUCTIONS:
1. Generate ONLY Python pandas code - no explanations, no markdown
2. Use 'df' as the DataFrame variable name
3. The last line should be the result expression (no print statements)
4. Use only safe pandas/numpy operations
5. Code must be a single expression or a few lines ending with the result
6. If the question is a follow-up, use the conversation context to resolve references

CODE:"""


def build_response_prompt(question: str, results: str, code: str) -> str:
    """
    Build the prompt for natural language response generation.
    
    Args:
        question: Original user question
        results: Execution results as string
        code: The executed code
        
    Returns:
        Prompt for response formatting
    """
    return build_response_prompt_with_history(question, results, code, history="")


def build_response_prompt_with_history(
    question: str,
    results: str,
    code: str,
    history: str = ""
) -> str:
    """
    Build the prompt for natural language response generation with optional chat history.
    """
    history_block = f"CONVERSATION CONTEXT:\n{history}\n\n" if history else ""
    return f"""You are a helpful data analyst assistant. Explain the analysis results in natural, conversational language.

QUESTION: "{question}"

{history_block}

CODE EXECUTED:
{code}

RESULTS:
{results}

INSTRUCTIONS:
1. Provide a clear, concise explanation of what the results mean
2. Highlight key insights and patterns
3. Use specific numbers from the results
4. Keep the response conversational and helpful
5. Format with markdown for readability
6. Do NOT include the code in your response
7. If the question is a follow-up, use the conversation context to resolve references

RESPONSE:"""


def format_chat_history(
    messages: Optional[Iterable[dict]],
    max_turns: int = 6,
    max_chars: int = 2000,
    max_message_chars: int = 400
) -> str:
    """
    Format chat history into a compact text block for LLM context.
    """
    if not messages:
        return ""

    filtered = [
        msg for msg in messages
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    if not filtered:
        return ""

    limit = max_turns * 2 if max_turns > 0 else len(filtered)
    tail = filtered[-limit:]

    lines = []
    total_chars = 0
    for msg in tail:
        role = msg.get("role", "user").strip().capitalize()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if len(content) > max_message_chars:
            content = content[:max_message_chars].rstrip() + "…"
        line = f"{role}: {content}"
        if total_chars + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines)


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from LLM response.
    
    Handles various formats:
    - Plain code
    - Markdown code blocks (```python or ```)
    - Mixed text and code
    
    Args:
        response: Raw LLM response
        
    Returns:
        Extracted and cleaned code string
    """
    # Try to extract from markdown code block
    patterns = [
        r'```python\s*\n(.*?)```',  # ```python ... ```
        r'```\s*\n(.*?)```',         # ``` ... ```
        r'```(.*?)```',              # Inline code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return clean_code(code)
    
    # If no code block, try to extract code-like lines
    lines = response.strip().split('\n')
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, comments that look like prose, and non-code lines
        if not stripped:
            continue
        if stripped.startswith('#') and len(stripped) > 50:
            continue
        if any(stripped.lower().startswith(x) for x in ['here', 'this', 'the ', 'to ', 'i ']):
            continue
        # Likely code if it contains pandas operations or assignments
        if any(x in stripped for x in ['df', 'pd.', 'np.', '=', '.', '(', '[', 'groupby']):
            code_lines.append(stripped)
    
    if code_lines:
        return clean_code('\n'.join(code_lines))
    
    # Last resort: return cleaned response
    return clean_code(response)


def clean_code(code: str) -> str:
    """
    Clean and normalize extracted code.
    
    Args:
        code: Raw code string
        
    Returns:
        Cleaned code string
    """
    # Remove markdown artifacts
    code = re.sub(r'^```\w*\s*', '', code)
    code = re.sub(r'```\s*$', '', code)
    
    # Remove 'CODE:' prefix if present
    code = re.sub(r'^CODE:\s*', '', code, flags=re.IGNORECASE)
    
    # Remove print statements (we capture the result directly)
    code = re.sub(r'\bprint\s*\((.*)\)', r'\1', code)
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    # Remove any trailing comments that look like explanations
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Keep the line but remove trailing explanation comments
        if '#' in line:
            parts = line.split('#')
            # Keep code part and short comments only
            if len(parts[0].strip()) > 0:
                if len(parts) > 1 and len(parts[1]) < 30:
                    cleaned_lines.append(line)
                else:
                    cleaned_lines.append(parts[0].rstrip())
            else:
                # It's a full-line comment - keep if short
                if len(line) < 50:
                    cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def format_result_for_display(result) -> Tuple[str, str]:
    """
    Format execution result for display.
    
    Args:
        result: The execution result (DataFrame, Series, scalar, etc.)
        
    Returns:
        Tuple of (formatted_text, result_type)
    """
    if isinstance(result, pd.DataFrame):
        if len(result) > 20:
            display_df = pd.concat([result.head(10), result.tail(10)])
            text = f"Showing first 10 and last 10 of {len(result)} rows:\n{display_df.to_string()}"
        else:
            text = result.to_string()
        return text, "dataframe"
    
    elif isinstance(result, pd.Series):
        if len(result) > 20:
            display_series = pd.concat([result.head(10), result.tail(10)])
            text = f"Showing first 10 and last 10 of {len(result)} items:\n{display_series.to_string()}"
        else:
            text = result.to_string()
        return text, "series"
    
    elif isinstance(result, (int, float)):
        text = str(round(result, 4) if isinstance(result, float) else result)
        return text, "scalar"
    
    elif isinstance(result, (list, tuple)):
        text = str(result)
        return text, "list"
    
    else:
        text = str(result)
        return text, "other"


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input for security.
    
    Args:
        text: Raw user input
        max_length: Maximum allowed length
        
    Returns:
        Sanitized input string
        
    Raises:
        ValueError: If input is too long or contains dangerous patterns
    """
    # Check length
    if len(text) > max_length:
        raise ValueError(f"Input too long. Maximum {max_length} characters allowed.")
    
    # Remove potentially dangerous patterns
    text = text.strip()
    
    # Check for code injection patterns
    dangerous_patterns = [
        r'__\w+__',           # Dunder methods
        r'import\s+\w+',      # Import statements
        r'exec\s*\(',         # exec()
        r'eval\s*\(',         # eval()
        r'open\s*\(',         # open()
        r'os\.\w+',           # os module
        r'sys\.\w+',          # sys module
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("Input contains potentially unsafe patterns.")
    
    return text

