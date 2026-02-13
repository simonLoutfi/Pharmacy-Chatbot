from dataclasses import dataclass, field
from typing import FrozenSet, List
from pathlib import Path
import os


def _load_dotenv(path: Path) -> None:
    """Load environment variables from a simple .env file if present."""
    if not path.exists():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Best-effort loading; avoid breaking config import.
        return


_ENV_PATHS = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]
for _env_path in _ENV_PATHS:
    _load_dotenv(_env_path)


@dataclass(frozen=True)
class LLMConfig:
    """Ollama/DeepSeek Coder configuration."""
    model_name: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", "deepseek-coder:6.7b")
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    temperature: float = 0.1  # Low for deterministic code generation
    max_tokens: int = 2048
    timeout: int = 120  # seconds


@dataclass(frozen=True)
class SecurityConfig:
    """Security settings for code validation and execution."""
    
    # Maximum allowed input length
    max_input_length: int = 1000
    
    # Execution limits
    execution_timeout: int = 10  # seconds
    max_memory_mb: int = 512
    
    # Allowed pandas/numpy operations (allowlist)
    allowed_operations: FrozenSet[str] = field(default_factory=lambda: frozenset({
        # DataFrame operations
        'groupby', 'agg', 'aggregate', 'apply', 'transform', 'pipe',
        # Statistical operations
        'mean', 'sum', 'count', 'std', 'var', 'min', 'max',
        'median', 'quantile', 'describe', 'mode', 'sem', 'skew', 'kurt',
        # Filtering & Selection
        'filter', 'query', 'loc', 'iloc', 'isin', 'contains',
        'between', 'isna', 'notna', 'dropna', 'fillna', 'where', 'mask',
        # Transformations
        'merge', 'join', 'concat', 'pivot', 'pivot_table', 'melt',
        'stack', 'unstack', 'explode', 'crosstab',
        # Analysis
        'corr', 'cov', 'value_counts', 'unique', 'nunique', 'duplicated',
        # Sorting & Indexing
        'sort_values', 'sort_index', 'reset_index', 'set_index', 'reindex',
        # Selection
        'head', 'tail', 'sample', 'drop_duplicates', 'nlargest', 'nsmallest',
        # String operations
        'str', 'lower', 'upper', 'strip', 'replace', 'split',
        # DateTime operations
        'dt', 'year', 'month', 'day', 'hour', 'minute',
        # Type operations
        'astype', 'to_numeric', 'to_datetime',
        # Aggregation aliases
        'first', 'last', 'nth', 'size',
        # Comparison
        'eq', 'ne', 'lt', 'le', 'gt', 'ge',
        # Math
        'abs', 'round', 'floor', 'ceil', 'clip',
        # Boolean
        'any', 'all', 'bool',
        # Shape
        'shape', 'columns', 'index', 'values', 'dtypes', 'info',
        'len', 'copy', 'rename', 'assign',
    }))
    
    # Blocked operations (denylist) - security critical
    blocked_operations: FrozenSet[str] = field(default_factory=lambda: frozenset({
        # Code execution (critical)
        'eval', 'exec', 'compile', '__import__', 'execfile',
        'input', 'raw_input',
        # File operations
        'open', 'file', 'read', 'write', 'remove', 'delete',
        'rmdir', 'mkdir', 'chmod', 'chown', 'unlink', 'rename',
        'listdir', 'walk', 'glob', 'scandir',
        # System operations
        'system', 'popen', 'call', 'run', 'spawn', 'kill',
        'fork', 'wait', 'exit', 'quit', 'abort',
        # Network operations
        'socket', 'urllib', 'requests', 'http', 'ftp', 'smtp',
        'connect', 'send', 'recv', 'bind', 'listen',
        # Dangerous builtins
        '__builtins__', '__globals__', '__locals__', '__dict__',
        '__class__', '__bases__', '__subclasses__', '__getattribute__',
        '__setattr__', '__delattr__', '__code__', '__func__',
        # Reflection/metaprogramming
        'getattr', 'setattr', 'delattr', 'hasattr', 'vars', 'dir',
        'globals', 'locals', 'type', 'object',
        # Pickle/serialization (code execution risk)
        'pickle', 'marshal', 'dill', 'shelve', 'load', 'loads', 'dump', 'dumps',
        # Subprocess
        'subprocess', 'Popen', 'check_output', 'check_call',
        # Import-related
        'importlib', '__loader__', '__spec__',
    }))
    
    # Blocked modules (denylist)
    blocked_modules: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'os', 'sys', 'subprocess', 'shutil', 'pickle', 'marshal',
        'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib',
        'sqlite3', 'ctypes', 'multiprocessing', 'threading', 'asyncio',
        'importlib', 'builtins', 'code', 'codeop', 'compile',
        'gc', 'inspect', 'traceback', 'linecache', 'tempfile',
        'pathlib', 'io', 'zipfile', 'tarfile', 'gzip', 'bz2',
    }))
    
    # Allowed variable names in execution context
    allowed_variables: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'df', 'pd', 'np', 'result', 'filtered', 'grouped', 'merged',
        'temp', 'data', 'subset', 'output', 'stats', 'summary',
    }))


@dataclass(frozen=True)
class DataConfig:
    """Data file configuration."""
    data_dir: str = "data"  # Folder containing pharmacy stock movement data
    file_glob: str = "*.xlsx"
    data_file: str = "STORE 2024.xlsx"  # Backward-compat fallback
    sheet_name: str = 0  # First sheet (sheet names vary)
    header_row: int = 3  # Headers are in row 3 (0-indexed)
    skip_rows: int = 4  # Skip first 4 rows when reading


@dataclass(frozen=True)
class UIConfig:
    """Streamlit UI configuration."""
    page_title: str = "Educational Data Chatbot"
    page_icon: str = "bar_chart"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Chat settings
    max_chat_history: int = 50
    memory_turns: int = 6
    
    # Visualization settings
    default_chart_height: int = 400
    chart_theme: str = "plotly_white"


@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Application paths - project root is parent of src/
    base_path: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_path(self) -> Path:
        paths = self.data_paths
        if paths:
            return paths[0]
        return self.base_path / self.data.data_dir / self.data.data_file

    @property
    def data_dir_path(self) -> Path:
        return self.base_path / self.data.data_dir

    @property
    def data_paths(self) -> List[Path]:
        paths = sorted(self.data_dir_path.glob(self.data.file_glob))
        if not paths and self.data.data_file:
            fallback = self.data_dir_path / self.data.data_file
            if fallback.exists():
                paths = [fallback]
        return paths


# Global configuration instance
config = AppConfig()

