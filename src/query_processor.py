

import time
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from pathlib import Path

import pandas as pd

from config import config
from exceptions import (
    ChatbotError, CodeGenerationError, CodeValidationError,
    CodeExecutionError, DataLoadError, InputValidationError
)
from logger import app_logger
from utils import (
    extract_schema,
    sanitize_input,
    format_result_for_display,
    format_chat_history
)
from code_generator import get_code_generator, get_response_generator, GenerationResult
from code_validator import validate_code, ValidationResult
from code_executor import execute_code, ExecutionResult


@dataclass
class QueryResult:
    """Complete result of a query processing."""
    success: bool
    question: str
    answer: str
    data: Any = None
    data_type: str = "none"
    code: str = ""
    execution_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    error: Optional[str] = None
    error_code: Optional[str] = None
    warnings: list = field(default_factory=list)
    
    @property
    def has_data(self) -> bool:
        """Check if result contains displayable data."""
        return self.data is not None and self.data_type != "none"


class DataManager:
    """
    Manages data loading and caching.
    
    Implements lazy loading and caching for the DataFrame.
    """
    
    def __init__(self, data_path: Path = None):
        """Initialize with data file path."""
        self.data_path = data_path or config.data_path
        self._data_paths = [self.data_path] if data_path else None
        self._df: Optional[pd.DataFrame] = None
        self._schema: Optional[str] = None
    
    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and cache the DataFrame.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Loaded DataFrame
        """
        if self._df is None or force_reload:
            try:
                data_paths = self._data_paths or config.data_paths
                if not data_paths:
                    raise DataLoadError(
                        message="No Excel files found in data folder",
                        file_path=str(config.data_dir_path)
                    )

                app_logger.info(
                    "Loading data files",
                    count=len(data_paths),
                    paths=[str(p) for p in data_paths[:5]]
                )

                frames = []
                for path in data_paths:
                    df = pd.read_excel(
                        path,
                        sheet_name=config.data.sheet_name,
                        header=config.data.header_row
                    )

                    # Clean up column names (remove leading/trailing whitespace)
                    df.columns = df.columns.str.strip()

                    # Remove rows that are entirely NaN
                    df = df.dropna(how='all')

                    # Tag source metadata for multi-file queries
                    df["SOURCE_FILE"] = path.name
                    year = self._extract_year_from_filename(path.stem)
                    df["SOURCE_YEAR"] = year if year is not None else pd.NA

                    frames.append(df)

                if not frames:
                    raise DataLoadError(
                        message="No usable Excel data found",
                        file_path=str(config.data_dir_path)
                    )

                self._df = pd.concat(frames, ignore_index=True, sort=False)

                app_logger.info(
                    "Data loaded successfully",
                    shape=str(self._df.shape),
                    columns=list(self._df.columns)
                )

            except DataLoadError:
                raise
            except FileNotFoundError:
                raise DataLoadError(
                    message="Data file not found",
                    file_path=str(config.data_dir_path)
                )
            except Exception as e:
                raise DataLoadError(
                    message=f"Error loading data: {str(e)}",
                    file_path=str(config.data_dir_path)
                )
        
        return self._df
    
    def get_schema(self, force_refresh: bool = False) -> str:
        """
        Get or generate DataFrame schema description.
        
        Args:
            force_refresh: Force schema regeneration
            
        Returns:
            Schema description string
        """
        if self._schema is None or force_refresh:
            df = self.load_data()
            self._schema = extract_schema(df)
        
        return self._schema
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the DataFrame (loads if needed)."""
        return self.load_data()

    @staticmethod
    def _extract_year_from_filename(name: str) -> Optional[int]:
        """Extract a year from a filename (e.g., STORE 2024, Store21)."""
        match_4 = re.search(r'(?<!\d)(19|20)\d{2}(?!\d)', name)
        if match_4:
            return int(match_4.group(0))

        match_2 = re.search(r'(?<!\d)(\d{2})(?!\d)', name)
        if match_2:
            year_two = int(match_2.group(1))
            return 2000 + year_two

        return None
    
    @property
    def schema(self) -> str:
        """Get the schema (generates if needed)."""
        return self.get_schema()


class QueryProcessor:
    """
    Main query processing orchestrator.
    
    Coordinates:
    1. Input validation
    2. Code generation (LLM)
    3. Code validation (security)
    4. Code execution (sandboxed)
    5. Response formatting (LLM)
    """
    
    def __init__(self, data_manager: DataManager = None):
        """Initialize with optional data manager."""
        self.data_manager = data_manager or DataManager()
        self._code_generator = None
        self._response_generator = None
    
    @property
    def code_generator(self):
        """Lazy load code generator."""
        if self._code_generator is None:
            self._code_generator = get_code_generator()
        return self._code_generator
    
    @property
    def response_generator(self):
        """Lazy load response generator."""
        if self._response_generator is None:
            self._response_generator = get_response_generator()
        return self._response_generator
    
    def process_question(
        self,
        question: str,
        history: str = "",
        messages: Optional[list] = None,
        **_: object
    ) -> QueryResult:
        """
        Process a user question through the complete pipeline.
        
        Args:
            question: User's natural language question
            
        Returns:
            QueryResult with answer, data, and metadata
        """
        start_time = time.perf_counter()
        warnings = []
        
        app_logger.info(
            "Processing question",
            question=question[:100]
        )
        
        try:
            # Step 1: Input validation
            question = self._validate_input(question)
            
            # Step 2: Load data and schema
            df = self.data_manager.df
            schema = self.data_manager.schema

            history_text = history or format_chat_history(
                messages,
                max_turns=config.ui.memory_turns
            )
            
            # Step 3: Generate code
            gen_result = self._generate_code(question, schema, history_text)
            
            if not gen_result.success:
                return self._create_error_result(
                    question=question,
                    error="Could not generate analysis code",
                    error_code="GENERATION_FAILED",
                    start_time=start_time
                )
            
            # Step 4: Validate code
            try:
                val_result = self._validate_code(gen_result.code)
                warnings.extend(val_result.warnings)
            except (CodeValidationError, Exception) as e:
                return self._create_error_result(
                    question=question,
                    error=str(e.user_message if hasattr(e, 'user_message') else e),
                    error_code="VALIDATION_FAILED",
                    start_time=start_time
                )
            
            # Step 5: Execute code
            exec_result = self._execute_code(val_result.sanitized_code, df)
            
            if not exec_result.success:
                return self._create_error_result(
                    question=question,
                    error=exec_result.error or "Execution failed",
                    error_code="EXECUTION_FAILED",
                    start_time=start_time,
                    code=gen_result.code
                )
            
            # Step 6: Format response
            formatted_result, result_type = format_result_for_display(exec_result.result)
            answer = self._generate_response(
                question,
                formatted_result,
                gen_result.code,
                history_text
            )
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            app_logger.query(question, success=True, duration_ms=total_time)
            
            return QueryResult(
                success=True,
                question=question,
                answer=answer,
                data=exec_result.result,
                data_type=result_type,
                code=gen_result.code,
                execution_time_ms=exec_result.execution_time_ms,
                generation_time_ms=gen_result.generation_time_ms,
                total_time_ms=total_time,
                warnings=warnings
            )
            
        except ChatbotError as e:
            return self._create_error_result(
                question=question,
                error=e.user_message,
                error_code=e.code.name if hasattr(e, 'code') else "ERROR",
                start_time=start_time
            )
            
        except Exception as e:
            app_logger.error(
                "Unexpected error processing question",
                error=str(e),
                error_type=type(e).__name__
            )
            return self._create_error_result(
                question=question,
                error="An unexpected error occurred. Please try again.",
                error_code="UNEXPECTED_ERROR",
                start_time=start_time
            )
    
    def _validate_input(self, question: str) -> str:
        """Validate and sanitize user input."""
        try:
            return sanitize_input(question, config.security.max_input_length)
        except ValueError as e:
            raise InputValidationError(str(e))
    
    def _generate_code(self, question: str, schema: str, history: str = "") -> GenerationResult:
        """Generate pandas code from question."""
        return self.code_generator.generate_from_question(question, schema, history)
    
    def _validate_code(self, code: str) -> ValidationResult:
        """Validate generated code for security."""
        return validate_code(code)
    
    def _execute_code(self, code: str, df: pd.DataFrame) -> ExecutionResult:
        """Execute validated code."""
        return execute_code(code, df)
    
    def _generate_response(self, question: str, results: str, code: str, history: str = "") -> str:
        """Generate natural language response."""
        try:
            return self.response_generator.generate_response(question, results, code, history)
        except Exception as e:
            app_logger.warning(
                "Response generation failed",
                error=str(e)
            )
            return f"Here are the results:\n\n{results}"
    
    def _create_error_result(
        self,
        question: str,
        error: str,
        error_code: str,
        start_time: float,
        code: str = ""
    ) -> QueryResult:
        """Create an error result."""
        total_time = (time.perf_counter() - start_time) * 1000
        
        app_logger.query(question, success=False, duration_ms=total_time)
        
        return QueryResult(
            success=False,
            question=question,
            answer=f"Error: {error}",
            error=error,
            error_code=error_code,
            code=code,
            total_time_ms=total_time
        )
    
    def check_system(self) -> Dict[str, bool]:
        """
        Check system health and connectivity.
        
        Returns:
            Dict with component health status
        """
        status = {
            'data_loaded': False,
            'ollama_connected': False,
            'model_available': False,
            'ready': False
        }
        
        # Check data
        try:
            self.data_manager.load_data()
            status['data_loaded'] = True
        except Exception:
            pass
        
        # Check Ollama connection
        try:
            status['ollama_connected'] = self.code_generator.check_connection()
            status['model_available'] = status['ollama_connected']
        except Exception:
            pass
        
        status['ready'] = all([
            status['data_loaded'],
            status['ollama_connected'],
            status['model_available']
        ])
        
        return status


# Global instances
_data_manager: Optional[DataManager] = None
_processor: Optional[QueryProcessor] = None


def get_data_manager() -> DataManager:
    """Get or create the global data manager."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


def get_processor() -> QueryProcessor:
    """Get or create the global query processor."""
    global _processor
    if _processor is None:
        _processor = QueryProcessor(get_data_manager())
    return _processor


def process_question(
    question: str,
    history: str = "",
    messages: Optional[list] = None,
    **_: object
) -> QueryResult:
    """
    Convenience function for processing questions.
    
    Args:
        question: User's question
        
    Returns:
        QueryResult with answer and data
    """
    return get_processor().process_question(question, history=history, messages=messages)

