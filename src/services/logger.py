"""
Lore Lantern Logging System

Clean terminal output for production + detailed file logging for debugging.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
from functools import wraps
import sys
import json


class LoreLanternLogger:
    """
    Two-mode logging system:
    - Terminal: Clean, timestamped key events only
    - Debug file: Full detailed logs for troubleshooting
    """

    def __init__(self, debug_mode: bool = False, settings=None):
        self.debug_mode = debug_mode
        self.settings = settings
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Create debug directory if debug flags are enabled
        if settings and (settings.debug_agent_io or settings.debug_storage or settings.debug_api_calls):
            self.debug_log_dir = Path(settings.debug_log_dir)
            self.debug_log_dir.mkdir(parents=True, exist_ok=True)
            self.debug_log_format = settings.debug_log_format

            # Setup JSON log files for structured logging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if settings.debug_agent_io:
                self.agent_io_log = self.debug_log_dir / f"agent_io_{timestamp}.jsonl"
            if settings.debug_storage:
                self.storage_log = self.debug_log_dir / f"storage_{timestamp}.jsonl"
            if settings.debug_api_calls:
                self.api_calls_log = self.debug_log_dir / f"api_calls_{timestamp}.jsonl"

        # Setup file logger for debug mode
        if debug_mode:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"lorelantern_debug_{timestamp}.txt"

            self.file_logger = logging.getLogger("lorelantern_debug")
            self.file_logger.setLevel(logging.DEBUG)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.file_logger.addHandler(file_handler)

            print(f"📝 Debug mode enabled. Logging to: {log_file}")

    def _timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime("%H:%M:%S")

    def _terminal_log(self, emoji: str, message: str, color: str = ""):
        """Print clean log to terminal"""
        timestamp = self._timestamp()

        # ANSI color codes
        colors = {
            "green": "\033[92m",
            "blue": "\033[94m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "cyan": "\033[96m",
            "reset": "\033[0m"
        }

        color_code = colors.get(color, "")
        reset = colors["reset"] if color_code else ""

        print(f"{color_code}[{timestamp}] {emoji} {message}{reset}")

    def _debug_log(self, level: str, component: str, message: str, data: Optional[dict] = None):
        """Write detailed log to debug file"""
        if self.debug_mode and hasattr(self, 'file_logger'):
            log_msg = f"{component} | {message}"
            if data:
                log_msg += f" | Data: {data}"

            log_func = getattr(self.file_logger, level.lower(), self.file_logger.info)
            log_func(log_msg)

    # ===== Terminal Output Methods =====

    def job_received(self, job_type: str, story_id: str, details: str = ""):
        """Log when a job is received"""
        msg = f"Job received: {job_type} (Story: {story_id[:8]})"
        if details:
            msg += f" - {details}"
        self._terminal_log("📨", msg, "cyan")
        self._debug_log("info", "JOB", f"Received {job_type}", {
            "story_id": story_id,
            "details": details
        })

    def job_completed(self, job_type: str, story_id: str, duration: Optional[float] = None):
        """Log when a job completes"""
        msg = f"Job completed: {job_type} (Story: {story_id[:8]})"
        if duration:
            msg += f" in {duration:.1f}s"
        self._terminal_log("✅", msg, "green")
        self._debug_log("info", "JOB", f"Completed {job_type}", {
            "story_id": story_id,
            "duration": duration
        })

    def job_failed(self, job_type: str, story_id: str, error: str):
        """Log when a job fails"""
        msg = f"Job failed: {job_type} (Story: {story_id[:8]}) - {error}"
        self._terminal_log("❌", msg, "red")
        self._debug_log("error", "JOB", f"Failed {job_type}", {
            "story_id": story_id,
            "error": error
        })

    def dialogue_output(self, story_id: str, speaker: str, message: str, phase: str = ""):
        """Log dialogue from narrator/DialogueAgent"""
        preview = message[:100] + "..." if len(message) > 100 else message
        msg = f"Narrator → User: \"{preview}\""
        if phase:
            msg += f" ({phase})"
        self._terminal_log("🗣️", msg, "blue")
        self._debug_log("info", "DIALOGUE", f"Output from {speaker}", {
            "story_id": story_id,
            "speaker": speaker,
            "message": message,
            "phase": phase,
            "length": len(message)
        })

    def event_emitted(self, event_type: str, story_id: str, data: dict):
        """Log when an event is emitted"""
        msg = f"Event: {event_type} (Story: {story_id[:8]})"
        self._terminal_log("📡", msg, "yellow")
        self._debug_log("info", "EVENT", f"Emitted {event_type}", {
            "story_id": story_id,
            "event_data": data
        })

    def websocket_connected(self, story_id: str, client_id: str = ""):
        """Log WebSocket connection"""
        msg = f"WebSocket connected (Story: {story_id[:8]})"
        self._terminal_log("🔌", msg, "green")
        self._debug_log("info", "WEBSOCKET", "Client connected", {
            "story_id": story_id,
            "client_id": client_id
        })

    def websocket_disconnected(self, story_id: str):
        """Log WebSocket disconnection"""
        msg = f"WebSocket disconnected (Story: {story_id[:8]})"
        self._terminal_log("🔌", msg, "yellow")
        self._debug_log("info", "WEBSOCKET", "Client disconnected", {
            "story_id": story_id
        })

    def agent_working(self, agent_name: str, task: str):
        """Log when an agent starts working"""
        msg = f"{agent_name} working: {task}"
        self._terminal_log("⚙️", msg)
        self._debug_log("info", "AGENT", f"{agent_name} started task", {
            "agent": agent_name,
            "task": task
        })

    def agent_completed(self, agent_name: str, task: str, duration: Optional[float] = None):
        """Log when an agent completes work"""
        msg = f"{agent_name} completed: {task}"
        if duration:
            msg += f" ({duration:.1f}s)"
        self._terminal_log("✓", msg, "green")
        self._debug_log("info", "AGENT", f"{agent_name} completed task", {
            "agent": agent_name,
            "task": task,
            "duration": duration
        })

    def api_request(self, method: str, endpoint: str, client_ip: str = ""):
        """Log API request"""
        msg = f"API {method} {endpoint}"
        if client_ip:
            msg += f" from {client_ip}"
        self._terminal_log("🌐", msg)
        self._debug_log("info", "API", f"{method} request", {
            "method": method,
            "endpoint": endpoint,
            "client_ip": client_ip
        })

    def error(self, component: str, message: str, error: Exception = None):
        """Log error"""
        msg = f"Error in {component}: {message}"
        if error:
            msg += f" ({type(error).__name__})"
        self._terminal_log("⚠️", msg, "red")
        self._debug_log("error", component, message, {
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None
        })

    def info(self, message: str):
        """Log general info"""
        self._terminal_log("ℹ️", message)
        self._debug_log("info", "SYSTEM", message)

    def warning(self, message: str):
        """Log warning"""
        self._terminal_log("⚠️", message, "yellow")
        self._debug_log("warning", "SYSTEM", message)

    def debug(self, component: str, message: str, data: Optional[dict] = None):
        """Log debug information (file only)"""
        if self.debug_mode:
            self._debug_log("debug", component, message, data)

    # ===== Debug Logging Methods =====

    def _write_json_log(self, log_file: Path, data: Dict[str, Any]):
        """Write structured JSON log entry"""
        try:
            with open(log_file, 'a') as f:
                json.dump(data, f)
                f.write('\n')
        except Exception as e:
            self.error("LOGGER", f"Failed to write JSON log: {e}")

    def _truncate_data(self, data: Any, max_length: int = 500) -> str:
        """Truncate data for display"""
        data_str = str(data)
        if len(data_str) > max_length:
            return data_str[:max_length] + f"... ({len(data_str)} chars total)"
        return data_str

    def agent_input(self, agent_name: str, task_description: str, context: Optional[Dict] = None, story_id: str = ""):
        """Log agent input (what the agent receives)"""
        # Skip if debug flag not enabled
        if not self.settings or not self.settings.debug_agent_io:
            return

        timestamp = datetime.now().isoformat()

        # Terminal output
        context_preview = ""
        if context:
            context_keys = list(context.keys())[:3]
            context_preview = f" (context: {', '.join(context_keys)})"
        msg = f"{agent_name} INPUT{context_preview}"
        self._terminal_log("📥", msg, "cyan")

        # Structured JSON log
        log_data = {
            "timestamp": timestamp,
            "type": "agent_input",
            "agent": agent_name,
            "story_id": story_id,
            "task_description": self._truncate_data(task_description),
            "context_keys": list(context.keys()) if context else [],
            "context_size": len(str(context)) if context else 0
        }

        if hasattr(self, 'agent_io_log'):
            self._write_json_log(self.agent_io_log, log_data)

    def agent_output(self, agent_name: str, output: Any, status: str = "success",
                    duration: Optional[float] = None, story_id: str = ""):
        """Log agent output (what the agent produced)"""
        # Skip if debug flag not enabled
        if not self.settings or not self.settings.debug_agent_io:
            return

        timestamp = datetime.now().isoformat()

        # Terminal output
        output_size = len(str(output)) if output else 0
        duration_str = f" in {duration:.1f}s" if duration else ""
        msg = f"{agent_name} OUTPUT: {status} ({output_size} chars){duration_str}"

        emoji = "✅" if status == "success" else "❌"
        color = "green" if status == "success" else "red"
        self._terminal_log(emoji, msg, color)

        # Structured JSON log
        log_data = {
            "timestamp": timestamp,
            "type": "agent_output",
            "agent": agent_name,
            "story_id": story_id,
            "status": status,
            "output_preview": self._truncate_data(output, 300),
            "output_size": output_size,
            "duration_seconds": duration
        }

        if hasattr(self, 'agent_io_log'):
            self._write_json_log(self.agent_io_log, log_data)

    def storage_operation(self, operation: str, path: str, data_summary: str,
                          size_bytes: int = 0, duration: Optional[float] = None):
        """Log storage write/update operations (Azure SQL or Firebase)"""
        # Skip if debug flag not enabled
        if not self.settings or not self.settings.debug_storage:
            return

        timestamp = datetime.now().isoformat()

        # Terminal output
        duration_str = f" in {duration*1000:.0f}ms" if duration else ""
        msg = f"Storage {operation.upper()} → {path} ({size_bytes} bytes){duration_str}"
        self._terminal_log("💾", msg, "yellow")

        # Structured JSON log
        log_data = {
            "timestamp": timestamp,
            "type": "storage_operation",
            "operation": operation,
            "path": path,
            "data_summary": data_summary,
            "size_bytes": size_bytes,
            "duration_seconds": duration
        }

        if hasattr(self, 'storage_log'):
            self._write_json_log(self.storage_log, log_data)

    def storage_read(self, path: str, result_summary: str, size_bytes: int = 0,
                     duration: Optional[float] = None):
        """Log storage read operations (Azure SQL or Firebase)"""
        # Skip if debug flag not enabled
        if not self.settings or not self.settings.debug_storage:
            return

        timestamp = datetime.now().isoformat()

        # Terminal output
        duration_str = f" in {duration*1000:.0f}ms" if duration else ""
        msg = f"Storage READ ← {path} ({size_bytes} bytes){duration_str}"
        self._terminal_log("📖", msg, "blue")

        # Structured JSON log
        log_data = {
            "timestamp": timestamp,
            "type": "storage_read",
            "path": path,
            "result_summary": result_summary,
            "size_bytes": size_bytes,
            "duration_seconds": duration
        }

        if hasattr(self, 'storage_log'):
            self._write_json_log(self.storage_log, log_data)

    # Backwards compatibility aliases
    def firebase_operation(self, *args, **kwargs):
        """Deprecated: Use storage_operation instead"""
        return self.storage_operation(*args, **kwargs)

    def firebase_read(self, *args, **kwargs):
        """Deprecated: Use storage_read instead"""
        return self.storage_read(*args, **kwargs)

    def llm_api_call(self, provider: str, model: str, prompt_tokens: int = 0,
                    completion_tokens: int = 0, latency: Optional[float] = None,
                    status: str = "success", cost_usd: Optional[float] = None):
        """Log LLM API call with token usage and cost"""
        # Skip if debug flag not enabled
        if not self.settings or not self.settings.debug_api_calls:
            return

        timestamp = datetime.now().isoformat()

        # Terminal output
        total_tokens = prompt_tokens + completion_tokens
        latency_str = f" in {latency:.1f}s" if latency else ""
        cost_str = f" (${cost_usd:.4f})" if cost_usd else ""
        msg = f"API {provider}/{model}: {total_tokens} tokens{latency_str}{cost_str}"

        emoji = "🤖" if status == "success" else "⚠️"
        color = "green" if status == "success" else "yellow"
        self._terminal_log(emoji, msg, color)

        # Structured JSON log
        log_data = {
            "timestamp": timestamp,
            "type": "llm_api_call",
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency,
            "status": status,
            "cost_usd": cost_usd
        }

        if hasattr(self, 'api_calls_log'):
            self._write_json_log(self.api_calls_log, log_data)


# Global logger instance
_logger: Optional[LoreLanternLogger] = None


def get_logger(settings=None) -> LoreLanternLogger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        # Check environment for debug mode
        import os
        debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        _logger = LoreLanternLogger(debug_mode=debug_mode, settings=settings)
    return _logger


def init_logger(debug_mode: bool = False, settings=None):
    """Initialize logger with specific debug mode and settings"""
    global _logger
    _logger = LoreLanternLogger(debug_mode=debug_mode, settings=settings)
    return _logger
