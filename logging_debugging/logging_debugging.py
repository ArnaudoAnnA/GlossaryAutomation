import csv
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from zoneinfo import ZoneInfo

_TIMEZONE = ZoneInfo("Europe/Rome")
LOG_DIRECTORY = Path(__file__).resolve().parent / ".." / "logs"
_LOG_FILENAME_TEMPLATE = "log_{timestamp}[{session_id}].csv"
_CSV_HEADERS = ["timestamp", "log_type", "file", "message"]

# Queue contains log rows as lists of strings, or None for shutdown
_thread_safe_queues: dict[str, Queue[list[str] | None]] = {}


def _write_logs(session_id: str):
    """Background worker that writes queued log rows to a CSV file."""
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(_TIMEZONE).strftime("%Y%m%d-%H%M%S")
    filename = _LOG_FILENAME_TEMPLATE.format(session_id=session_id, timestamp=timestamp)
    log_file_path = LOG_DIRECTORY / filename

    with open(log_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(_CSV_HEADERS)
        f.flush()

        while True:
            row = _thread_safe_queues[session_id].get()

            if row is None:
                _thread_safe_queues[session_id].task_done()
                break

            writer.writerow(row)
            f.flush()
            _thread_safe_queues[session_id].task_done()


class LogType(Enum):
    ERROR = "ERROR"
    API_GET = "API_GET"
    API_DELETE = "API_DELETE"
    API_PUT = "API_PUT"
    API_POST = "API_POST"
    LOGIN = "LOGIN"
    USER_INPUT = "USER_INPUT"
    LLM_OUTPUT = "LLM_OUTPUT"
    STATE_MACHINE_STATE = "STATE_MACHINE_STATE"


GLOBAL_LOG = "Global_events"


def log_thread_safe(session_id: str | None, current_file: str, log_type: LogType, message: str):
    """Safely queues structured log fields for CSV writing."""
    if not session_id:
        session_id = GLOBAL_LOG

    if session_id not in _thread_safe_queues:
        _thread_safe_queues[session_id] = Queue()
        threading.Thread(target=_write_logs, args=(session_id,), daemon=True).start()

    timestamp_str = datetime.now(_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp_str, log_type.value, current_file, message]

    _thread_safe_queues[session_id].put(row)


def shutdown_logger(session_id: str):
    """Signals the worker thread to finish writing and close the CSV."""
    if session_id in _thread_safe_queues:
        _thread_safe_queues[session_id].put(None)