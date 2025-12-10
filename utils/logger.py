"""
Centralized logging configuration with color-coded console output.
Provides consistent logging across the entire RAG pipeline.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime


# ANSI color codes for terminal output
class LogColors:
    """ANSI color codes for different log levels."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Log level colors
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta
    
    # Component colors
    TIME = '\033[90m'       # Gray
    NAME = '\033[34m'       # Blue
    SEPARATOR = '\033[90m'  # Gray


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded output for console."""
    
    # Format templates
    FORMATS = {
        logging.DEBUG: (
            f"{LogColors.TIME}%(asctime)s{LogColors.RESET} "
            f"{LogColors.DEBUG}[DEBUG]{LogColors.RESET} "
            f"{LogColors.NAME}%(name)s{LogColors.RESET} "
            f"{LogColors.SEPARATOR}│{LogColors.RESET} "
            f"%(message)s"
        ),
        logging.INFO: (
            f"{LogColors.TIME}%(asctime)s{LogColors.RESET} "
            f"{LogColors.INFO}[INFO]{LogColors.RESET}  "
            f"{LogColors.NAME}%(name)s{LogColors.RESET} "
            f"{LogColors.SEPARATOR}│{LogColors.RESET} "
            f"%(message)s"
        ),
        logging.WARNING: (
            f"{LogColors.TIME}%(asctime)s{LogColors.RESET} "
            f"{LogColors.WARNING}[WARN]{LogColors.RESET}  "
            f"{LogColors.NAME}%(name)s{LogColors.RESET} "
            f"{LogColors.SEPARATOR}│{LogColors.RESET} "
            f"{LogColors.WARNING}%(message)s{LogColors.RESET}"
        ),
        logging.ERROR: (
            f"{LogColors.TIME}%(asctime)s{LogColors.RESET} "
            f"{LogColors.ERROR}[ERROR]{LogColors.RESET} "
            f"{LogColors.NAME}%(name)s{LogColors.RESET} "
            f"{LogColors.SEPARATOR}│{LogColors.RESET} "
            f"{LogColors.ERROR}%(message)s{LogColors.RESET}"
        ),
        logging.CRITICAL: (
            f"{LogColors.TIME}%(asctime)s{LogColors.RESET} "
            f"{LogColors.CRITICAL}{LogColors.BOLD}[CRITICAL]{LogColors.RESET} "
            f"{LogColors.NAME}%(name)s{LogColors.RESET} "
            f"{LogColors.SEPARATOR}│{LogColors.RESET} "
            f"{LogColors.CRITICAL}{LogColors.BOLD}%(message)s{LogColors.RESET}"
        ),
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    """Plain formatter without colors for file output."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logger(
    name: str = None,
    log_dir: Path = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up a logger with colored console output and file logging.
    
    Args:
        name: Logger name (defaults to root logger if None)
        log_dir: Directory for log files (defaults to PROJECT_ROOT/logs)
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler (rotating) without colors
    if log_to_file:
        if log_dir is None:
            # Default to logs directory in project root
            log_dir = Path(__file__).parent.parent / "logs"
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use date-based log file naming
        log_file = log_dir / f"rag_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(PlainFormatter())
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    This is a convenience function that ensures consistent logger setup
    across all modules in the project.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    # If root logger hasn't been set up, set it up now
    root = logging.getLogger()
    if not root.handlers:
        setup_logger()
    
    # Return a child logger
    return logging.getLogger(name)


# Optional: Set up root logger on import
# This ensures consistent formatting even if modules don't explicitly call setup_logger
setup_logger()