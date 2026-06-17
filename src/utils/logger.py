import logging
import logging.config
from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOG_FILE = _PROJECT_ROOT / "config" / "logging.yaml"
_LOGGING_DIR = _PROJECT_ROOT / "output"


def setup_logging(config_path: Path = _LOG_FILE) -> None:
    """Configure logging from a YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError("logging.yaml config file not found")

    # Đảm bảo thư mục output tồn tại trước khi ghi log
    if not _LOGGING_DIR.exists():
        _LOGGING_DIR.mkdir(exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        logging_config = yaml.safe_load(f)

    logging.config.dictConfig(logging_config)
    logging.info("Read logging.yaml successfully")


# Initialize logging on module import
setup_logging()

logger = logging.getLogger("HackAIthon_Agent")

logging.info("Initialized logger successfully")
