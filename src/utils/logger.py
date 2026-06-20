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

    with open(config_path, "r", encoding="utf-8") as f:
        logging_config = yaml.safe_load(f)

    # Động hóa đường dẫn file log để luôn trỏ vào thư mục project_root/output tuyệt đối
    if "handlers" in logging_config and "file" in logging_config["handlers"]:
        log_file_path = _LOGGING_DIR / "pipeline.log"
        _LOGGING_DIR.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"]["filename"] = str(log_file_path)

    logging.config.dictConfig(logging_config)
    logging.info("Read logging.yaml successfully")


# Initialize logging on module import
setup_logging()

logger = logging.getLogger("HackAIthon_Agent")

logging.info("Initialized logger successfully")
