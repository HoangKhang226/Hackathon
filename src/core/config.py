from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from src.utils.logger import logger

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SETTING_FILE = _PROJECT_ROOT / "config" / "settings.yaml"


class AppConfig(BaseModel):
    name: str
    environment: str
    debug: bool


class LlmConfig(BaseModel):
    model_name: str
    max_seq_length: int
    load_in_4bit: bool


class AgentParams(BaseModel):
    temperature: float
    max_new_tokens: int = 256
    batch_size: int = 10



class VotingParams(BaseModel):
    temperature: float
    num_runs: int


class AgentsConfig(BaseModel):
    router: AgentParams
    qa: AgentParams
    reading: AgentParams
    voting: VotingParams


class SandboxConfig(BaseModel):
    timeout_sec: int
    max_retries: int


class PipelineConfig(BaseModel):
    batch_size: int


class PathsConfig(BaseModel):
    data_dir: str
    output_dir: str
    checkpoint_file: str
    test_file: str


class Settings(BaseSettings):
    app: AppConfig
    llm: LlmConfig
    agents: AgentsConfig
    sandbox: SandboxConfig
    pipeline: PipelineConfig
    paths: PathsConfig

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )


def load_setting() -> Settings:
    if not _SETTING_FILE.exists():
        raise FileNotFoundError("settings.yaml not found")

    with open(_SETTING_FILE, "r", encoding="utf-8") as f:
        setting_config = yaml.safe_load(f)

    return Settings(**setting_config)


try:
    settings = load_setting()
    logger.info("Settings loaded successfully")
except Exception as e:
    logger.error(f"Error while loading settings: {e}")
    raise
