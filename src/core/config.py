from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import logging

logger = logging.getLogger("HackAIthon_Agent")

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
    max_new_tokens: int = 256
    num_runs: int = 3
    batch_size: int = 2


class AgentsConfig(BaseModel):
    router: AgentParams
    fast_qa: AgentParams
    reading: AgentParams
    coder: AgentParams
    correction: AgentParams
    fallback: AgentParams
    voting: VotingParams


class SandboxConfig(BaseModel):
    timeout_sec: int
    max_retries: int


class PathsConfig(BaseModel):
    data_dir: str
    output_dir: str
    checkpoint_file: str


class Settings(BaseSettings):
    app: AppConfig
    llm: LlmConfig
    agents: AgentsConfig
    sandbox: SandboxConfig
    paths: PathsConfig

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )


def load_settings() -> Settings:
    if not _SETTING_FILE.exists():
        raise FileNotFoundError(f"settings.yaml not found at {_SETTING_FILE}")

    with open(_SETTING_FILE, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(**raw)


try:
    settings = load_settings()
    logger.info("Settings loaded: model=%s, seq_len=%d", settings.llm.model_name, settings.llm.max_seq_length)
except Exception as e:
    logger.error("Failed to load settings: %s", e)
    raise
