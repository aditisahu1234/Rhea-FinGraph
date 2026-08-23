from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Rhea FinGraph"
    environment: str = "local"
    model_version: str = "untrained"

    postgres_url: str = "postgresql://fingraph:change-me-local-only@localhost:5432/fingraph"
    redis_url: str = "redis://localhost:6379/0"
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "change-me-local-only"
    elasticsearch_url: str = "http://localhost:9200"
    helix_url: str = "http://localhost:7842"

    # `.env` is shared with Docker Compose, which has additional POSTGRES_ and port
    # variables. The API reads only its FINGRAPH_ settings and safely ignores the rest.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FINGRAPH_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
