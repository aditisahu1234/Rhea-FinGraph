from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Rhea FinGraph"
    environment: str = "local"
    model_version: str = "untrained"
    model_dir: Path = Path("artifacts/models/baseline")

    postgres_url: str = "postgresql://fingraph:change-me-local-only@localhost:5432/fingraph"
    redis_url: str = "redis://localhost:6379/0"
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    # Accept either the FINGRAPH_-prefixed name or the bare Compose-style name
    # (the shared .env uses the latter, so a live local Neo4j is reachable).
    neo4j_password: str = Field(
        default="change-me-local-only",
        validation_alias=AliasChoices("FINGRAPH_NEO4J_PASSWORD", "NEO4J_PASSWORD"),
    )
    elasticsearch_url: str = "http://localhost:9200"
    helix_url: str = "http://localhost:7842"

    # Helix v2 — durable episodic memory + healing artifacts (Layer 5).
    healing_dir: Path = Path("artifacts/healing")
    # Score 5 demo events + 3 feedback outcomes at API startup so the
    # dashboard shows live audit/streaming/healing data on first load.
    demo_seed: bool = True

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
