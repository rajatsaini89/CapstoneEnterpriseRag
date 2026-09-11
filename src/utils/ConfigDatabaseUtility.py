import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ConfigDatabaseUtility:
    """Create and manage application configuration stored in SQLite."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = (
            Path(database_path)
            if database_path
            else Path(__file__).resolve().parents[2] / "evaluation_questions.db"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def migrate_from_json(self, config_path: str | Path, keys: tuple[str, ...]) -> None:
        """Copy missing configuration entries from the legacy JSON file."""
        path = Path(config_path)
        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        for key in keys:
            if self.get(key) is None and key in config:
                self.create(key, config[key])

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _create_table(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS applicationConfig (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ConfigKey TEXT NOT NULL UNIQUE,
                    ConfigValue TEXT NOT NULL
                )
                """
            )

    def create(self, key: str, value: Any) -> int:
        """Create a configuration entry and return its generated ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO applicationConfig (ConfigKey, ConfigValue) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            entry_id = cursor.lastrowid

        if entry_id is None:
            raise RuntimeError("SQLite did not return an ID for the inserted configuration")
        return entry_id

    def get(self, key: str) -> Any | None:
        """Return one configuration value by key, or None when it does not exist."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT ConfigValue FROM applicationConfig WHERE ConfigKey = ?",
                (key,),
            ).fetchone()

        return json.loads(row["ConfigValue"]) if row else None

    def get_all(self) -> dict[str, Any]:
        """Return all configuration values keyed by configuration name."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT ConfigKey, ConfigValue FROM applicationConfig ORDER BY Id"
            ).fetchall()

        return {row["ConfigKey"]: json.loads(row["ConfigValue"]) for row in rows}

    def update(self, key: str, value: Any) -> bool:
        """Update a configuration value by key."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE applicationConfig SET ConfigValue = ? WHERE ConfigKey = ?",
                (json.dumps(value), key),
            )

        return cursor.rowcount > 0

    def delete(self, key: str) -> bool:
        """Delete a configuration value by key."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM applicationConfig WHERE ConfigKey = ?",
                (key,),
            )

        return cursor.rowcount > 0