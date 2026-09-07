import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class EvaluationQuestionUtility:
    """Create and manage evaluation questions stored in SQLite."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path) if database_path else self._default_database_path()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    @staticmethod
    def _default_database_path() -> Path:
        return Path(__file__).resolve().parents[2] / "evaluation_questions.db"

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
            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evaluationQuestions'"
            ).fetchone()

            if table_exists:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(evaluationQuestions)")
                }
                if "Id" not in columns:
                    connection.execute(
                        "ALTER TABLE evaluationQuestions RENAME TO evaluationQuestions_legacy"
                    )
                    connection.execute(
                        """
                        CREATE TABLE evaluationQuestions (
                            Id INTEGER PRIMARY KEY AUTOINCREMENT,
                            Question TEXT NOT NULL,
                            Ground_truth TEXT NOT NULL
                        )
                        """
                    )
                    connection.execute(
                        """
                        INSERT INTO evaluationQuestions (Question, Ground_truth)
                        SELECT Question, Ground_truth FROM evaluationQuestions_legacy
                        """
                    )
                    connection.execute("DROP TABLE evaluationQuestions_legacy")
                return

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluationQuestions (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Question TEXT NOT NULL,
                    Ground_truth TEXT NOT NULL
                )
                """
            )

    def create(self, question: str, ground_truth: str) -> int:
        """Add an evaluation question and return its generated ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO evaluationQuestions (Question, Ground_truth) VALUES (?, ?)",
                (question, ground_truth),
            )
            question_id = cursor.lastrowid

        if question_id is None:
            raise RuntimeError("SQLite did not return an ID for the inserted question")
        return question_id

    def get(self, question_id: int) -> dict[str, int | str] | None:
        """Return one evaluation question by ID, or None when it does not exist."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT Id, Question, Ground_truth FROM evaluationQuestions WHERE Id = ?",
                (question_id,),
            ).fetchone()

        return dict(row) if row else None

    def get_all(self) -> list[dict[str, int | str]]:
        """Return all evaluation questions."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT Id, Question, Ground_truth FROM evaluationQuestions ORDER BY Id"
            ).fetchall()

        return [dict(row) for row in rows]

    def update(self, question_id: int, question: str, ground_truth: str) -> bool:
        """Update an evaluation question by ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE evaluationQuestions SET Question = ?, Ground_truth = ? WHERE Id = ?",
                (question, ground_truth, question_id),
            )

        return cursor.rowcount > 0

    def delete(self, question_id: int) -> bool:
        """Delete an evaluation question by ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM evaluationQuestions WHERE Id = ?",
                (question_id,),
            )

        return cursor.rowcount > 0