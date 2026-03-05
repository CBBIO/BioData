"""Helpers to access the BioData PostgreSQL database with pgvector support.

This module is tailored to the BioData schema described in ``Info.txt``.
It focuses on embedding search and GO annotation retrieval.
"""

from __future__ import annotations

import os
from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, Literal, cast


Params = Union[Sequence[Any], Mapping[str, Any], None]
DistanceMetric = Literal["l2", "cosine", "inner_product"]
ConfigDict = Dict[str, Any]


class BioDataError(Exception):
    """Base exception for ``CBBIO.BioData`` errors."""


class DriverDependencyError(BioDataError):
    """Raised when required dependencies are missing."""


class ConnectionNotOpenError(BioDataError):
    """Raised when an operation needs an open connection."""


class NotFoundError(BioDataError):
    """Raised when an expected BioData record is not found."""


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
REQUIRED_TABLES: Tuple[str, ...] = (
    "protein",
    "sequence",
    "sequence_embeddings",
    "sequence_embedding_type",
    "protein_go_term_annotation",
    "go_terms",
)


def _default_config_dict() -> ConfigDict:
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "BioData",
            "user": "usuario",
            "password": "clave",
        },
        "client": {
            "autocommit": True,
            "register_halfvec": True,
        },
        "search": {
            "default_metric": "l2",
            "default_k": 10,
        },
    }


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> ConfigDict:
    merged: ConfigDict = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, MappingABC) and isinstance(value, MappingABC):
            merged[key] = _merge_dicts(
                _to_config_dict(cast(object, existing)),
                _to_config_dict(cast(object, value)),
            )
        else:
            merged[key] = value
    return merged


def _to_config_dict(value: object) -> ConfigDict:
    if not isinstance(value, MappingABC):
        return {}
    mapping = cast(Mapping[Any, Any], value)
    return {str(key): item for key, item in mapping.items()}


def _parse_bool(value: str, *, env_name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise BioDataError(f"Invalid boolean value for {env_name}: {value!r}")


def _apply_env_overrides(config: Mapping[str, Any]) -> ConfigDict:
    db = _to_config_dict(config.get("database"))
    client = _to_config_dict(config.get("client"))
    search = _to_config_dict(config.get("search"))

    if os.getenv("BIODATA_DB_HOST") is not None:
        db["host"] = os.getenv("BIODATA_DB_HOST")
    if os.getenv("BIODATA_DB_PORT") is not None:
        db["port"] = int(os.getenv("BIODATA_DB_PORT", "5432"))
    if os.getenv("BIODATA_DB_NAME") is not None:
        db["name"] = os.getenv("BIODATA_DB_NAME")
    if os.getenv("BIODATA_DB_USER") is not None:
        db["user"] = os.getenv("BIODATA_DB_USER")
    if os.getenv("BIODATA_DB_PASSWORD") is not None:
        db["password"] = os.getenv("BIODATA_DB_PASSWORD")

    if os.getenv("BIODATA_AUTOCOMMIT") is not None:
        client["autocommit"] = _parse_bool(os.getenv("BIODATA_AUTOCOMMIT", ""), env_name="BIODATA_AUTOCOMMIT")
    if os.getenv("BIODATA_REGISTER_HALFVEC") is not None:
        client["register_halfvec"] = _parse_bool(
            os.getenv("BIODATA_REGISTER_HALFVEC", ""), env_name="BIODATA_REGISTER_HALFVEC"
        )

    if os.getenv("BIODATA_DEFAULT_METRIC") is not None:
        search["default_metric"] = os.getenv("BIODATA_DEFAULT_METRIC")
    if os.getenv("BIODATA_DEFAULT_K") is not None:
        search["default_k"] = int(os.getenv("BIODATA_DEFAULT_K", "10"))

    return {
        "database": db,
        "client": client,
        "search": search,
    }


def load_config(config_path: Optional[Union[str, Path]] = None, *, strict: bool = False) -> ConfigDict:
    """Load config YAML, merged on top of built-in defaults.

    When ``strict`` is ``False`` and ``pyyaml`` is unavailable, defaults are returned.
    Environment variables with prefix ``BIODATA_`` override YAML values.
    """
    default_cfg = _default_config_dict()
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    loaded_cfg: ConfigDict = {}

    if path.exists():
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            if strict:
                raise DriverDependencyError("Missing dependency 'pyyaml'. Install with: pip install pyyaml") from exc
            loaded_cfg = {}
        else:
            with path.open("r", encoding="utf-8") as handle:
                parsed_obj: object = yaml.safe_load(handle)
            if parsed_obj is None:
                parsed_obj = {}
            if not isinstance(parsed_obj, dict):
                raise BioDataError(f"Invalid config format in {path}. Root must be a YAML mapping.")
            loaded_cfg = _to_config_dict(cast(object, parsed_obj))

    merged = _merge_dicts(default_cfg, loaded_cfg)
    merged = _merge_dicts(merged, _apply_env_overrides(merged))
    return merged


def _config_defaults(config: Mapping[str, Any]) -> ConfigDict:
    db_config = _to_config_dict(config.get("database"))
    client_config = _to_config_dict(config.get("client"))
    search_config = _to_config_dict(config.get("search"))

    host = str(db_config.get("host", "localhost"))
    port = int(db_config.get("port", 5432))
    database = str(db_config.get("name", "BioData"))
    user = str(db_config.get("user", "usuario"))
    password = str(db_config.get("password", "clave"))

    metric = str(search_config.get("default_metric", "l2")).strip().lower()
    if metric not in {"l2", "cosine", "inner_product"}:
        raise BioDataError(f"Invalid search.default_metric: {metric!r}")

    return {
        "host": host,
        "port": port,
        "database": database,
        "user": user,
        "password": password,
        "dsn": f"postgresql://{user}:{password}@{host}:{port}/{database}",
        "autocommit": bool(client_config.get("autocommit", True)),
        "register_halfvec": bool(client_config.get("register_halfvec", True)),
        "default_metric": metric,
        "default_k": int(search_config.get("default_k", 10)),
    }


_CONFIG = load_config()
_DEFAULTS = _config_defaults(_CONFIG)

DEFAULT_HOST = _DEFAULTS["host"]
DEFAULT_PORT = _DEFAULTS["port"]
DEFAULT_DATABASE = _DEFAULTS["database"]
DEFAULT_USER = _DEFAULTS["user"]
DEFAULT_PASSWORD = _DEFAULTS["password"]
DEFAULT_DSN = _DEFAULTS["dsn"]
DEFAULT_AUTOCOMMIT = _DEFAULTS["autocommit"]
DEFAULT_REGISTER_HALFVEC = _DEFAULTS["register_halfvec"]
DEFAULT_SEARCH_METRIC: DistanceMetric = _DEFAULTS["default_metric"]
DEFAULT_SEARCH_K = _DEFAULTS["default_k"]


@dataclass(frozen=True)
class EmbeddingType:
    id: int
    name: str
    model_name: Optional[str]
    task_name: Optional[str]
    description: Optional[str]


@dataclass(frozen=True)
class Neighbor:
    protein_id: str
    layer_index: int
    distance: float


@dataclass(frozen=True)
class GOAnnotation:
    go_id: str
    category: str
    description: str
    evidence_code: str


class BioDataClient:
    """Client for BioData PostgreSQL + pgvector operations."""

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        autocommit: Optional[bool] = None,
        register_halfvec: Optional[bool] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        config = load_config(config_path)
        defaults = _config_defaults(config)

        self.dsn = dsn or defaults["dsn"]
        self.autocommit = defaults["autocommit"] if autocommit is None else autocommit
        self.register_halfvec = defaults["register_halfvec"] if register_halfvec is None else register_halfvec
        self.default_metric: DistanceMetric = defaults["default_metric"]
        self.default_k: int = defaults["default_k"]
        self._conn: Any = None

    def __enter__(self) -> "BioDataClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    def connect(self) -> None:
        """Open a psycopg connection and register pgvector halfvec support."""
        if self._conn is not None:
            return

        try:
            import psycopg  # type: ignore
        except ModuleNotFoundError as exc:
            raise DriverDependencyError("Missing dependency 'psycopg'. Install with: pip install psycopg[binary]") from exc

        psycopg_module = cast(Any, psycopg)
        conn: Any = psycopg_module.connect(self.dsn, autocommit=self.autocommit)

        if self.register_halfvec:
            try:
                from pgvector.psycopg import register_vector  # type: ignore
            except ModuleNotFoundError as exc:
                raise DriverDependencyError("Missing dependency 'pgvector'. Install with: pip install pgvector") from exc
            register_vector_fn = cast(Callable[..., Any], register_vector)
            register_vector_fn(conn, "halfvec")

        self._conn = conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Transactional context manager."""
        conn = self._require_connection()
        try:
            yield
            if not self.autocommit:
                conn.commit()
        except Exception:
            if not self.autocommit:
                conn.rollback()
            raise

    def query_all(self, sql: str, params: Params = None) -> List[Dict[str, Any]]:
        """Execute a query and return rows as dictionaries."""
        conn = self._require_connection()
        with _cursor(conn) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            return [_row_to_dict(row, cur) for row in rows]

    def query_one(self, sql: str, params: Params = None) -> Optional[Dict[str, Any]]:
        """Execute a query and return one row as a dictionary."""
        conn = self._require_connection()
        with _cursor(conn) as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_dict(row, cur)

    def scalar(self, sql: str, params: Params = None) -> Any:
        """Return first column of first row for a query."""
        row = self.query_one(sql, params)
        if row is None:
            return None
        return next(iter(row.values()))

    def health_check(
        self,
        *,
        check_extension: bool = True,
        check_required_tables: bool = True,
    ) -> Dict[str, Any]:
        """Return connectivity and schema health details."""
        status: Dict[str, Any] = {
            "connected": self.is_connected,
            "database": self.scalar("SELECT current_database();"),
            "server_version": self.scalar("SHOW server_version;"),
        }

        if check_extension:
            has_vector = self.scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            status["pgvector_installed"] = bool(has_vector)

        if check_required_tables:
            rows = self.query_all(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE';
                """
            )
            existing = {str(row["table_name"]) for row in rows}
            missing = [table for table in REQUIRED_TABLES if table not in existing]
            status["missing_tables"] = missing

        return status

    def count_sequence_embeddings(self) -> int:
        """Return total rows in ``sequence_embeddings``."""
        value = self.scalar("SELECT COUNT(*) FROM sequence_embeddings;")
        return int(value or 0)

    def list_embedding_types(self) -> List[EmbeddingType]:
        """Return known sequence embedding types from ``sequence_embedding_type``."""
        rows = self.query_all(
            """
            SELECT id, name, model_name, task_name, description
            FROM sequence_embedding_type
            ORDER BY id;
            """
        )
        return [
            EmbeddingType(
                id=int(row["id"]),
                name=str(row["name"]),
                model_name=_as_optional_str(row.get("model_name")),
                task_name=_as_optional_str(row.get("task_name")),
                description=_as_optional_str(row.get("description")),
            )
            for row in rows
        ]

    def get_embedding_type_by_name(self, name: str) -> Optional[EmbeddingType]:
        """Get embedding type metadata by exact ``sequence_embedding_type.name``."""
        row = self.query_one(
            """
            SELECT id, name, model_name, task_name, description
            FROM sequence_embedding_type
            WHERE name = %s
            LIMIT 1;
            """,
            (name,),
        )
        if row is None:
            return None
        return EmbeddingType(
            id=int(row["id"]),
            name=str(row["name"]),
            model_name=_as_optional_str(row.get("model_name")),
            task_name=_as_optional_str(row.get("task_name")),
            description=_as_optional_str(row.get("description")),
        )

    def list_available_layers(self, embedding_type_id: int) -> List[int]:
        """List available layer indices for an embedding type."""
        rows = self.query_all(
            """
            SELECT DISTINCT layer_index
            FROM sequence_embeddings
            WHERE embedding_type_id = %s
            ORDER BY layer_index;
            """,
            (embedding_type_id,),
        )
        return [int(row["layer_index"]) for row in rows]

    def get_protein_embedding(
        self,
        uniprot_id: str,
        embedding_type_id: int,
        layer_index: int = 0,
        *,
        as_numpy: bool = False,
    ) -> Any:
        """Fetch one protein embedding by UniProt ID, embedding type, and layer.

        Returns:
        - ``None`` if not found.
        - Python sequence by default.
        - NumPy ``float32`` array when ``as_numpy=True``.
        """
        conn = self._require_connection()
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT se.embedding
                FROM protein p
                JOIN sequence s ON p.sequence_id = s.id
                JOIN sequence_embeddings se ON se.sequence_id = s.id
                WHERE p.id = %s
                  AND se.embedding_type_id = %s
                  AND se.layer_index = %s
                LIMIT 1;
                """,
                (uniprot_id, embedding_type_id, layer_index),
            )
            row = cur.fetchone()

        if row is None:
            return None

        vector = row[0]
        if not as_numpy:
            return vector

        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise DriverDependencyError("NumPy is required for as_numpy=True. Install with: pip install numpy") from exc

        return np.array(vector, dtype=np.float32)

    def find_nearest_neighbors(
        self,
        query_embedding: Any,
        embedding_type_id: int,
        layer_index: int = 0,
        k: Optional[int] = None,
        *,
        metric: Optional[DistanceMetric] = None,
        exclude_protein_ids: Optional[Sequence[str]] = None,
    ) -> List[Neighbor]:
        """Find nearest proteins using pgvector distance operators.

        Metrics:
        - ``l2``: ``<->``
        - ``cosine``: ``<=>``
        - ``inner_product``: ``<#>``
        """
        conn = self._require_connection()
        effective_metric = metric or self.default_metric
        operator = _metric_operator(effective_metric)
        effective_k = self.default_k if k is None else int(k)
        if effective_k < 1:
            raise BioDataError("k must be >= 1")

        excluded_ids = [str(value) for value in (exclude_protein_ids or [])]
        extra_where = ""
        params: List[Any] = [query_embedding, embedding_type_id, layer_index]
        if excluded_ids:
            extra_where = " AND p.id <> ALL(%s)"
            params.append(excluded_ids)

        sql = (
            "SELECT p.id AS protein_id, "
            "       se.layer_index, "
            f"       se.embedding {operator} %s::halfvec AS distance "
            "FROM sequence_embeddings se "
            "JOIN sequence s ON se.sequence_id = s.id "
            "JOIN protein p ON p.sequence_id = s.id "
            "WHERE se.embedding_type_id = %s "
            "  AND se.layer_index = %s"
            f"{extra_where} "
            f"ORDER BY se.embedding {operator} %s::halfvec "
            "LIMIT %s;"
        )

        params.extend([query_embedding, effective_k])
        with _cursor(conn) as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        neighbors: List[Neighbor] = []
        for protein_id, row_layer, distance in rows:
            neighbors.append(Neighbor(protein_id=str(protein_id), layer_index=int(row_layer), distance=float(distance)))
        return neighbors

    def fetch_go_annotations(self, protein_ids: Sequence[str]) -> Dict[str, List[GOAnnotation]]:
        """Fetch GO annotations grouped by protein ID."""
        ids = [str(p) for p in protein_ids]
        if not ids:
            return {}

        conn = self._require_connection()
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT pga.protein_id,
                       pga.go_id,
                       gt.category,
                       gt.description,
                       pga.evidence_code
                FROM protein_go_term_annotation pga
                JOIN go_terms gt ON gt.go_id = pga.go_id
                WHERE pga.protein_id = ANY(%s)
                ORDER BY pga.protein_id, pga.go_id;
                """,
                (ids,),
            )
            rows = cur.fetchall()

        grouped: Dict[str, List[GOAnnotation]] = {}
        for protein_id, go_id, category, description, evidence_code in rows:
            grouped.setdefault(str(protein_id), []).append(
                GOAnnotation(
                    go_id=str(go_id),
                    category=str(category),
                    description=str(description),
                    evidence_code=str(evidence_code),
                )
            )
        return grouped

    def neighbors_with_go(
        self,
        query_uniprot_id: str,
        embedding_type_id: int,
        layer_index: int = 0,
        k: Optional[int] = None,
        *,
        metric: Optional[DistanceMetric] = None,
        include_query: bool = False,
    ) -> Tuple[List[Neighbor], Dict[str, List[GOAnnotation]]]:
        """End-to-end helper: query embedding -> neighbors -> GO annotations."""
        query_embedding = self.get_protein_embedding(
            query_uniprot_id,
            embedding_type_id,
            layer_index,
            as_numpy=False,
        )
        if query_embedding is None:
            raise NotFoundError(
                f"No embedding found for {query_uniprot_id} (type={embedding_type_id}, layer={layer_index})."
            )

        neighbors = self.find_nearest_neighbors(
            query_embedding,
            embedding_type_id,
            layer_index=layer_index,
            k=k,
            metric=metric,
            exclude_protein_ids=[] if include_query else [query_uniprot_id],
        )

        annotations = self.fetch_go_annotations([n.protein_id for n in neighbors])
        return neighbors, annotations

    def _require_connection(self) -> Any:
        if self._conn is None:
            raise ConnectionNotOpenError("Connection is not open. Call connect() or use 'with BioDataClient(...)'.")
        return self._conn


# Public helpers

def build_dsn(
    user: str = DEFAULT_USER,
    password: str = DEFAULT_PASSWORD,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    database: str = DEFAULT_DATABASE,
) -> str:
    """Build a PostgreSQL DSN for BioData."""
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def connect(
    dsn: Optional[str] = None,
    *,
    autocommit: Optional[bool] = None,
    register_halfvec: Optional[bool] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> BioDataClient:
    """Create and connect a ``BioDataClient``."""
    client = BioDataClient(
        dsn=dsn,
        autocommit=autocommit,
        register_halfvec=register_halfvec,
        config_path=config_path,
    )
    client.connect()
    return client


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _metric_operator(metric: DistanceMetric) -> str:
    mapping = {
        "l2": "<->",
        "cosine": "<=>",
        "inner_product": "<#>",
    }
    key = str(metric).strip().lower()
    operator = mapping.get(key)
    if operator is None:
        raise BioDataError(f"Unsupported metric: {metric!r}. Use one of: l2, cosine, inner_product.")
    return operator


def _row_to_dict(row: Any, cursor: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        row_map = cast(Mapping[Any, Any], row)
        return {str(key): value for key, value in row_map.items()}

    if hasattr(row, "keys"):
        keys = cast(Sequence[Any], row.keys())
        return {str(key): row[key] for key in keys}

    if getattr(cursor, "description", None):
        description = cast(Sequence[Sequence[Any]], cursor.description)
        columns: List[str] = [str(col[0]) for col in description]
        values = cast(Sequence[Any], row)
        return {str(column): value for column, value in zip(columns, values)}

    return {"value": row}


@contextmanager
def _cursor(conn: Any) -> Iterator[Any]:
    cur = conn.cursor()
    try:
        yield cur
    finally:
        close = getattr(cur, "close", None)
        if callable(close):
            close()


__all__ = [
    "BioDataClient",
    "BioDataError",
    "ConnectionNotOpenError",
    "DriverDependencyError",
    "EmbeddingType",
    "GOAnnotation",
    "Neighbor",
    "NotFoundError",
    "DistanceMetric",
    "DEFAULT_DATABASE",
    "DEFAULT_DSN",
    "DEFAULT_HOST",
    "DEFAULT_PASSWORD",
    "DEFAULT_PORT",
    "DEFAULT_USER",
    "DEFAULT_AUTOCOMMIT",
    "DEFAULT_REGISTER_HALFVEC",
    "DEFAULT_SEARCH_METRIC",
    "DEFAULT_SEARCH_K",
    "DEFAULT_CONFIG_PATH",
    "REQUIRED_TABLES",
    "build_dsn",
    "connect",
    "load_config",
]
