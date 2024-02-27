import logging
from typing import ClassVar, List, Optional, Type

from asyncpg.exceptions import UniqueViolationError
from databases import Database
from pydantic import BaseModel, TypeAdapter

from shared.resources import DatabaseCredentials

logger = logging.getLogger("databases")


class Entity(BaseModel):
    _table_name: ClassVar[Optional[str]] = None
    _pk: ClassVar[Optional[str]] = None

    pass


class AbstractRepository:
    def __init__(self, db: Database, entity: Type):
        self._db = db
        self._entity = entity
        self._table_name = entity._table_name

    def _get_query_parameters(self, dump):
        keys = list(dump.keys())
        columns = ",".join(keys)
        placeholders = ",".join(map(lambda x: f":{x}", keys))
        return columns, placeholders

    async def add(self, entities, ignore_conflict=False):
        if not isinstance(entities, list):
            entities = [entities]

        if entities == []:
            return

        dumps = list(map(lambda x: x.model_dump(), entities))

        columns, placeholders = self._get_query_parameters(dumps[0])

        query = f"INSERT INTO {self._table_name}({columns}) VALUES ({placeholders})"
        logger.debug(f"Executing query: {query}")

        if ignore_conflict:
            query += " ON CONFLICT DO NOTHING"

        await self._db.execute_many(query=query, values=dumps)

    async def update(self, entity, fields: List[str]):
        dump = entity.model_dump()
        query_set = [f"{field} = :{field}" for field in fields]

        pk = entity._pk
        if not isinstance(pk, tuple):
            pk = (pk,)

        where_clause = " AND ".join(f"{key} = :{key}" for key in pk)
        query = f"UPDATE {self._table_name} SET {','.join(query_set)} WHERE {where_clause}"
        logger.debug(f"Executing query: {query}")
        await self._db.execute(
            query=query,
            values={k: dump[k] for k in fields}
            | {key: dump[key] for key in pk},
        )

    async def get(self, field=None, value=None) -> List:
        query = f"SELECT * FROM {self._table_name}"
        logger.debug(f"Executing query: {query}")
        if field is not None:
            query += f" WHERE {field} = :{field}"
            rows = await self._db.fetch_all(query=query, values={field: value})
        else:
            rows = await self._db.fetch_all(query=query)

        mapped = map(
            lambda row: TypeAdapter(self._entity).validate_python(
                dict(row._mapping)
            ),
            rows,
        )

        return list(mapped)


class PgRepository(AbstractRepository):
    async def add_or_update(self, entity: Entity, fields: List[str]):
        try:
            await self.add(entity)
        except UniqueViolationError:
            await self.update(entity, fields)


def create_db_string(creds: DatabaseCredentials):
    return f"{creds.driver}://{creds.username}:{creds.password}@{creds.url}:{creds.port}/{creds.db_name}"
