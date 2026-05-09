import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from seshat.utils.db import ensure_psycopg_scheme

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_online() -> None:
    url = ensure_psycopg_scheme(
        os.environ["DATABASE_URL"],
        warn_msg="Unexpected driver %r in DATABASE_URL; replacing with '+psycopg' for Alembic migrations",
    )
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
