from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from model_db.src.configurations import DBConfigurations

engine = create_engine(
    DBConfigurations.sql_alchemy_database_url,
    encoding="itf-8",
    pool_recycle=3600,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.rollback()
    finally:
        db.close()

@contextmanager
def get_context_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.rollback()
    finally:
        db.close()