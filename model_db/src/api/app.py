from logging import getLogger

from fastapi import FastAPI
from model_db.src.api import api, health
from model_db.src.configurations import APIConfigurations
from model_db.src import initialize
from model_db.src.db.database import engine

logger = getLogger(__name__)

initialize.initialize_table(engine=engine, checkfirst=True)

app = FastAPI(
    title=APIConfigurations.title,
    description=APIConfigurations.description,
    version=APIConfigurations.version,
)

app.include_router(health.router, prefix=f"/v{APIConfigurations.version}/health", tags=["health"])
app.include_router(api.router, prefix=f"/v{APIConfigurations.version}/api", tags=["api"])