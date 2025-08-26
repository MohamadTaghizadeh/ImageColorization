import os
from dbutils import models
from dbutils.database import SessionLocal, engine
from config.config_handler import config


models.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Base:
    def __init__(self):
        if not os.path.exists(config["DB_DIR"]):
            os.makedirs(config["DB_DIR"])
