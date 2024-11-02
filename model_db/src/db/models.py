from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from model_db.src.db.database import Base

class Project(Base):
    __tablename__ = "projects"

    project_id = Column(
        String(255),
        primary_key = True,
        comment = "Project ID",
    )
    project_name = Column(
        String(255),
        nullable = False,
        unique = True,
        comment = "Project Name",
    )
    description = Column(
        Text,
        nullable=True,
        comment = "Project Description",
    )
    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False,
    )

class Model(Base):
    __tablename__ = "models"

    model_id = Column(
        String(255),
        primary_key=True,
        comment="Model ID",
    )
    project_id = Column(
        String(255),
        ForeignKey("projects.project_id"),
        nullable=False,
        comment="Project ID",
    )
    model_name = Column(
        String(255),
        nullable=False,
        comment="Model Name",
    )
    description = Column(
        Text,
        nullable=True,
        comment="Model Description",
    )
    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False,
    )

class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(
        String(255),
        primary_key=True,
        comment="Experiment ID",
    )
    model_id = Column(
        String(255),
        ForeignKey("models.model_id"),
        nullable=False,
        comment="Model ID",
    )
    model_version_id = Column(
        String(255),
        nullable=False,
        comment="Model Version ID",
    )
    parameters = Column(
        JSON,
        nullable=True,
        comment="Parameters",
    )
    training_dataset = Column(
        Text,
        nullable=True,
        comment="Training Dataset",
    )
    validation_dataset = Column(
        Text,
        nullable=True,
        comment="Validation Dataset",
    )
    test_dataset = Column(
        Text,
        nullable=True,
        comment="Test Dataset",
    )
    evaluations = Column(
        Text,
        nullable=True,
        comment="Evaluations",
    )
    artifact_file_paths = Column(
        JSON,
        nullable=True,
        comment="Artifact File Paths",
    )
    created_datetime = Column(
        DateTime(timezone=True),
        server_default=current_timestamp(),
        nullable=False,
    )