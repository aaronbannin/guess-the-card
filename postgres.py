# from datetime import datetime
# import uuid

# from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON
# from sqlalchemy.engine import URL
# from sqlalchemy.dialects.postgresql import UUID
# from sqlalchemy.orm import DeclarativeBase

# from config import Config
# from models import Role


# url = URL.create(
#     drivername="postgresql",
#     username=Config.POSTGRES_USER,
#     host=Config.POSTGRES_HOST,
#     port=5432,
#     database=Config.POSTGRES_DB
# )

# postgres_engine = create_engine(url)


# class Base(DeclarativeBase):
#     pass

# class ChatLogs(Base):
#     __tablename__ = 'chat_logs'

#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     run_id = Column(UUID(as_uuid=True), nullable=False)
#     run_started_at = Column(DateTime())
#     role = Column(String(100), nullable=False)
#     llm = Column(JSON)
#     response = Column(Text, nullable=False)
#     context = Column(Text)
#     created_at = Column(DateTime(), default=datetime.now)
#     updated_at = Column(DateTime(), default=datetime.now, onupdate=datetime.now)

#     def __str__(self) -> str:
#         role = Role[self.role]
#         return f"{self.role.pretty} {str(self.response).strip()}"


# Base.metadata.create_all(postgres_engine)
