from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password_hash = Column(String)

    role = Column(String)
    full_name = Column(String)
    grade = Column(String)
    school = Column(String)

    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)




class UserProgress(Base):
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))

    word_id = Column(Integer)
    level = Column(String)

    score = Column(Float)
    honesty_indicator = Column(Float)

    answers_json = Column(Text)
    completion_time = Column(DateTime, server_default=func.now())


class UserSkills(Base):
    __tablename__ = "user_skills"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))

    logic_score = Column(Float, default=0)
    analysis_score = Column(Float, default=0)
    ethics_score = Column(Float, default=0)
    knowledge_score = Column(Float, default=0)

    updated_at = Column(DateTime, server_default=func.now())


class Achievement(Base):
    __tablename__ = "achievements"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))

    achievement_type = Column(String)
    achievement_data = Column(Text)

    awarded_at = Column(DateTime, server_default=func.now())
