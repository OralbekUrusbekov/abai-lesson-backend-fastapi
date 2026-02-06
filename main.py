import sys
import locale
from data import ABAI_WORDS, EXERCISE_TEMPLATES, ASSESSMENT_CRITERIA, DEVELOPED_SKILLS
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None

import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
import json
import uuid
from datetime import datetime, date
from dotenv import load_dotenv
import base64
import asyncio
import tempfile
import speech_recognition as sr
import io
import re
import unicodedata
from gtts import gTTS
import asyncpg
from contextlib import asynccontextmanager
import logging
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ========== КОНФИГУРАЦИЯ POSTGRESQL ==========
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "dpg-d62tn10nputs73enl9ug-a.oregon-postgres.render.com"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "abai_db"),
    "user": os.getenv("DB_USER", "oralbek"),
    "password": os.getenv("DB_PASSWORD", "HkaWkf28DTyx8aCIRz0Pkxd0XxdpBbQL"),
}


# Создаем пул соединений
connection_pool = None


def create_connection_pool():
    """Создание пула соединений с PostgreSQL"""
    global connection_pool
    try:
        connection_pool = SimpleConnectionPool(
            1, 20,
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            database=DATABASE_CONFIG["database"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"]
        )
        logger.info("Пул соединений PostgreSQL создан успешно")
    except Exception as e:
        logger.error(f"Ошибка создания пула соединений: {e}")
        raise


def get_db_connection():
    """Получение соединения из пула"""
    if connection_pool is None:
        create_connection_pool()
    return connection_pool.getconn()


def release_db_connection(conn):
    """Возвращение соединения в пул"""
    if connection_pool:
        connection_pool.putconn(conn)


def init_database():
    """Инициализация базы данных PostgreSQL"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Создание таблицы users
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role VARCHAR(20) NOT NULL CHECK (role IN ('student', 'teacher', 'parent')),
                full_name VARCHAR(100),
                grade VARCHAR(20),
                school VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
            )
        ''')

        # Создание индексов для users
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)')

        # Создание таблицы user_progress
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id SERIAL PRIMARY KEY,
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                word_id INTEGER NOT NULL,
                level VARCHAR(1) NOT NULL CHECK (level IN ('A', 'B', 'C')),
                score DECIMAL(5,2) DEFAULT 0,
                honesty_indicator DECIMAL(5,2) DEFAULT 0,
                answers_json JSONB,
                completion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Создание индексов для user_progress
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_progress_user_id ON user_progress(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_progress_word_id ON user_progress(word_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_progress_completion ON user_progress(completion_time DESC)')

        # Создание таблицы user_skills
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_skills (
                id SERIAL PRIMARY KEY,
                user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                logic_score DECIMAL(5,2) DEFAULT 0,
                analysis_score DECIMAL(5,2) DEFAULT 0,
                ethics_score DECIMAL(5,2) DEFAULT 0,
                knowledge_score DECIMAL(5,2) DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Создание таблицы achievements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS achievements (
                id SERIAL PRIMARY KEY,
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                achievement_type VARCHAR(50) NOT NULL,
                achievement_data JSONB,
                awarded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Создание индексов для achievements
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_achievements_user_id ON achievements(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_achievements_type ON achievements(achievement_type)')

        # Создание таблицы для аналитики учителя
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teacher_analytics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value JSONB,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                period_start DATE,
                period_end DATE
            )
        ''')

        conn.commit()
        logger.info("База данных PostgreSQL инициализирована успешно")

    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        raise
    finally:
        if conn:
            release_db_connection(conn)


# ========== LIFECYCLE ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск
    logger.info("Запуск Digital Abai API с PostgreSQL...")
    create_connection_pool()
    init_database()
    yield
    # Завершение
    logger.info("Завершение Digital Abai API...")
    if connection_pool:
        connection_pool.closeall()


app = FastAPI(
    title="Цифровой Абай API",
    description="API для платформы 'Цифровой Абай: Лаборатория критического мышления'",
    version="2.0.0",
    lifespan=lifespan
)

# ========== CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== МОДЕЛИ ДАННЫХ ==========
class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    PARENT = "parent"

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    password: str = Field(..., min_length=8)
    role: UserRole
    full_name: Optional[str] = Field(None, max_length=100)
    grade: Optional[str] = Field(None, max_length=20)
    school: Optional[str] = Field(None, max_length=100)


class UserLogin(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str
    word_id: Optional[int] = None
    level: Optional[str] = None
    user_id: Optional[str] = None
    get_audio: Optional[bool] = True


class VoiceRequest(BaseModel):
    audio_base64: Optional[str] = None
    text: Optional[str] = None
    word_id: Optional[int] = None
    user_id: Optional[str] = None


class AssessmentRequest(BaseModel):
    answer: str
    word_id: int
    level: str = Field(..., pattern="^(A|B|C)$")
    criteria: List[str]
    user_id: str


class ExerciseSubmit(BaseModel):
    word_id: int
    level: str
    answers: Dict[str, Any]
    time_spent: int
    user_id: str


# ========== КОНТЕНТНАЯ БАЗА ==========




ABAI_SYSTEM_PROMPT = """Ты - цифровой Абай, помощник на образовательной платформе "Цифровой Абай: Лаборатория критического мышления"."""
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ========== УТИЛИТЫ ==========
def safe_str(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except:
            return obj.decode('latin-1', 'ignore')
    else:
        return str(obj)


def encode_utf8(text):
    if isinstance(text, bytes):
        try:
            return text.decode('utf-8')
        except:
            return text.decode('latin-1', 'ignore')
    return str(text)


def safe_text_for_tts(text: str) -> str:
    if not text:
        return "Пустой текст."

    text = safe_str(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s.,!?;:\-—()«»\n\u0400-\u04FF]', '', text, flags=re.UNICODE)

    replacements = {
        'ә': 'а', 'ғ': 'г', 'қ': 'к', 'ң': 'н', 'ө': 'о',
        'ұ': 'у', 'ү': 'у', 'һ': 'х', 'і': 'и'
    }

    for kaz, rus in replacements.items():
        text = text.replace(kaz, rus)

    if len(text) > 1500:
        text = text[:1497] + "..."

    return text.strip()


def abai_style_text(text: str) -> str:
    text = safe_str(text)
    text = text.replace(".", ". ... ")
    text = text.replace("!", "! ... ")
    text = text.replace("?", "? ... ")

    if not text.lower().startswith(("құрметті", "қарағым")):
        text = "Қарағым. ... " + text

    return text.strip()


async def text_to_abai_speech_safe(text: str):
    try:
        safe_text = safe_text_for_tts(text)
        styled_text = abai_style_text(safe_text)

        if not styled_text:
            styled_text = "Ойлан, қарағым."

        tts = gTTS(text=styled_text, lang='ru', slow=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name

        tts.save(temp_path)

        with open(temp_path, 'rb') as f:
            audio_bytes = f.read()

        os.remove(temp_path)

        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return f"data:audio/mp3;base64,{audio_base64}"

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


def calculate_honesty_indicator(answer: str, level: str) -> float:
    length_score = min(len(answer) / 500, 1.0) * 30

    key_elements = {
        "A": ["по тексту", "Абай говорит", "согласно"],
        "B": ["сравни", "анализ", "причина", "следствие"],
        "C": ["я считаю", "по моему мнению", "предлагаю", "проект"]
    }

    content_score = 0
    answer_lower = answer.lower()

    for element in key_elements.get(level, []):
        if element in answer_lower:
            content_score += 10

    if len(answer) < 50 and level in ["B", "C"]:
        content_score *= 0.5

    content_score = min(content_score, 40)

    unique_words = len(set(answer_lower.split()))
    uniqueness_score = min(unique_words / 100, 1.0) * 30

    total = length_score + content_score + uniqueness_score
    return min(total, 100)


async def update_user_skills(user_id: str, level: str, score: float, honesty: float):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        level_coeff = {
            "A": {"knowledge": 0.6, "logic": 0.2, "analysis": 0.1, "ethics": 0.1},
            "B": {"knowledge": 0.3, "logic": 0.3, "analysis": 0.3, "ethics": 0.1},
            "C": {"knowledge": 0.2, "logic": 0.2, "analysis": 0.3, "ethics": 0.3}
        }

        coeff = level_coeff.get(level, level_coeff["A"])

        cursor.execute('SELECT * FROM user_skills WHERE user_id = %s', (user_id,))
        skills = cursor.fetchone()

        if skills:
            new_knowledge = float(skills["knowledge_score"]) + score * coeff["knowledge"]
            new_logic = float(skills["logic_score"]) + score * coeff["logic"]
            new_analysis = float(skills["analysis_score"]) + honesty * coeff["analysis"]
            new_ethics = float(skills["ethics_score"]) + honesty * coeff["ethics"]

            cursor.execute('''
                UPDATE user_skills 
                SET logic_score = %s, analysis_score = %s, ethics_score = %s, 
                    knowledge_score = %s, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s
            ''', (new_logic, new_analysis, new_ethics, new_knowledge, user_id))
        else:
            cursor.execute('''
                INSERT INTO user_skills (user_id, logic_score, analysis_score, ethics_score, knowledge_score)
                VALUES (%s, %s, %s, %s, %s)
            ''', (user_id,
                  score * coeff["logic"],
                  honesty * coeff["analysis"],
                  honesty * coeff["ethics"],
                  score * coeff["knowledge"]))

        conn.commit()

    except Exception as e:
        logger.error(f"Error updating skills: {e}")
    finally:
        if conn:
            release_db_connection(conn)


# ========== API ЭНДПОИНТЫ ==========

@app.get("/")
async def root():
    return {
        "message": "Добро пожаловать в API Цифрового Абая (PostgreSQL)",
        "version": "2.0.0",
        "database": "PostgreSQL"
    }


# ========== АУТЕНТИФИКАЦИЯ ==========
@app.post("/api/auth/register")
async def register_user(user: UserCreate):
    try:
        user_id = str(uuid.uuid4())

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO users (id, username, email, password_hash, role, full_name, grade, school)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            user_id,
            user.username,
            user.email,
            user.password,  # В реальности: hash_password(user.password)
            user.role.value,
            user.full_name,
            user.grade,
            user.school
        ))

        cursor.execute('''
            INSERT INTO user_skills (user_id)
            VALUES (%s)
        ''', (user_id,))

        conn.commit()

        return {
            "success": True,
            "user_id": user_id,
            "message": "Пользователь успешно зарегистрирован"
        }

    except Exception as e:
        if "duplicate key" in str(e).lower():
            raise HTTPException(400, "Пользователь с таким email или username уже существует")
        raise HTTPException(500, f"Ошибка регистрации: {str(e)}")
    finally:
        if conn:
            release_db_connection(conn)


@app.post("/api/auth/login")
async def login_user(login: UserLogin):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT id, username, email, role, full_name, grade, school 
            FROM users 
            WHERE email = %s AND password_hash = %s AND is_active = TRUE
        ''', (login.email, login.password))

        user = cursor.fetchone()

        if not user:
            raise HTTPException(401, "Неверный email или пароль")

        cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s', (user["id"],))
        conn.commit()

        token = base64.b64encode(f"{user['id']}:{datetime.now().isoformat()}".encode()).decode()

        return {
            "success": True,
            "token": token,
            "user": dict(user),
            "message": "Вход выполнен успешно"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка входа: {str(e)}")
    finally:
        if conn:
            release_db_connection(conn)


# ========== КОНТЕНТ ==========
@app.get("/api/words")
async def get_words():
    return {
        "success": True,
        "count": len(ABAI_WORDS),
        "words": list(ABAI_WORDS.values()),
        "levels": ASSESSMENT_CRITERIA
    }


@app.get("/api/word/{word_id}")
async def get_word(word_id: int):
    if word_id not in ABAI_WORDS:
        raise HTTPException(404, "Слово не найдено")

    word = ABAI_WORDS[word_id].copy()
    audio_content = await text_to_abai_speech_safe(word["content"][:500])
    word["audio"] = audio_content

    return {
        "success": True,
        "word": word
    }


# ========== ЧАТ С АБАЕМ ==========
@app.post("/api/chat")
async def chat_with_abai(req: ChatRequest):
    try:
        message = encode_utf8(req.message)

        context = ""
        if req.word_id and req.word_id in ABAI_WORDS:
            word = ABAI_WORDS[req.word_id]
            context = f"\nКонтекст: Изучаем {word['title']}\nКлючевые темы: {', '.join(word['key_themes'])}"

        if req.level:
            level_guides = {
                "A": "Ученик на базовом уровне. Помоги проверить понимание текста.",
                "B": "Ученик на аналитическом уровне. Помоги анализировать текст.",
                "C": "Ученик на творческом уровне. Стимулируй глубокие размышления."
            }
            context += f"\n{level_guides.get(req.level, '')}"

        messages = [
            {"role": "system", "content": ABAI_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nВопрос ученика: {message}"}
        ]

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise HTTPException(502, f"Ошибка ИИ сервиса: {response.status_code}")

        data = response.json()
        ai_response = encode_utf8(data["choices"][0]["message"]["content"])

        audio_base64 = None
        if req.get_audio:
            audio_base64 = await text_to_abai_speech_safe(ai_response)

        return {
            "success": True,
            "response": ai_response,
            "audio": audio_base64,
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }

    except httpx.TimeoutException:
        raise HTTPException(504, "Таймаут соединения с ИИ")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, f"Ошибка чата: {str(e)}")


# ========== ОЦЕНКА ОТВЕТОВ ==========
@app.post("/api/assess-answer")
async def assess_answer(req: AssessmentRequest):
    try:
        if req.word_id not in ABAI_WORDS:
            raise HTTPException(404, "Слово не найдено")

        word = ABAI_WORDS[req.word_id]
        honesty_indicator = calculate_honesty_indicator(req.answer, req.level)

        assessment_prompt = f"""
        Оцени ответ ученика на задание уровня {req.level} по Слову назидания: {word['title']}

        Критерии оценки:
        {json.dumps(req.criteria, indent=2, ensure_ascii=False)}

        Ответ ученика:
        {req.answer[:2000]}

        Верни ответ в формате JSON:
        {{
            "scores": {{
                "relevance": число 0-10,
                "depth": число 0-10,
                "logic": число 0-10,
                "abai_usage": число 0-10,
                "originality": число 0-10
            }},
            "total_score": число (среднее арифметическое),
            "feedback": "развернутая обратная связь на русском языке",
            "strengths": ["сильные стороны ответа"],
            "improvements": ["предложения по улучшению"],
            "next_level_suggestion": "A/B/C"
        }}
        """

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": assessment_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            assessment = {
                "scores": {
                    "relevance": 6,
                    "depth": 5,
                    "logic": 6,
                    "abai_usage": 5,
                    "originality": 4
                },
                "total_score": 5.2,
                "feedback": "Ответ принят. Для более точной оценки используйте текстовый анализ.",
                "strengths": ["Ученик ответил на вопрос"],
                "improvements": ["Можно углубить анализ"],
                "next_level_suggestion": req.level
            }
        else:
            data = response.json()
            try:
                assessment = json.loads(data["choices"][0]["message"]["content"])
            except:
                assessment = {
                    "scores": {"relevance": 5, "depth": 5, "logic": 5, "abai_usage": 5, "originality": 5},
                    "total_score": 5.0,
                    "feedback": "Оценка сгенерирована автоматически.",
                    "strengths": [],
                    "improvements": [],
                    "next_level_suggestion": req.level
                }

        feedback_audio = await text_to_abai_speech_safe(assessment["feedback"])

        if req.user_id:
            total_score = assessment.get("total_score", 5.0) * 10

            await update_user_skills(req.user_id, req.level, total_score, honesty_indicator)

            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO user_progress (user_id, word_id, level, score, honesty_indicator, answers_json)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
                req.user_id,
                req.word_id,
                req.level,
                total_score,
                honesty_indicator,
                json.dumps({"answer": req.answer[:500]})
            ))

            conn.commit()
            release_db_connection(conn)

        return {
            "success": True,
            "assessment": assessment,
            "honesty_indicator": honesty_indicator,
            "feedback_audio": feedback_audio,
            "word": word["title"],
            "level": req.level,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(500, f"Ошибка оценки: {str(e)}")


# ========== ПРОГРЕСС ==========
@app.get("/api/progress/{user_id}")
async def get_user_progress(user_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT 
                COUNT(*) as total_tasks,
                AVG(score) as avg_score,
                AVG(honesty_indicator) as avg_honesty,
                MAX(completion_time) as last_activity
            FROM user_progress 
            WHERE user_id = %s
        ''', (user_id,))

        stats = cursor.fetchone()

        cursor.execute('''
            SELECT 
                word_id,
                level,
                COUNT(*) as attempts,
                AVG(score) as avg_score,
                AVG(honesty_indicator) as avg_honesty
            FROM user_progress 
            WHERE user_id = %s
            GROUP BY word_id, level
            ORDER BY word_id, level
        ''', (user_id,))

        word_progress = cursor.fetchall()

        cursor.execute('SELECT * FROM user_skills WHERE user_id = %s', (user_id,))
        skills = cursor.fetchone()

        cursor.execute('''
            SELECT achievement_type, achievement_data, awarded_at 
            FROM achievements 
            WHERE user_id = %s
            ORDER BY awarded_at DESC
            LIMIT 10
        ''', (user_id,))

        achievements = cursor.fetchall()

        progress_by_word = {}
        for row in word_progress:
            word_id = row["word_id"]
            if word_id not in progress_by_word:
                progress_by_word[word_id] = {}
            progress_by_word[word_id][row["level"]] = dict(row)

        return {
            "success": True,
            "user_id": user_id,
            "statistics": dict(stats) if stats else {},
            "progress_by_word": progress_by_word,
            "skills": dict(skills) if skills else {},
            "achievements": [dict(a) for a in achievements]
        }

    except Exception as e:
        logger.error(f"Progress error: {e}")
        raise HTTPException(500, f"Ошибка получения прогресса: {str(e)}")
    finally:
        if conn:
            release_db_connection(conn)


# ========== АНАЛИТИКА ДЛЯ УЧИТЕЛЯ ==========
@app.get("/api/analytics/teacher")
async def get_teacher_analytics():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT 
                COUNT(DISTINCT user_id) as active_users,
                COUNT(*) as total_completed_tasks,
                AVG(score) as platform_avg_score,
                AVG(honesty_indicator) as platform_avg_honesty
            FROM user_progress
            WHERE completion_time > CURRENT_DATE - INTERVAL '30 days'
        ''')

        platform_stats = cursor.fetchone()

        cursor.execute('''
            SELECT 
                word_id,
                COUNT(*) as attempts,
                AVG(score) as avg_score,
                AVG(honesty_indicator) as avg_honesty
            FROM user_progress
            GROUP BY word_id
            ORDER BY attempts DESC
        ''')

        word_popularity = cursor.fetchall()

        cursor.execute('''
            SELECT 
                level,
                COUNT(*) as attempts,
                AVG(score) as avg_score,
                AVG(honesty_indicator) as avg_honesty
            FROM user_progress
            GROUP BY level
            ORDER BY level
        ''')

        level_difficulty = cursor.fetchall()

        return {
            "success": True,
            "platform_statistics": dict(platform_stats) if platform_stats else {},
            "word_popularity": [dict(w) for w in word_popularity],
            "level_difficulty": [dict(l) for l in level_difficulty],
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(500, f"Ошибка аналитики: {str(e)}")
    finally:
        if conn:
            release_db_connection(conn)


# ========== ДЕРЕВО ДОБРОДЕТЕЛЕЙ ==========
@app.get("/api/achievements/virtue-tree/{user_id}")
async def get_virtue_tree(user_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT 
                word_id,
                level,
                honesty_indicator,
                completion_time,
                answers_json
            FROM user_progress 
            WHERE user_id = %s AND level = 'C'
            ORDER BY completion_time DESC
        ''', (user_id,))

        reflections = cursor.fetchall()

        fruits = []
        total_fruits = 0

        for idx, reflection in enumerate(reflections):
            honesty = float(reflection["honesty_indicator"])

            if honesty >= 80:
                fruit_type = "golden"
                size = "large"
                value = 3
            elif honesty >= 60:
                fruit_type = "ripe"
                size = "medium"
                value = 2
            else:
                fruit_type = "green"
                size = "small"
                value = 1

            word = ABAI_WORDS.get(reflection["word_id"], {"title": f"Слово {reflection['word_id']}"})

            fruits.append({
                "id": idx + 1,
                "word_id": reflection["word_id"],
                "word_title": word["title"],
                "type": fruit_type,
                "size": size,
                "value": value,
                "honesty": honesty,
                "date": reflection["completion_time"].isoformat() if reflection["completion_time"] else None
            })

            total_fruits += value

        tree_level = min(total_fruits // 10 + 1, 10)

        return {
            "success": True,
            "user_id": user_id,
            "tree": {
                "level": tree_level,
                "total_fruits": total_fruits,
                "fruits": fruits
            }
        }

    except Exception as e:
        raise HTTPException(500, f"Ошибка получения дерева добродетелей: {str(e)}")
    finally:
        if conn:
            release_db_connection(conn)


# ========== ЗДОРОВЬЕ СИСТЕМЫ ==========
@app.get("/health")
async def health_check():
    services = {
        "database": "unknown",
        "openrouter": "unknown",
        "tts": "unknown"
    }

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        services["database"] = "healthy"
        release_db_connection(conn)
    except:
        services["database"] = "unhealthy"

    try:
        test_audio = await text_to_abai_speech_safe("Проверка")
        services["tts"] = "healthy" if test_audio else "unhealthy"
    except:
        services["tts"] = "unhealthy"

    if OPENROUTER_API_KEY:
        services["openrouter"] = "configured"
    else:
        services["openrouter"] = "missing_key"

    return {
        "status": "healthy" if all(s != "unhealthy" for s in services.values()) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": services,
        "version": "2.0.0",
        "database": "PostgreSQL"
    }


# ========== SIMPLE TTS (NO DB) ==========
@app.post("/api/tts-simple")
async def tts_simple(req: TTSRequest):
    try:
        text = req.text.strip()

        if not text:
            raise HTTPException(400, "Пустой текст")

        audio_base64 = await text_to_abai_speech_safe(text)

        if not audio_base64:
            raise HTTPException(500, "Не удалось создать аудио")

        return {
            "success": True,
            "audio": audio_base64
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS simple error: {e}")
        raise HTTPException(500, "Ошибка TTS")



# ========== ЗАПУСК СЕРВЕРА ==========
if __name__ == "__main__":
    import uvicorn

    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# Конфигурация Digital Abai с PostgreSQL
OPENROUTER_API_KEY=your_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=digital_abai
DB_USER=postgres
DB_PASSWORD=password
ENVIRONMENT=development
""")
        print("Создан файл .env. Пожалуйста, настройте параметры подключения к PostgreSQL.")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )