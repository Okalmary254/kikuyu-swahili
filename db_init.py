
import os
import sys
from sqlalchemy import create_engine, text # type: ignore
from sqlalchemy.exc import SQLAlchemyError # type: ignore
import logging
import asyncio
import asyncpg # type: ignore
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)


##create db
ADMIN_DATABASE_URL = os.getenv("ADMIN_DATABASE_URL", "postgresql://user:password@postgres:5432/postgres")  # Admin DB for creation

def create_database_if_not_exists():
    try:
        admin_engine = create_engine(ADMIN_DATABASE_URL)
        with admin_engine.connect() as conn:
            conn = conn.execution_options(autocommit=True)  # Enables autocommit for DDL statements
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": os.getenv("POSTGRES_DB", "chatdb")}
            )
            if not result.fetchone():
                logger.info(f"Database '{os.getenv('POSTGRES_DB')}' does not exist. Creating...")
                conn.execute(text(f"CREATE DATABASE \"{os.getenv('POSTGRES_DB')}\""))
                logger.info(f"Database '{os.getenv('POSTGRES_DB')}' created successfully")
            else:
                logger.info(f"Database '{os.getenv('POSTGRES_DB')}' already exists")
        admin_engine.dispose()
    except SQLAlchemyError as e:
        logger.error(f"Error creating database: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error creating database: {e}")
        sys.exit(1)


##create tables
def create_tables():
    try:
        engine = create_engine(DATABASE_URL) # type: ignore
        with engine.connect() as conn:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))  # Enables UUID functions
            conn.commit()
            conn.execute(text('DROP TABLE IF EXISTS chat_messages'))  # Clean slate
            conn.commit()
            create_table_sql = """
            CREATE TABLE chat_messages (
                message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id UUID NOT NULL,
                user_message TEXT,
                system_message TEXT,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT check_message_content CHECK (user_message IS NOT NULL OR system_message IS NOT NULL)
            );
            CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
            CREATE INDEX idx_chat_messages_timestamp ON chat_messages(timestamp);
            CREATE INDEX idx_chat_messages_role ON chat_messages(role);
            """
            conn.execute(text(create_table_sql))
            conn.commit()
            logger.info("Tables created successfully")
        engine.dispose()
    except SQLAlchemyError as e:
        logger.error(f"Error creating tables: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error creating tables: {e}")
        sys.exit(1)

##testing connections, lol
async def test_database_connection():
    try:
        conn = await asyncpg.connect(DATABASE_URL) # type: ignore
        result = await conn.fetchval("SELECT version()")  # Get Postgres version
        logger.info(f"Database connection successful. PostgreSQL version: {result}")
        
        table_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
            'chat_messages'
        )
        
        if table_exists:
            logger.info("chat_messages table exists and is accessible")
            test_session_id = uuid.uuid4()
            test_message_id = await conn.fetchval("""
                INSERT INTO chat_messages (session_id, user_message, role, timestamp)
                VALUES ($1, $2, $3, $4)
                RETURNING message_id
            """, test_session_id, "Test message", "user", datetime.now(timezone.utc))
            
            logger.info(f"Test record inserted with ID: {test_message_id}")
            
            retrieved_message = await conn.fetchrow("""
                SELECT message_id, session_id, user_message, role
                FROM chat_messages WHERE message_id = $1
            """, test_message_id)
            
            if retrieved_message:
                logger.info(f"Test record successfully retrieved: {dict(retrieved_message)}")
            else:
                logger.error("Failed to retrieve test record")
            
            await conn.execute("DELETE FROM chat_messages WHERE message_id = $1", test_message_id)
            logger.info("Test record cleaned up")
        else:
            logger.error("chat_messages table does not exist!")
        
        await conn.close()
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        sys.exit(1)        