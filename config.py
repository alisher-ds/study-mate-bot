import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Telegram and Provider Credentials
BOT_TOKEN: Optional[str] = os.getenv("BOT_TOKEN")
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

# Default Model and Storage Configuration
AI_MODEL: str = "llama-3.3-70b-versatile"
DB_NAME: str = "users.db"
