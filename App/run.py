import sys
import os

# Add the parent directory (App) to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
