import sys
import os

# Add the current directory to the sys.path to allow relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from app import app

if __name__ == "__main__":
    app.run(debug=True)
