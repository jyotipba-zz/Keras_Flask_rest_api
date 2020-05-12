''' module to run gunicorn.'''

from app import app
from app import load_model

if __name__ == '__main__':
    app.run()
