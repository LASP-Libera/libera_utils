"""db module"""
# Installed
from sqlalchemy.ext.declarative import declarative_base
# Local
from .database import _DatabaseManager

Base = declarative_base()

# Convenience method for getting database managers
getdb = _DatabaseManager.get
