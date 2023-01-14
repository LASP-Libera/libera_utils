"""db module"""
# Standard
import re
# Installed
from sqlalchemy.ext.declarative import declarative_base, declared_attr
# Local
from .database import _DatabaseManager


def camel_to_snake(name):
    """Convert a CamelCase string to a snake_case.

    Parameters
    ----------
    name: str
        String to convert.

    Returns
    -------
    : str
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class BaseAugmentation:
    """
    Augmentation class that provides some standard behavior to all of our models.
    See https://docs.sqlalchemy.org/en/13/orm/extensions/declarative/mixins.html#augmenting-the-base
    """
    __name__ = None  # Gets overridden later. We just want our class declaration to know this exists.
    __abstract__ = True
    __table_args__ = {"schema": "sdp"}

    @declared_attr
    def __tablename__(cls):  # pylint: disable=E0213
        """Set the __tablename__ of every model to be the snake_case form of the class name."""
        return camel_to_snake(cls.__name__)


# TODO: This will need to change when upgrading to SQLAlchemy 2.0
Base = declarative_base(cls=BaseAugmentation)


# Convenience method for getting database managers
getdb = _DatabaseManager.get
"""Convenience alias for :meth:`libera_utils.db.database._DatabaseManager.get`"""
