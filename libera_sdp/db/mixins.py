"""Module containing Mixin classes for providing common functionality to ORM objects."""
# Installed
from sqlalchemy import inspect
from sqlalchemy.orm import Session


class ReprMixin:
    """
    Mixin class to provide a nice and configurable __repr__ method. To control which attributes
    get included in __repr__, use __repr_attrs__ on the class mixing this in.

    Yanked from https://github.com/absent1706/sqlalchemy-mixins
    """
    __abstract__ = True
    __repr_attrs__ = []
    __repr_max_length__ = 36

    @property
    def _id_str(self):
        """Figure out what the ID string should be.

        Returns
        -------
        : str
        """
        ids = inspect(self).identity
        if ids:
            return '-'.join([str(x) for x in ids]) if len(ids) > 1 \
                   else str(ids[0])
        return 'None'

    @property
    def _repr_attrs_str(self):
        """
        Create the string representation of the attributes listed in __repr_attrs__

        Returns
        -------
        : str
        """
        max_length = self.__repr_max_length__

        values = []
        single = len(self.__repr_attrs__) == 1
        for key in self.__repr_attrs__:
            if not hasattr(self, key):
                raise KeyError(f"{self.__class__} has incorrect attribute '{key}' in __repr__attrs__")
            value = getattr(self, key)
            wrap_in_quote = isinstance(value, str)

            value = str(value)
            if len(value) > max_length:
                value = value[:max_length] + '...'

            if wrap_in_quote:
                value = f"'{value}'"
            values.append(value if single else f"{key}:{value}")

        return ' '.join(values)

    def __repr__(self):
        # Get id like '#123'
        id_str = ('#' + self._id_str) if self._id_str else ''
        # Join class name, id and repr_attrs
        return f"<{self.__class__.__name__} {id_str}{f' {self._repr_attrs_str}' if self._repr_attrs_str else ''}>"


class DataProductMixin:
    """
    Mixin class that provides methods that are commonly used in data product ORM classes.
    """
    __abstract__ = True

    @classmethod
    def query(cls, session: Session = None, **filters):
        """Simple query interface that retrieves objects based on simple filter parameters. Does not support
        queries involving other relations.

        Parameters
        ----------
        session: Session, Optional
            Existing session to avoid overhead of accessing cached database manager and creating a new session.
        filters: dict, Optional
            Dictionary passed as extra keyword arguments. Passed to `.filter()`

        Returns
        -------
        : list
            List of objects
        """
        # TODO: Implement generic filterable query
        pass

    @classmethod
    def latest(cls, **filters):
        """Finds the latest products (highest version), filtered by **filters

        Parameters
        ----------
        filters: dict

        Returns
        -------

        """
        # TODO: Implement "best" latest data product retrieval
        pass

    @classmethod
    def flagged(cls, **filters):
        """Queries products with quality flags

        Parameters
        ----------
        filters

        Returns
        -------

        """
        # TODO: Implement retrieval of quality-flagged data products
        pass
