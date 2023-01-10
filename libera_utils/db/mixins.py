"""
Module containing Mixin classes for providing common functionality to ORM objects.
"""
# Installed
from sqlalchemy import inspect
from sqlalchemy.orm import Session, Query


class ReprMixin:
    """
    Mixin class to provide a nice and configurable __repr__ method. To control which attributes
    get included in __repr__, use __repr_attrs__ on the class mixing this in.

    Yanked from https://github.com/absent1706/sqlalchemy-mixins
    """
    __abstract__ = True
    __repr_attrs__ = []
    __repr_max_length__ = 36

    def __init__(self, *args, **kwargs):
        # TODO: Remove this after PyCharm updates past 2021.3.2. It's only here to suppress an IDE warning.
        pass

    @property
    def _id_str(self):
        """
        Figure out what the ID string should be.

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

    def __init__(self, *args, **kwargs):
        # TODO: Remove this after PyCharm updates past 2021.3.2. It's only here to suppress an IDE warning.
        pass

    @classmethod
    def _filter(cls, query: Query, **filters):
        """ Filters an existing query object for a set of keyword filters """
        for attr, value in filters.items():
            query.filter(getattr(cls, attr) == value)
        return query

    @classmethod
    def latest(cls, session: Session = None, **filters):
        """
        Finds the latest products (highest revision), filtered by **filters**

        Parameters
        ----------
        session: Session, Optional
            If None, creates a new session
        filters: dict
            Optional filters to narrow down results

        Returns
        -------
        : list
        """
        # TODO: Implement logic for finding the latest revision of each "unique observation", whatever that ends up
        #    meaning. For BDS products, that could be a specific timerange or just a date.
        pass

    @classmethod
    def flagged(cls, session: Session = None, **filters):
        """
        Queries products with quality flags, optionally filtered by **filters**

        Parameters
        ----------
        session: Session, Optional
            If None, creates a new session
        filters: dict
            Optional filters to narrow results

        Returns
        -------
        : list
        """
        # TODO: Implement retrieval of quality-flagged data products once quality flags are defined
        pass
