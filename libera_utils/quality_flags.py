"""Quality flag definitions"""
# Standard
from operator import or_ as _or_
from functools import reduce
# Local
from libera_utils.backports.enum_3_11 import Flag, EnumMeta, Enum, STRICT


# pylint: disable-all
class FrozenFlagMeta(EnumMeta):
    """
    Metaclass that freezes an enum entirely, preventing values from being updated, added, or deleted.
    """
    def __new__(mcs, name, bases, classdict):
        classdict['__frozenenummeta_creating_class__'] = True
        flag = super().__new__(mcs, name, bases, classdict)
        del flag.__frozenenummeta_creating_class__
        return flag

    def __call__(cls, value, names=None, *, module=None, **kwargs):
        if names is None:  # simple value lookup
            return cls.__new__(cls, value)
        enum = Enum._create_(value, names, module=module, **kwargs)
        enum.__class__ = type(cls)
        return enum

    def __setattr__(cls, name, value):
        members = cls.__dict__.get('_member_map_', {})
        if hasattr(cls, '__frozenenummeta_creating_class__') or name in members:
            return super().__setattr__(name, value)
        if hasattr(cls, name):
            msg = "{!r} object attribute {!r} is read-only"
        else:
            msg = "{!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls.__name__, name))

    def __delattr__(cls, name):
        members = cls.__dict__.get('_member_map_', {})
        if hasattr(cls, '__frozenenummeta_creating_class__') or name in members:
            return super().__delattr__(name)
        if hasattr(cls, name):
            msg = "{!r} object attribute {!r} is read-only"
        else:
            msg = "{!r} object has no attribute {!r}"
        raise AttributeError(msg.format(cls.__name__, name))


class QualityFlag(Flag, boundary=STRICT):
    """
    Subclass of Flag that add a method for decomposing a flag into its individual components
    and a property to return a list of all messages associated with a quality flag
    """

    def decompose(self):
        """
        Return the set of all set flags that form a subset of the queried flag value. Note that this is not the
        minimum set of quality flags but rather a full set of all flags such that when they are ORed together, they
        produce `self.value`

        Returns
        -------
        : tuple
            A tuple containing (members, not_covered)
            `members` is a list of flag values that are subsets of `value`
            `not_covered` is zero if the OR of members recreates `value`. Non-zero otherwise if bits are set in `value`
            that do not exist as named values in cls.
        """
        value = self.value
        not_covered = value
        flags_to_check = [  # Creates the "basis" for the quality flag
            (m, v)
            for v, m in list(self.__class__._value2member_map_.items())
            if m.name in (x.name for x in self)
        ]
        members = []
        for member, member_value in flags_to_check:
            if member_value and member_value & value == member_value:
                members.append(member)
                not_covered &= ~member_value
        if not members and value in self.__class__._value2member_map_:
            members.append(self.__class__._value2member_map_[value])
        members.sort(key=lambda m: m._value_, reverse=True)
        return members, not_covered

    @property
    def summary(self):
        """Summarize quality flag value

        Returns
        -------
        : tuple
            (value, message_list) where value is the integer value of the quality flag and message list is a list of
            strings describing the quality flag bits which are set.
        """
        members, not_covered = self.decompose()
        print(members)
        if not_covered:
            raise ValueError(f"{self.__name__} has value {self.value} but that value cannot be created by elements "
                             f"of {self.__class__}. This should never happen unless a quality flag was declared "
                             f"without using the FrozenFlagMeta metaclass.")

        try:
            return int(self.value), [m.value.message for m in members]
        except Exception as err:
            raise AttributeError(
                "Tried to summarize a quality flag but its values don't appear to have messages.") from err


class FlagBit(int):
    """Subclass of int to capture both an integer value and an accompanying message"""
    def __new__(cls, *args, message=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.message = message
        return obj

    def __str__(self):
        return f"{super().__str__()}: {self.message}"


def with_all_none(f):
    """Decorator that adds `NONE` and `ALL` psuedo-members to a QualityFlag `f`

    For example:

    .. code-block:: python

        @with_all_none
        class MyQualityFlag(QualityFlag, metaclass=FrozenFlagMeta):
            MISSING_DATA = FlagBit(0b1, message="Data is missing!")
            VOLTAGE_TOO_HIGH = FlagBit(0b10, message="Voltage is too high!")
        qf = MyQualityFlag.ALL  # Equivalent to MyQualityFlag(0b11)
        qf.summary
    """
    f._member_map_['NONE'] = f(FlagBit(0, message="No flags set."))
    f._member_map_['ALL'] = f(reduce(_or_, f))
    return f
