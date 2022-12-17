"""Quality flag definitions"""

from enum import Flag, EnumMeta, Enum
from operator import or_ as _or_
from functools import reduce

#pylint: disable-all
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


class QualityFlag(Flag):
    """
    Subclass of Flag that add a method for decomposing a flag into its individual components
    and a property to return a list of all messages associated with a quality flag
    """
    # Overriding enum module implementation to incorporate patched _decompose method below
    # TODO: Remove this if we upgrade to python 3.9.0a2 or newer
    @classmethod
    def _create_pseudo_member_(cls, value):
        """
        Create a composite member iff value contains only members.
        """
        pseudo_member = cls._value2member_map_.get(value, None)
        if pseudo_member is None:
            # verify all bits are accounted for
            _, extra_flags = _patched_enum_decompose(cls, value)
            if extra_flags:
                raise ValueError("%r is not a valid %s" % (value, cls.__name__))
            # construct a singleton enum pseudo-member
            pseudo_member = object.__new__(cls)
            pseudo_member._name_ = None
            pseudo_member._value_ = value
            # use setdefault in case another thread already created a composite
            # with this value
            pseudo_member = cls._value2member_map_.setdefault(value, pseudo_member)
        return pseudo_member

    # Overriding enum module implementation to incorporate patched _decompose method below
    # TODO: Remove this if we upgrade to python 3.9.0a2 or newer
    def __repr__(self):
        cls = self.__class__
        if self._name_ is not None:
            return '<%s.%s: %r>' % (cls.__name__, self._name_, self._value_)
        members, uncovered = _patched_enum_decompose(cls, self._value_)
        return '<%s.%s: %r>' % (
                cls.__name__,
                '|'.join([str(m._name_ or m._value_) for m in members]),
                self._value_,
                )

    # Overriding enum module implementation to incorporate patched _decompose method below
    # TODO: Remove this if we upgrade to python 3.9.0a2 or newer
    def __str__(self):
        cls = self.__class__
        if self._name_ is not None:
            return '%s.%s' % (cls.__name__, self._name_)
        members, uncovered = _patched_enum_decompose(cls, self._value_)
        if len(members) == 1 and members[0]._name_ is None:
            return '%s.%r' % (cls.__name__, members[0]._value_)
        else:
            return '%s.%s' % (
                    cls.__name__,
                    '|'.join([str(m._name_ or m._value_) for m in members]),
                    )

    # Overriding enum module implementation to incorporate patched _decompose method below
    # TODO: Remove this if we upgrade to python 3.9.0a2 or newer
    def __invert__(self):
        members, uncovered = _patched_enum_decompose(self.__class__, self._value_)
        inverted = self.__class__(0)
        for m in self.__class__:
            if m not in members and not (m._value_ & self._value_):
                inverted = inverted | m
        return self.__class__(inverted)

    def decompose(self):
        """
        Return the set of all set flags that form a subset of the queried flag value. Note that this is not the
        minimum set of quality flags but rather a full set of all flags such that when they are ORed together, they
        produce self.value

        :return: members, not_covered
            `members` is a list of flag values that are subsets of `value`
            `not_covered` is zero if the OR of members recreates `value`. Non-zero otherwise if bits are set in `value`
                that do not exist as named values in cls.
        """
        value = self.value
        not_covered = value
        flags_to_check = [
            (m, v)
            for v, m in list(self.__class__._value2member_map_.items())
            if m.name is not None
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
        members, not_covered = self.decompose()

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


# This is a patched version of enum._decompose method that was fixed in Python 3.9.0a2
# See: https://bugs.python.org/issue38045
# TODO: Remove this if we upgrade to python 3.9.0a2 or newer
def _patched_enum_decompose(flag, value):
    """Extract all members from the value."""
    # _decompose is only called if the value is not named
    not_covered = value
    negative = value < 0
    members = []
    for member in flag:
        member_value = member.value
        if member_value and member_value & value == member_value:
            members.append(member)
            not_covered &= ~member_value
    if not negative:
        tmp = not_covered
        while tmp:
            high_bit = tmp.bit_length() - 1
            flag_value = 2 ** high_bit
            if flag_value in flag._value2member_map_:
                members.append(flag._value2member_map_[flag_value])
                not_covered &= ~flag_value
            tmp &= ~flag_value
    if not members and value in flag._value2member_map_:
        members.append(flag._value2member_map_[value])
    members.sort(key=lambda m: m._value_, reverse=True)
    if len(members) > 1 and members[0].value == value:
        # we have the breakdown, don't need the value member itself
        members.pop(0)
    return members, not_covered


def with_all_none(f):
    """Add NONE and ALL psuedo-members to f"""
    f._member_map_['NONE'] = f(FlagBit(0, message="No flags set."))
    f._member_map_['ALL'] = f(reduce(_or_, f))
    return f
