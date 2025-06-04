"""Quality flag definitions"""

from enum import NAMED_FLAGS, STRICT, UNIQUE, Flag, verify


class LiberaFlag(Flag, boundary=STRICT):
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
            for v, m in list(self.__class__._value2member_map_.items())  # pylint: disable=protected-access
            if m.name in (x.name for x in self)
        ]
        members = []
        for member, member_value in flags_to_check:
            if member_value and member_value & value == member_value:
                members.append(member)
                not_covered &= ~member_value
        if not members and value in self.__class__._value2member_map_:  # pylint: disable=protected-access
            members.append(self.__class__._value2member_map_[value])  # pylint: disable=protected-access
        members.sort(key=lambda m: m._value_, reverse=True)  # pylint: disable=protected-access
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
            raise ValueError(
                f"{self.__class__.__name__} has value {self.value} "
                "but that value cannot be created by elements "
                f"of {self.__class__}. This should never happen unless a quality flag was declared "
                "without using the FrozenFlagMeta metaclass."
            )

        try:
            return self.value, [m.value.message for m in members]
        except Exception as err:
            raise AttributeError(
                "Tried to summarize a quality flag but its values don't appear to have messages."
            ) from err


class FlagBit(int):
    """Subclass of int to capture both an integer value and an accompanying message"""

    def __new__(cls, *args, message=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj.message = message
        return obj

    def __str__(self):
        return f"{super().__str__()}: {self.message}"  # pylint: disable=no-member


@verify(UNIQUE, NAMED_FLAGS)
class LiberaQualityFlag(LiberaFlag):
    """
    TODO: Once these quality flags are well defined, write tests against them
    """

    MISSING_DATA = FlagBit(0b1, message="At least some data is missing")
