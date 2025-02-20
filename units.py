"""rename to strong_typing?

Physical-type correctness in scientific Python
https://arxiv.org/pdf/1807.07643.pdf
"""


from typing import Optional


class Meter(float):
    # if meter inherits from Meter(float), then it would be possible to use the 
    # variable of this class, But I want user to be forced to specify the unit.
    # But I want the type be use in numpy arrays, so it has to behave like float.
    def __init__(self, value: float = 0.0):
        self._value = value

    ### getters
    @property
    def meter(self) -> float:
        return self._value
    @property
    def centimeter(self) -> float:
        return self._value * 1e2
    @property
    def milimeter(self) -> float:
        return self._value * 1e3

    ### setters
    @meter.setter
    def meter(self, value_in: float):
        self._value = value_in
    @centimeter.setter
    def centimeter(self, value_in: float):
        self._value = value_in * 1e-2
    @milimeter.setter
    def milimeter(self, value_in: float):
        self._value = value_in * 1e-3


class Centimeter(float):
    def __init__(self, value: float):
        self._value = value


d1: Meter = Meter(20)


# Alternatively:
class Distance:
    """Note this class doesn't allow setting attribute after initialization"""
    def __init__(
            self,
            *,
            meter: Optional[float],
            centimeter: Optional[float],
            milimeter: Optional[float],
    ) -> None:
        inarg_count = sum([meter is not None, centimeter is not None, milimeter is not None])
        assert inarg_count == 1, "one and only one value must be provided"
        if meter is not None:
            self._meter = meter
            return
        if centimeter is not None:
            self._meter = centimeter * 10e-2
            return
        if milimeter is not None:
            self._meter = centimeter * 10e-3
            return
        assert False, "this should not happen"

    def as_meter(self) -> float:
        return self._meter
    def as_centimeter(self) -> float:
        return self._meter * 10e2
    def as_miliimeter(self) -> float:
        return self._meter * 10e3
    
    

        


# Operator	Magic Method
# +	__add__(self, other)
# –	__sub__(self, other)
# *	__mul__(self, other)
# /	__truediv__(self, other)
# //	__floordiv__(self, other)
# %	__mod__(self, other)
# **	__pow__(self, other)
# >>	__rshift__(self, other)
# <<	__lshift__(self, other)
# &	__and__(self, other)
# |	__or__(self, other)
# ^	__xor__(self, other)

# Comparison Operators:
# Operator	Magic Method
# <	__lt__(self, other)
# >	__gt__(self, other)
# <=	__le__(self, other)
# >=	__ge__(self, other)
# ==	__eq__(self, other)
# !=	__ne__(self, other)

# Assignment Operators:
# Operator	Magic Method
# -=	__isub__(self, other)
# +=	__iadd__(self, other)
# *=	__imul__(self, other)
# /=	__idiv__(self, other)
# //=	__ifloordiv__(self, other)
# %=	__imod__(self, other)
# **=	__ipow__(self, other)
# >>=	__irshift__(self, other)
# <<=	__ilshift__(self, other)
# &=	__iand__(self, other)
# |=	__ior__(self, other)
# ^=	__ixor__(self, other)

# Unary Operators:
# Operator	Magic Method
# –	__neg__(self)
# +	__pos__(self)
# ~	__invert__(self)
