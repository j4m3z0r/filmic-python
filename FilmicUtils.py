#!/usr/bin/python

import math

def Saturate(v) :
    return max(0.0, min(1.0, v))

def powf(v, p) :
    return math.pow(v, p)

def sqrtf(v) :
    return math.sqrt(v)

def logf(v) :
    return math.log(v)

def log2f(v) :
    return math.log(v) / math.log(2)

def expf(v) :
    return math.exp(v)

def exp2f(v) :
    return 2 ** v

def MaxFloat(x, y) :
    return max(float(x), float(y))

class Vec3 :
    def __init__(self, x, y=None, z=None) :
        # "copy constructor"
        if isinstance(x, Vec3) :
            self.x = x.x
            self.y = x.y
            self.z = x.z
            return

        self.x = float(x)
        if y is None and z is None :
            self.y = float(x)
            self.z = float(x)

        else :
            self.y = float(y)
            self.z = float(z)

    def __sub__(self, other):
        if type(other) == float :
            return Vec3(self.x - other, self.y - other, self.z - other)

        raise NotImplementedError, "__sub__ undefined for Vec3 and %s" % type(other)

    def __add__(self, other):
        if type(other) == float :
            return Vec3(self.x + other, self.y + other, self.z + other)

        raise NotImplementedError, "__add__ undefined for Vec3 and %s" % type(other)

    def __radd__(self, other):
        return self + other # commutativity FTW

    def __mul__(self, other):
        if type(other) == float :
            return Vec3(self.x * other, self.y * other, self.z * other)

        raise NotImplementedError, "__mul__ undefined for Vec3 and %s" % type(other)
