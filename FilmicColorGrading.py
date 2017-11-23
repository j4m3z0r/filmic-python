#!/usr/bin/python

import copy
import math
import FilmicToneCurve
from FilmicUtils import *

# enum eTableSpacing
kTableSpacing_Linear = 0
kTableSpacing_Quadratic = 1
kTableSpacing_Quartic = 2
kTableSpacing_Num = 3

def ApplyLiftInvGammaGain(lift, invGamma, gain, v) :
    # lerp gain
    lerpV = Saturate(powf(v,invGamma))
    dst = gain*lerpV + lift*(1.0-lerpV)
    return dst

def ApplySpacing(v, spacing) :
    if spacing == kTableSpacing_Linear :
        return v

    if spacing == kTableSpacing_Quadratic :
        return v*v

    if spacing == kTableSpacing_Quartic :
        return v*v*v*v

    # assert?
    return 0.0

def ApplySpacingInv(v, spacing) :
    if spacing == kTableSpacing_Linear :
        return v

    if spacing == kTableSpacing_Quadratic :
        return sqrtf(v)

    if spacing == kTableSpacing_Quartic :
        return sqrtf(sqrtf(v))

    # assert?
    return 0.0

# convert from gamma space to linear space, and then normalize it
def ColorLinearFromGammaNormalize(val) :
    # We aren't using a perceptually linear luminance because the entire point of this operation
    # is to change the color channels relative to each other. So it's fine if these adjustments
    # change the perceived luminance.

    ret = Vec3(0, 0, 0)
    ret.x = powf(val.x, 2.2)
    ret.y = powf(val.y, 2.2)
    ret.z = powf(val.z, 2.2)

    mid = (ret.x + ret.y + ret.z)/3.0

    ret.x /= mid
    ret.y /= mid
    ret.z /= mid

    return ret

def ColorLinearFromGamma(val) :
    ret = Vec3(0, 0, 0)
    ret.x = powf(val.x,2.2)
    ret.y = powf(val.y,2.2)
    ret.z = powf(val.z,2.2)

    return ret

# the FilmicColorGrading class contains only static methods, so we just put
# them all in module scope for simplicity, rather than making a class that we
# never instantiate.
def RawFromUserParams(userParams) :
    rawParams = RawParams()

    # convert color from gamma to linear space
    rawParams.m_colorFilter.x = powf(userParams.m_colorFilter.x,2.2)
    rawParams.m_colorFilter.y = powf(userParams.m_colorFilter.y,2.2)
    rawParams.m_colorFilter.z = powf(userParams.m_colorFilter.z,2.2)

    # hardcode the luminance weights to something reasonable, but not perceptually correct
    rawParams.m_luminanceWeights = Vec3(.25, .5, .25)

    # direct copy for saturation
    rawParams.m_saturation = userParams.m_saturation

    # direct copy for strength, but midpoint/epsilon are hardcoded
    rawParams.m_exposureBias = userParams.m_exposureBias
    rawParams.m_contrastStrength = userParams.m_contrast
    rawParams.m_contrastEpsilon = 1e-5
    rawParams.m_contrastMidpoint = 0.18

    filmicParams = FilmicToneCurve.CurveParamsUser()
    filmicParams.m_gamma = userParams.m_filmicGamma
    filmicParams.m_shoulderAngle = userParams.m_filmicShoulderAngle
    filmicParams.m_shoulderLength = userParams.m_filmicShoulderLength
    filmicParams.m_shoulderStrength = userParams.m_filmicShoulderStrength
    filmicParams.m_toeLength = userParams.m_filmicToeLength
    filmicParams.m_toeStrength = userParams.m_filmicToeStrength

    rawParams.m_filmicCurve = FilmicToneCurve.CalcDirectParamsFromUser(filmicParams)

    liftC = copy.copy(userParams.m_shadowColor)
    gammaC = copy.copy(userParams.m_midtoneColor)
    gainC = copy.copy(userParams.m_highlightColor)

    avgLift = (liftC.x+liftC.y+liftC.z)/3.0
    liftC = liftC - avgLift

    avgGamma = (gammaC.x + gammaC.y + gammaC.z)/3.0
    gammaC = (gammaC - avgGamma)

    avgGain = (gainC.x+gainC.y+gainC.z)/3.0
    gainC = (gainC - avgGain)

    rawParams.m_liftAdjust  = 0.0 + (liftC + userParams.m_shadowOffset)
    rawParams.m_gainAdjust  = 1.0 + (gainC + userParams.m_highlightOffset)

    midGrey = Vec3(0.5 + (gammaC + userParams.m_midtoneOffset))
    H = Vec3(rawParams.m_gainAdjust)
    S = Vec3(rawParams.m_liftAdjust)

    rawParams.m_gammaAdjust.x = logf((0.5 - S.x)/(H.x-S.x))/logf(midGrey.x)
    rawParams.m_gammaAdjust.y = logf((0.5 - S.y)/(H.y-S.y))/logf(midGrey.y)
    rawParams.m_gammaAdjust.z = logf((0.5 - S.z)/(H.z-S.z))/logf(midGrey.z)

    # gamma after filmic curve to convert to display space
    rawParams.m_postGamma = userParams.m_postGamma

    return rawParams

def EvalFromRawParams(rawParams) :
    dstParams = EvalParams()

    # bake color filter and exposure bias together
    dstParams.m_linColorFilterExposure = rawParams.m_colorFilter * exp2f(rawParams.m_exposureBias)

    dstParams.m_luminanceWeights = rawParams.m_luminanceWeights
    dstParams.m_saturation = rawParams.m_saturation

    dstParams.m_contrastStrength = rawParams.m_contrastStrength
    dstParams.m_contrastLogMidpoint = log2f(rawParams.m_contrastMidpoint)
    dstParams.m_contrastEpsilon = rawParams.m_contrastEpsilon

    dstParams.m_filmicCurve = FilmicToneCurve.CreateCurve(rawParams.m_filmicCurve)

    dstParams.m_postGamma = rawParams.m_postGamma

    dstParams.m_liftAdjust = rawParams.m_liftAdjust
    dstParams.m_invGammaAdjust.x = 1.0 / (rawParams.m_gammaAdjust.x)
    dstParams.m_invGammaAdjust.y = 1.0 / (rawParams.m_gammaAdjust.y)
    dstParams.m_invGammaAdjust.z = 1.0 / (rawParams.m_gammaAdjust.z)
    dstParams.m_gainAdjust = rawParams.m_gainAdjust

    return dstParams

def BakeFromEvalParams(srcParams, curveSize, spacing) :
    # in the curve, we are baking the following steps:
    # v = EvalContrast(v)
    # v = EvalFilmicCurve(v)
    # v = EvalLiftGammaGain(v)

    # So what is the maximum value to bake into the curve? It's filmic W with inverse contrast applied
    maxTableValue = EvalParams.EvalLogContrastFuncRev(srcParams.m_filmicCurve.m_W,srcParams.m_contrastEpsilon,srcParams.m_contrastLogMidpoint,srcParams.m_contrastStrength)

    dstCurve = BakedParams()
    dstCurve.m_curveSize = curveSize
    dstCurve.m_spacing = spacing

    dstCurve.m_saturation = srcParams.m_saturation
    dstCurve.m_linColorFilterExposure = srcParams.m_linColorFilterExposure * (1.0 / maxTableValue)
    dstCurve.m_luminanceWeights = srcParams.m_luminanceWeights

    dstCurve.m_curveB = [None] * curveSize
    dstCurve.m_curveG = [None] * curveSize
    dstCurve.m_curveR = [None] * curveSize

    for i in range(curveSize) :
        t = float(i)/float(curveSize-1)

        t = ApplySpacing(t,spacing) * maxTableValue

        rgb = Vec3(t,t,t)
        rgb = srcParams.EvalContrast(rgb)
        rgb = srcParams.EvalFilmicCurve(rgb)
        rgb = srcParams.EvalLiftGammaGain(rgb)

        dstCurve.m_curveR[i] = rgb.x
        dstCurve.m_curveG[i] = rgb.y
        dstCurve.m_curveB[i] = rgb.z

    return dstCurve
# End static methods converted to module methods.

class UserParams :
    def __init__(self) :
        self.Reset()

    def Reset(self) :
        self.m_colorFilter = Vec3(1, 1, 1)
        self.m_saturation = 1.0
        self.m_exposureBias = 0.0

        # no contrast midpoint, hardcoded to .18
        # no contrast epislon, hardcoded to 1e-5
        self.m_contrast = 1.0

        # filmic tonemapping
        self.m_filmicToeStrength = 0.0
        self.m_filmicToeLength = 0.5
        self.m_filmicShoulderStrength = 0.0
        self.m_filmicShoulderLength = 0.5
        self.m_filmicShoulderAngle = 0.0
        self.m_filmicGamma = 1.0 # gamma to convolve into the filmic curve

        self.m_postGamma = 1.0 # after filmic curve, as a separate step

        self.m_shadowColor = Vec3(1.0, 1.0, 1.0)
        self.m_midtoneColor = Vec3(1.0, 1.0, 1.0)
        self.m_highlightColor = Vec3(1.0, 1.0, 1.0)

        self.m_shadowOffset = 0.0
        self.m_midtoneOffset = 0.0
        self.m_highlightOffset = 0.0

# These params are roughly in the order they are applied.
class RawParams :
    def __init__(self) :
        self.Reset()

    def Reset(self) :
        self.m_colorFilter = Vec3(1, 1, 1)

        # Saturation could be argued to go later, but if you do it later I feel like it gets in the way of log contrast. It's also
        # nice to be here so that everything after can be merged into a 1d curve for each channel.
        self.m_saturation = 1.0

        self.m_luminanceWeights = Vec3(.25, .50, .25)

        # exposure and contrast
        self.m_exposureBias = 0.0 # in f stops
        self.m_contrastStrength = 1.0
        self.m_contrastMidpoint = 0.20
        self.m_contrastEpsilon = 1e-5

        # filmic curve
        self.m_filmicCurve = FilmicToneCurve.CurveParamsDirect()

        # lift/gamma/gain, aka highlights/midtones/shadows, aka slope/power/offset
        self.m_liftAdjust = Vec3(0, 0, 0)
        self.m_gammaAdjust = Vec3(0, 0, 0)
        self.m_gainAdjust = Vec3(0, 0, 0)

        # final adjustment to image, after all other curves
        self.m_postGamma = 1.0

# modified version of the the raw params which has precalculated values
class EvalParams :
    def __init__(self) :
        self.Reset()

    def Reset(self) :
        self.m_linColorFilterExposure = Vec3(1, 1, 1)

        self.m_luminanceWeights = Vec3(.25, .5, .25)
        self.m_saturation = 1.0

        self.m_contrastStrength = 1.0
        self.m_contrastLogMidpoint = log2f(.18)
        self.m_contrastEpsilon = 1e-5

        self.m_filmicCurve = FilmicToneCurve.FullCurve()

        self.m_postGamma = 1.0

        self.m_liftAdjust = Vec3(0.0, 0.0, 0.0)
        self.m_invGammaAdjust = Vec3(1.0, 1.0, 1.0) # note that we invert gamma to skip the divide, also convolves the final gamma into it
        self.m_gainAdjust = Vec3(1.0, 1.0, 1.0)

    # performs all of these calculations in order
    def EvalFullColor(self, src) :
        v = src
        v = self.EvalExposure(v)
        v = self.EvalSaturation(v)
        v = self.EvalContrast(v)
        v = self.EvalFilmicCurve(v)
        v = self.EvalLiftGammaGain(v)
        return v

    def EvalExposure(self, v) :
        return v * self.m_linColorFilterExposure

    def EvalSaturation(self, v) :
        grey = Vec3.Dot(v,self.m_luminanceWeights)
        ret = Vec3(grey) + self.m_saturation*(v - Vec3(grey))
        return ret

    @staticmethod
    def EvalLogContrastFunc(x, eps, logMidpoint, contrast) :
        logX = log2f(x+eps)
        adjX = logMidpoint + (logX - logMidpoint) * contrast
        ret = max(0.0,exp2f(adjX) - eps)
        return ret

    # inverse of the log contrast function
    @staticmethod
    def EvalLogContrastFuncRev(x, eps, logMidpoint, contrast) :
        # eps
        logX = log2f(x+eps)
        adjX = (logX - logMidpoint)/contrast + logMidpoint
        ret = max(0.0,exp2f(adjX) - eps)
        return ret

    def EvalContrast(self, v) :
        ret = Vec3(0, 0, 0)
        ret.x = EvalParams.EvalLogContrastFunc(v.x, self.m_contrastEpsilon, self.m_contrastLogMidpoint, self.m_contrastStrength)
        ret.y = EvalParams.EvalLogContrastFunc(v.y, self.m_contrastEpsilon, self.m_contrastLogMidpoint, self.m_contrastStrength)
        ret.z = EvalParams.EvalLogContrastFunc(v.z, self.m_contrastEpsilon, self.m_contrastLogMidpoint, self.m_contrastStrength)
        return ret

    def EvalFilmicCurve(self, v) : # also converts from linear to gamma
        ret = Vec3(0, 0, 0)
        ret.x = self.m_filmicCurve.Eval(v.x);
        ret.y = self.m_filmicCurve.Eval(v.y);
        ret.z = self.m_filmicCurve.Eval(v.z);

        # also apply the extra gamma, which has not been convolved into the filmic curve
        ret.x = powf(ret.x, self.m_postGamma);
        ret.y = powf(ret.y, self.m_postGamma);
        ret.z = powf(ret.z, self.m_postGamma);

        return ret;

    def EvalLiftGammaGain(self, v) :
        ret = Vec3(0, 0, 0);
        ret.x = ApplyLiftInvGammaGain(self.m_liftAdjust.x, self.m_invGammaAdjust.x, self.m_gainAdjust.x,v.x);
        ret.y = ApplyLiftInvGammaGain(self.m_liftAdjust.y, self.m_invGammaAdjust.y, self.m_gainAdjust.y,v.y);
        ret.z = ApplyLiftInvGammaGain(self.m_liftAdjust.z, self.m_invGammaAdjust.z, self.m_gainAdjust.z,v.z);
        return ret;


class BakedParams :
    def __init__(self) :
        self.Reset()

    def Reset(self) :
        self.m_linColorFilterExposure = Vec3(1, 1, 1)

        self.m_saturation = 1.0

        self.m_curveSize = 256

        self.m_curveR = list()
        self.m_curveG = list()
        self.m_curveB = list()

        self.m_spacing = kTableSpacing_Quadratic
        self.m_luminanceWeights = Vec3(.25, .5, .25)

    @staticmethod
    def SampleTable(curve, normX) :
        size = len(curve)

        x = normX * float(size-1) + .5

        # Tex2d-ish. When implementing in a shader, make sure to do the pad above, but everything below will be in the Tex2d call.
        baseIndex = int(max(0,x-.5))
        t = (x-0.5) - float(baseIndex)

        x0 = max(0,min(baseIndex,size-1))
        x1 = max(0,min(baseIndex+1,size-1))

        v0 = curve[x0]
        v1 = curve[x1]

        ret = v0*(1.0-t) + v1*t
        return ret

    def EvalColor(self, srcColor) :
        rgb = copy.copy(srcColor)

        # exposure and color filter
        rgb = rgb * self.m_linColorFilterExposure

        # saturation
        grey = Vec3.Dot(rgb, self.m_luminanceWeights)
        rgb = Vec3(grey) + self.m_saturation*(rgb - Vec3(grey))

        rgb.x = ApplySpacingInv(rgb.x,self.m_spacing)
        rgb.y = ApplySpacingInv(rgb.y,self.m_spacing)
        rgb.z = ApplySpacingInv(rgb.z,self.m_spacing)

        # contrast, filmic curve, gamme 
        rgb.x = BakedParams.SampleTable(self.m_curveR,rgb.x)
        rgb.y = BakedParams.SampleTable(self.m_curveG,rgb.y)
        rgb.z = BakedParams.SampleTable(self.m_curveB,rgb.z)

        return rgb

