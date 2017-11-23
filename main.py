#!/usr/bin/python

import FilmicColorGrading

def main() :
    nrSamples = 256

    # Set the suggested defaults from jhable's blog post:
    userParams = FilmicColorGrading.UserParams()
    userParams.m_filmicToeStrength = 0.5
    userParams.m_filmicToeLength = 0.5
    userParams.m_filmicShoulderStrength = 2.0
    userParams.m_filmicShoulderLength = 0.5
    userParams.m_filmicShoulderAngle = 1.0

    rawParams = FilmicColorGrading.RawFromUserParams(userParams)
    evalParams = FilmicColorGrading.EvalFromRawParams(rawParams)
    bakeParams = FilmicColorGrading.BakeFromEvalParams(evalParams, nrSamples, FilmicColorGrading.kTableSpacing_Quadratic)

    for i in range(len(bakeParams.m_curveG)) :
        v = float(i) / (nrSamples - 1)
        print "%.3f,%.3f" % (v, bakeParams.m_curveG[i])

if __name__ == '__main__' :
    main()

