from __future__ import print_function
import wave, struct
import numpy as np
import cv2


def getWavFrames(filename):
    ''' Returns a wav file as an array '''
    waveFile = wave.open(filename, 'r')

    length = waveFile.getnframes()
    frames = []
    for i in range(0,length):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        frames += [int(data[0])]
    waveFile.close()
    return frames


def applyMaxFilterReducedFPS(wavFrames, audioFPS, videoFPS):
    ''' Applies a max filter to a set of wavFrames, in blocks
        of length audioFPS/videoFPS
    '''
    newFrames = []
    blockLength = audioFPS/videoFPS
    print("Applying Filter")
    print('|' + ' '*50 + '|')
    print(' ', end='')
    tick = (len(wavFrames)/blockLength)/50
    i = 0
    while int((i*1.0/videoFPS)*audioFPS) + blockLength < len(wavFrames):
        if i % tick == 0:
            print('=', end='')
        start = int((i*1.0/videoFPS)*audioFPS)
        end = start + blockLength
        newFrames += [max(wavFrames[start:end])]
        i += 1
    print('')
    return newFrames


def getHighlightFrames(vidWavPairs):
    ''' Input: 
        Output: [(frame, vidFilename)] for every frame in the video

        Feels like it could be optimized to use less storage/time by not keeping
        the whole sliding window
    '''
    audioFPS = 16000
    videoFPS = 25
    audioFramesPerVisual = audioFPS/videoFPS
    
    if len(vidWavPairs) == 0:
        return []

    # Transform all the .wav files to arrays, and apply a sliding window max filter
    allWavFrames = []
    for (vidFilename, wavFilename) in vidWavPairs:
        print("Reading " + wavFilename + '...')
        wavFrames = getWavFrames(wavFilename)
        wavFrames = applyMaxFilterReducedFPS(wavFrames, audioFPS, videoFPS)
        allWavFrames += [(vidFilename, wavFrames)]

    # Chop to the length of the shortest wav array
    minWavArray = min([len(x[1]) for x in allWavFrames])
    allWavFrames = [(x[0], x[1][:minWavArray]) for x in allWavFrames]

    highlightFrames = []
    
    # For each frame, check which audio had highest max, add that video
    for i in xrange(len(allWavFrames[0][1])):
        maxVidFilename = max([(x[1][i], x[0]) for x in allWavFrames])[1]
        highlightFrames += [(i,maxVidFilename)]

    return highlightFrames


def makeNaiveVideo(outputFilename, vidWavPairs):
    ''' Creates an avi from a list of [(vidFilename, wavFilename)] '''
    highlights = getHighlightFrames(vidWavPairs)
    videoFPS = 25
    
    tempCap = cv2.VideoCapture(vidWavPairs[0][0])
    ret,frame = tempCap.read()
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outputF = cv2.VideoWriter(outputFilename, fourcc, videoFPS, (width,height))

    inputCaps = {}
    for (vidFilename, wavFilename) in vidWavPairs:
        cap = cv2.VideoCapture(vidFilename)
        inputCaps[vidFilename] = cap

    for (i, vidFilename) in highlights:
        ret, frame = (False, [0])
        for k in inputCaps:
            if k == vidFilename:
                ret, frame = inputCaps[k].read()
            else:
                a,b = inputCaps[k].read()
        
        outputF.write(frame)
            
    for k in inputCaps:
        inputCaps[k].release()
    outputF.release()


def makeNaiveVideoSlidingWindow(outputFilename, vidWavPairs, windowLength):
    ''' Creates an avi from a list of [(vidFilename, wavFilename)] '''
    highlights = getHighlightFrames(vidWavPairs)
    videoFPS = 25
    
    tempCap = cv2.VideoCapture(vidWavPairs[0][0])
    ret,frame = tempCap.read()
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outputF = cv2.VideoWriter(outputFilename, fourcc, videoFPS, (width,height))

    inputCaps = {}
    windowCounts = {}
    for (vidFilename, wavFilename) in vidWavPairs:
        cap = cv2.VideoCapture(vidFilename)
        inputCaps[vidFilename] = cap
        windowCounts[vidFilename] = 0

    
    for (i, vidFilename) in highlights:
        ret, frame = (False, [0])
        windowCounts[vidFilename] += 1
        if i > windowLength:
            windowCounts[highlights[i-windowLength][1]] -= 1

        # Get video with highest count in the window
        vidFilename = max([(windowCounts[k], k) for k in windowCounts])[1]
        for k in inputCaps:
            if k == vidFilename:
                ret, frame = inputCaps[k].read()
            else:
                a,b = inputCaps[k].read()
            
        outputF.write(frame)
            
    for k in inputCaps:
        inputCaps[k].release()
    outputF.release()


def makeNaiveVideoSlidingWindowNoThrash(outputFilename, vidWavPairs,
                                        windowLength, thrashFrames=25):
    ''' Creates an avi from a list of [(vidFilename, wavFilename)] '''
    highlights = getHighlightFrames(vidWavPairs)
    videoFPS = 25
    
    tempCap = cv2.VideoCapture(vidWavPairs[0][0])
    ret,frame = tempCap.read()
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outputF = cv2.VideoWriter(outputFilename, fourcc, videoFPS, (width,height))

    inputCaps = {}
    windowCounts = {}
    for (vidFilename, wavFilename) in vidWavPairs:
        cap = cv2.VideoCapture(vidFilename)
        inputCaps[vidFilename] = cap
        windowCounts[vidFilename] = 0

    
    lastChange = -2 * thrashFrames
    lastFilename = ''
    for (i, vidFilename) in highlights:
        ret, frame = (False, [0])
        windowCounts[vidFilename] += 1
        if i > windowLength:
            windowCounts[highlights[i-windowLength][1]] -= 1

        # Get video with highest count in the window
        vidFilename = max([(windowCounts[k], k) for k in windowCounts])[1]
        if vidFilename != lastFilename:
            if i - lastChange > thrashFrames:
                lastChange = i
                lastFilename = vidFilename
            else:
                vidFilename = lastFilename
        for k in inputCaps:
            if k == vidFilename:
                ret, frame = inputCaps[k].read()
            else:
                a,b = inputCaps[k].read()
           
        outputF.write(frame)
            
    for k in inputCaps:
        inputCaps[k].release()
    outputF.release()


def makeNaiveVideoFromHighlights(outputFilename, vidWavPairs, highlights):
    ''' Creates an avi from a list of [(vidFilename, wavFilename)] '''
    videoFPS = 25.0
    
    tempCap = cv2.VideoCapture(vidWavPairs[0][0])
    ret,frame = tempCap.read()
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outputF = cv2.VideoWriter(outputFilename, fourcc, videoFPS, (width,height))

    inputCaps = {}
    for (vidFilename, wavFilename) in vidWavPairs:
        cap = cv2.VideoCapture(vidFilename)
        inputCaps[vidFilename] = cap

    for (i, vidFilename) in highlights:
        ret, frame = (False, [0])
        for k in inputCaps:
            if k == vidFilename:
                ret, frame = inputCaps[k].read()
            else:
                a,b = inputCaps[k].read()
        
        outputF.write(frame)
            
    for k in inputCaps:
        inputCaps[k].release()
    outputF.release()

