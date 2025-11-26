#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Sun Nov 23 19:01:29 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'solo project'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/niteshshah/solo project_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('donekey') is None:
        # initialise donekey
        donekey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='donekey',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "studyroutine" ---
    bgRect = visual.Rect(
        win=win, name='bgRect',
        width=[2, 2][0], height=[2, 2][1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    studyWord = visual.TextStim(win=win, name='studyWord',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.14, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "studyroutine" ---
    bgRect = visual.Rect(
        win=win, name='bgRect',
        width=[2, 2][0], height=[2, 2][1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    studyWord = visual.TextStim(win=win, name='studyWord',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.14, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "studyroutine" ---
    bgRect = visual.Rect(
        win=win, name='bgRect',
        width=[2, 2][0], height=[2, 2][1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    studyWord = visual.TextStim(win=win, name='studyWord',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.14, wrapWidth=1.8, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "recallitem_" ---
    maskedWord = visual.TextStim(win=win, name='maskedWord',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    responsebox = visual.TextBox2(
         win, text=None, placeholder='type the word here', font='Arial',
         ori=0.0, pos=(0,-0.2), draggable=False,      letterHeight=0.05,
         size=(0.8,0.2), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='responsebox',
         depth=-1, autoLog=True,
    )
    donekey = keyboard.Keyboard(deviceName='donekey')
    
    # --- Initialize components for Routine "_endRoutine_" ---
    thanks = visual.TextStim(win=win, name='thanks',
        text='Thank you for participating!\nYou may now close the experiment window.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    RedListLoop = data.TrialHandler2(
        name='RedListLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Library/Containers/net.whatsapp.WhatsApp/Data/red_color_csv.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(RedListLoop)  # add the loop to the experiment
    thisRedListLoop = RedListLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRedListLoop.rgb)
    if thisRedListLoop != None:
        for paramName in thisRedListLoop:
            globals()[paramName] = thisRedListLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRedListLoop in RedListLoop:
        RedListLoop.status = STARTED
        if hasattr(thisRedListLoop, 'status'):
            thisRedListLoop.status = STARTED
        currentLoop = RedListLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRedListLoop.rgb)
        if thisRedListLoop != None:
            for paramName in thisRedListLoop:
                globals()[paramName] = thisRedListLoop[paramName]
        
        # --- Prepare to start Routine "studyroutine" ---
        # create an object to store info about Routine studyroutine
        studyroutine = data.Routine(
            name='studyroutine',
            components=[bgRect, studyWord],
        )
        studyroutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        bgRect.setFillColor(color_hex )
        bgRect.setLineColor(color_hex )
        studyWord.setColor(text_color, colorSpace='rgb')
        studyWord.setText(word
        )
        # store start times for studyroutine
        studyroutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        studyroutine.tStart = globalClock.getTime(format='float')
        studyroutine.status = STARTED
        thisExp.addData('studyroutine.started', studyroutine.tStart)
        studyroutine.maxDuration = None
        # keep track of which components have finished
        studyroutineComponents = studyroutine.components
        for thisComponent in studyroutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "studyroutine" ---
        studyroutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisRedListLoop, 'status') and thisRedListLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgRect* updates
            
            # if bgRect is starting this frame...
            if bgRect.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgRect.frameNStart = frameN  # exact frame index
                bgRect.tStart = t  # local t and not account for scr refresh
                bgRect.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgRect, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgRect.started')
                # update status
                bgRect.status = STARTED
                bgRect.setAutoDraw(True)
            
            # if bgRect is active this frame...
            if bgRect.status == STARTED:
                # update params
                pass
            
            # if bgRect is stopping this frame...
            if bgRect.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgRect.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    bgRect.tStop = t  # not accounting for scr refresh
                    bgRect.tStopRefresh = tThisFlipGlobal  # on global time
                    bgRect.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgRect.stopped')
                    # update status
                    bgRect.status = FINISHED
                    bgRect.setAutoDraw(False)
            
            # *studyWord* updates
            
            # if studyWord is starting this frame...
            if studyWord.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                studyWord.frameNStart = frameN  # exact frame index
                studyWord.tStart = t  # local t and not account for scr refresh
                studyWord.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(studyWord, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'studyWord.started')
                # update status
                studyWord.status = STARTED
                studyWord.setAutoDraw(True)
            
            # if studyWord is active this frame...
            if studyWord.status == STARTED:
                # update params
                pass
            
            # if studyWord is stopping this frame...
            if studyWord.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > studyWord.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    studyWord.tStop = t  # not accounting for scr refresh
                    studyWord.tStopRefresh = tThisFlipGlobal  # on global time
                    studyWord.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'studyWord.stopped')
                    # update status
                    studyWord.status = FINISHED
                    studyWord.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=studyroutine,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                studyroutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in studyroutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "studyroutine" ---
        for thisComponent in studyroutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for studyroutine
        studyroutine.tStop = globalClock.getTime(format='float')
        studyroutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('studyroutine.stopped', studyroutine.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if studyroutine.maxDurationReached:
            routineTimer.addTime(-studyroutine.maxDuration)
        elif studyroutine.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        # mark thisRedListLoop as finished
        if hasattr(thisRedListLoop, 'status'):
            thisRedListLoop.status = FINISHED
        # if awaiting a pause, pause now
        if RedListLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            RedListLoop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'RedListLoop'
    RedListLoop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    BlueListLoop = data.TrialHandler2(
        name='BlueListLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Library/Containers/net.whatsapp.WhatsApp/Data/blue_color_csv.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(BlueListLoop)  # add the loop to the experiment
    thisBlueListLoop = BlueListLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlueListLoop.rgb)
    if thisBlueListLoop != None:
        for paramName in thisBlueListLoop:
            globals()[paramName] = thisBlueListLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlueListLoop in BlueListLoop:
        BlueListLoop.status = STARTED
        if hasattr(thisBlueListLoop, 'status'):
            thisBlueListLoop.status = STARTED
        currentLoop = BlueListLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlueListLoop.rgb)
        if thisBlueListLoop != None:
            for paramName in thisBlueListLoop:
                globals()[paramName] = thisBlueListLoop[paramName]
        
        # --- Prepare to start Routine "studyroutine" ---
        # create an object to store info about Routine studyroutine
        studyroutine = data.Routine(
            name='studyroutine',
            components=[bgRect, studyWord],
        )
        studyroutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        bgRect.setFillColor(color_hex )
        bgRect.setLineColor(color_hex )
        studyWord.setColor(text_color, colorSpace='rgb')
        studyWord.setText(word
        )
        # store start times for studyroutine
        studyroutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        studyroutine.tStart = globalClock.getTime(format='float')
        studyroutine.status = STARTED
        thisExp.addData('studyroutine.started', studyroutine.tStart)
        studyroutine.maxDuration = None
        # keep track of which components have finished
        studyroutineComponents = studyroutine.components
        for thisComponent in studyroutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "studyroutine" ---
        studyroutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisBlueListLoop, 'status') and thisBlueListLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgRect* updates
            
            # if bgRect is starting this frame...
            if bgRect.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgRect.frameNStart = frameN  # exact frame index
                bgRect.tStart = t  # local t and not account for scr refresh
                bgRect.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgRect, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgRect.started')
                # update status
                bgRect.status = STARTED
                bgRect.setAutoDraw(True)
            
            # if bgRect is active this frame...
            if bgRect.status == STARTED:
                # update params
                pass
            
            # if bgRect is stopping this frame...
            if bgRect.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgRect.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    bgRect.tStop = t  # not accounting for scr refresh
                    bgRect.tStopRefresh = tThisFlipGlobal  # on global time
                    bgRect.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgRect.stopped')
                    # update status
                    bgRect.status = FINISHED
                    bgRect.setAutoDraw(False)
            
            # *studyWord* updates
            
            # if studyWord is starting this frame...
            if studyWord.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                studyWord.frameNStart = frameN  # exact frame index
                studyWord.tStart = t  # local t and not account for scr refresh
                studyWord.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(studyWord, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'studyWord.started')
                # update status
                studyWord.status = STARTED
                studyWord.setAutoDraw(True)
            
            # if studyWord is active this frame...
            if studyWord.status == STARTED:
                # update params
                pass
            
            # if studyWord is stopping this frame...
            if studyWord.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > studyWord.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    studyWord.tStop = t  # not accounting for scr refresh
                    studyWord.tStopRefresh = tThisFlipGlobal  # on global time
                    studyWord.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'studyWord.stopped')
                    # update status
                    studyWord.status = FINISHED
                    studyWord.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=studyroutine,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                studyroutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in studyroutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "studyroutine" ---
        for thisComponent in studyroutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for studyroutine
        studyroutine.tStop = globalClock.getTime(format='float')
        studyroutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('studyroutine.stopped', studyroutine.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if studyroutine.maxDurationReached:
            routineTimer.addTime(-studyroutine.maxDuration)
        elif studyroutine.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        # mark thisBlueListLoop as finished
        if hasattr(thisBlueListLoop, 'status'):
            thisBlueListLoop.status = FINISHED
        # if awaiting a pause, pause now
        if BlueListLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            BlueListLoop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'BlueListLoop'
    BlueListLoop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    GreyListLoop = data.TrialHandler2(
        name='GreyListLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Library/Containers/net.whatsapp.WhatsApp/Data/grey_color_csv.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(GreyListLoop)  # add the loop to the experiment
    thisGreyListLoop = GreyListLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGreyListLoop.rgb)
    if thisGreyListLoop != None:
        for paramName in thisGreyListLoop:
            globals()[paramName] = thisGreyListLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisGreyListLoop in GreyListLoop:
        GreyListLoop.status = STARTED
        if hasattr(thisGreyListLoop, 'status'):
            thisGreyListLoop.status = STARTED
        currentLoop = GreyListLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisGreyListLoop.rgb)
        if thisGreyListLoop != None:
            for paramName in thisGreyListLoop:
                globals()[paramName] = thisGreyListLoop[paramName]
        
        # --- Prepare to start Routine "studyroutine" ---
        # create an object to store info about Routine studyroutine
        studyroutine = data.Routine(
            name='studyroutine',
            components=[bgRect, studyWord],
        )
        studyroutine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        bgRect.setFillColor(color_hex )
        bgRect.setLineColor(color_hex )
        studyWord.setColor(text_color, colorSpace='rgb')
        studyWord.setText(word
        )
        # store start times for studyroutine
        studyroutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        studyroutine.tStart = globalClock.getTime(format='float')
        studyroutine.status = STARTED
        thisExp.addData('studyroutine.started', studyroutine.tStart)
        studyroutine.maxDuration = None
        # keep track of which components have finished
        studyroutineComponents = studyroutine.components
        for thisComponent in studyroutine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "studyroutine" ---
        studyroutine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisGreyListLoop, 'status') and thisGreyListLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *bgRect* updates
            
            # if bgRect is starting this frame...
            if bgRect.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                bgRect.frameNStart = frameN  # exact frame index
                bgRect.tStart = t  # local t and not account for scr refresh
                bgRect.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(bgRect, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bgRect.started')
                # update status
                bgRect.status = STARTED
                bgRect.setAutoDraw(True)
            
            # if bgRect is active this frame...
            if bgRect.status == STARTED:
                # update params
                pass
            
            # if bgRect is stopping this frame...
            if bgRect.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > bgRect.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    bgRect.tStop = t  # not accounting for scr refresh
                    bgRect.tStopRefresh = tThisFlipGlobal  # on global time
                    bgRect.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bgRect.stopped')
                    # update status
                    bgRect.status = FINISHED
                    bgRect.setAutoDraw(False)
            
            # *studyWord* updates
            
            # if studyWord is starting this frame...
            if studyWord.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                studyWord.frameNStart = frameN  # exact frame index
                studyWord.tStart = t  # local t and not account for scr refresh
                studyWord.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(studyWord, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'studyWord.started')
                # update status
                studyWord.status = STARTED
                studyWord.setAutoDraw(True)
            
            # if studyWord is active this frame...
            if studyWord.status == STARTED:
                # update params
                pass
            
            # if studyWord is stopping this frame...
            if studyWord.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > studyWord.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    studyWord.tStop = t  # not accounting for scr refresh
                    studyWord.tStopRefresh = tThisFlipGlobal  # on global time
                    studyWord.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'studyWord.stopped')
                    # update status
                    studyWord.status = FINISHED
                    studyWord.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=studyroutine,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                studyroutine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in studyroutine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "studyroutine" ---
        for thisComponent in studyroutine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for studyroutine
        studyroutine.tStop = globalClock.getTime(format='float')
        studyroutine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('studyroutine.stopped', studyroutine.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if studyroutine.maxDurationReached:
            routineTimer.addTime(-studyroutine.maxDuration)
        elif studyroutine.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        # mark thisGreyListLoop as finished
        if hasattr(thisGreyListLoop, 'status'):
            thisGreyListLoop.status = FINISHED
        # if awaiting a pause, pause now
        if GreyListLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            GreyListLoop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'GreyListLoop'
    GreyListLoop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    recallloop = data.TrialHandler2(
        name='recallloop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Library/Containers/net.whatsapp.WhatsApp/Data/recall_list.csv.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(recallloop)  # add the loop to the experiment
    thisRecallloop = recallloop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecallloop.rgb)
    if thisRecallloop != None:
        for paramName in thisRecallloop:
            globals()[paramName] = thisRecallloop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecallloop in recallloop:
        recallloop.status = STARTED
        if hasattr(thisRecallloop, 'status'):
            thisRecallloop.status = STARTED
        currentLoop = recallloop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecallloop.rgb)
        if thisRecallloop != None:
            for paramName in thisRecallloop:
                globals()[paramName] = thisRecallloop[paramName]
        
        # --- Prepare to start Routine "recallitem_" ---
        # create an object to store info about Routine recallitem_
        recallitem_ = data.Routine(
            name='recallitem_',
            components=[maskedWord, responsebox, donekey],
        )
        recallitem_.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        maskedWord.setText(masked)
        responsebox.reset()
        # create starting attributes for donekey
        donekey.keys = []
        donekey.rt = []
        _donekey_allKeys = []
        # store start times for recallitem_
        recallitem_.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recallitem_.tStart = globalClock.getTime(format='float')
        recallitem_.status = STARTED
        thisExp.addData('recallitem_.started', recallitem_.tStart)
        recallitem_.maxDuration = None
        # keep track of which components have finished
        recallitem_Components = recallitem_.components
        for thisComponent in recallitem_.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recallitem_" ---
        recallitem_.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRecallloop, 'status') and thisRecallloop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *maskedWord* updates
            
            # if maskedWord is starting this frame...
            if maskedWord.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                maskedWord.frameNStart = frameN  # exact frame index
                maskedWord.tStart = t  # local t and not account for scr refresh
                maskedWord.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(maskedWord, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'maskedWord.started')
                # update status
                maskedWord.status = STARTED
                maskedWord.setAutoDraw(True)
            
            # if maskedWord is active this frame...
            if maskedWord.status == STARTED:
                # update params
                pass
            
            # *responsebox* updates
            
            # if responsebox is starting this frame...
            if responsebox.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                responsebox.frameNStart = frameN  # exact frame index
                responsebox.tStart = t  # local t and not account for scr refresh
                responsebox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(responsebox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'responsebox.started')
                # update status
                responsebox.status = STARTED
                responsebox.setAutoDraw(True)
            
            # if responsebox is active this frame...
            if responsebox.status == STARTED:
                # update params
                pass
            
            # *donekey* updates
            waitOnFlip = False
            
            # if donekey is starting this frame...
            if donekey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                donekey.frameNStart = frameN  # exact frame index
                donekey.tStart = t  # local t and not account for scr refresh
                donekey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(donekey, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'donekey.started')
                # update status
                donekey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(donekey.clock.reset)  # t=0 on next screen flip
            if donekey.status == STARTED and not waitOnFlip:
                theseKeys = donekey.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _donekey_allKeys.extend(theseKeys)
                if len(_donekey_allKeys):
                    donekey.keys = _donekey_allKeys[-1].name  # just the last key pressed
                    donekey.rt = _donekey_allKeys[-1].rt
                    donekey.duration = _donekey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recallitem_,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recallitem_.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recallitem_.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recallitem_" ---
        for thisComponent in recallitem_.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recallitem_
        recallitem_.tStop = globalClock.getTime(format='float')
        recallitem_.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recallitem_.stopped', recallitem_.tStop)
        recallloop.addData('responsebox.text',responsebox.text)
        # check responses
        if donekey.keys in ['', [], None]:  # No response was made
            donekey.keys = None
        recallloop.addData('donekey.keys',donekey.keys)
        if donekey.keys != None:  # we had a response
            recallloop.addData('donekey.rt', donekey.rt)
            recallloop.addData('donekey.duration', donekey.duration)
        # the Routine "recallitem_" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRecallloop as finished
        if hasattr(thisRecallloop, 'status'):
            thisRecallloop.status = FINISHED
        # if awaiting a pause, pause now
        if recallloop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            recallloop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'recallloop'
    recallloop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "_endRoutine_" ---
    # create an object to store info about Routine _endRoutine_
    _endRoutine_ = data.Routine(
        name='_endRoutine_',
        components=[thanks],
    )
    _endRoutine_.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for _endRoutine_
    _endRoutine_.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    _endRoutine_.tStart = globalClock.getTime(format='float')
    _endRoutine_.status = STARTED
    thisExp.addData('_endRoutine_.started', _endRoutine_.tStart)
    _endRoutine_.maxDuration = None
    # keep track of which components have finished
    _endRoutine_Components = _endRoutine_.components
    for thisComponent in _endRoutine_.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "_endRoutine_" ---
    _endRoutine_.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks* updates
        
        # if thanks is starting this frame...
        if thanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks.frameNStart = frameN  # exact frame index
            thanks.tStart = t  # local t and not account for scr refresh
            thanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanks.started')
            # update status
            thanks.status = STARTED
            thanks.setAutoDraw(True)
        
        # if thanks is active this frame...
        if thanks.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=_endRoutine_,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            _endRoutine_.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in _endRoutine_.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "_endRoutine_" ---
    for thisComponent in _endRoutine_.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for _endRoutine_
    _endRoutine_.tStop = globalClock.getTime(format='float')
    _endRoutine_.tStopRefresh = tThisFlipGlobal
    thisExp.addData('_endRoutine_.stopped', _endRoutine_.tStop)
    thisExp.nextEntry()
    # the Routine "_endRoutine_" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
