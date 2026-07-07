#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.0),
    on July 07, 2026, at 13:12
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

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.0'
expName = 'Maryam_Gholipour_Male'  # from the Builder filename that created this script
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
        originPath='G:\\Ghazaleh Khosravi\\Task Design Psychopy\\VPT_Farsi_github\\Female_Task\\Code\\Maryam_Gholipour_Male.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
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
            winType='pyglet', allowGUI=False, allowStencil=False,
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
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
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
            backend='ioHub',
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
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
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
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
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
    
    # --- Initialize components for Routine "instruct_1" ---
    Pilot_Background = visual.Rect(
        win=win, name='Pilot_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Press_Space = visual.ImageStim(
        win=win,
        name='Press_Space', 
        image='instruct_1.PNG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Pilot_Resp_Space = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instruct_2" ---
    Pilot_Background_2 = visual.Rect(
        win=win, name='Pilot_Background_2',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Press_Space_3 = visual.ImageStim(
        win=win,
        name='Press_Space_3', 
        image='instruct_2.PNG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.92, 0.85),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Pilot_Resp_Space_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instruct_3" ---
    Pilot_Background_3 = visual.Rect(
        win=win, name='Pilot_Background_3',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Press_Space_4 = visual.ImageStim(
        win=win,
        name='Press_Space_4', 
        image='instruct_3.PNG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.7, 0.7),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Pilot_Resp_Space_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Fix_Cross" ---
    Background = visual.Rect(
        win=win, name='Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Fix_cross = visual.ShapeStim(
        win=win, name='Fix_cross', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "Blank_Page" ---
    Blank_Background = visual.Rect(
        win=win, name='Blank_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Perspective_cue" ---
    Cue_Persp_Background = visual.Rect(
        win=win, name='Cue_Persp_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Perspective_Cue2 = visual.ImageStim(
        win=win,
        name='Perspective_Cue2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "Blank_Page" ---
    Blank_Background = visual.Rect(
        win=win, name='Blank_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Room" ---
    Room_Background = visual.Rect(
        win=win, name='Room_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='gray', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=1.0, depth=0.0, interpolate=True)
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "feedback" ---
    FB_Background = visual.Rect(
        win=win, name='FB_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    feedback_text = visual.TextStim(win=win, name='feedback_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=1.0, 
        languageStyle='Arabic',
        depth=-1.0);
    
    # --- Initialize components for Routine "Experiment_section" ---
    Exp_Background = visual.Rect(
        win=win, name='Exp_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Press_Space_2 = visual.ImageStim(
        win=win,
        name='Press_Space_2', 
        image='Space.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Exp_Resp_Space = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Fix_Cross" ---
    Background = visual.Rect(
        win=win, name='Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Fix_cross = visual.ShapeStim(
        win=win, name='Fix_cross', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='black', fillColor='black',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "Blank_Page" ---
    Blank_Background = visual.Rect(
        win=win, name='Blank_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Perspective_cue" ---
    Cue_Persp_Background = visual.Rect(
        win=win, name='Cue_Persp_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    Perspective_Cue2 = visual.ImageStim(
        win=win,
        name='Perspective_Cue2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "Blank_Page" ---
    Blank_Background = visual.Rect(
        win=win, name='Blank_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Exp_Room" ---
    Room_Back = visual.ShapeStim(
        win=win, name='Room_Back',
        size=(2, 2), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='gray', fillColor='gray',
        opacity=1.0, depth=0.0, interpolate=True)
    Image = visual.ImageStim(
        win=win,
        name='Image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "End" ---
    End_Background = visual.Rect(
        win=win, name='End_Background',
        width=(2, 2)[0], height=(2, 2)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='end.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
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
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instruct_1" ---
    # create an object to store info about Routine instruct_1
    instruct_1 = data.Routine(
        name='instruct_1',
        components=[Pilot_Background, Press_Space, Pilot_Resp_Space],
    )
    instruct_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Pilot_Resp_Space
    Pilot_Resp_Space.keys = []
    Pilot_Resp_Space.rt = []
    _Pilot_Resp_Space_allKeys = []
    # store start times for instruct_1
    instruct_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_1.tStart = globalClock.getTime(format='float')
    instruct_1.status = STARTED
    thisExp.addData('instruct_1.started', instruct_1.tStart)
    instruct_1.maxDuration = None
    # keep track of which components have finished
    instruct_1Components = instruct_1.components
    for thisComponent in instruct_1.components:
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
    
    # --- Run Routine "instruct_1" ---
    thisExp.currentRoutine = instruct_1
    instruct_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Pilot_Background* updates
        
        # if Pilot_Background is starting this frame...
        if Pilot_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Pilot_Background.frameNStart = frameN  # exact frame index
            Pilot_Background.tStart = t  # local t and not account for scr refresh
            Pilot_Background.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pilot_Background, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pilot_Background.started')
            # update status
            Pilot_Background.status = STARTED
            Pilot_Background.setAutoDraw(True)
        
        # if Pilot_Background is active this frame...
        if Pilot_Background.status == STARTED:
            # update params
            pass
        
        # *Press_Space* updates
        
        # if Press_Space is starting this frame...
        if Press_Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Press_Space.frameNStart = frameN  # exact frame index
            Press_Space.tStart = t  # local t and not account for scr refresh
            Press_Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Press_Space, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Press_Space.started')
            # update status
            Press_Space.status = STARTED
            Press_Space.setAutoDraw(True)
        
        # if Press_Space is active this frame...
        if Press_Space.status == STARTED:
            # update params
            pass
        
        # *Pilot_Resp_Space* updates
        waitOnFlip = False
        
        # if Pilot_Resp_Space is starting this frame...
        if Pilot_Resp_Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Pilot_Resp_Space.frameNStart = frameN  # exact frame index
            Pilot_Resp_Space.tStart = t  # local t and not account for scr refresh
            Pilot_Resp_Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pilot_Resp_Space, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pilot_Resp_Space.started')
            # update status
            Pilot_Resp_Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Pilot_Resp_Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Pilot_Resp_Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Pilot_Resp_Space.status == STARTED and not waitOnFlip:
            theseKeys = Pilot_Resp_Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Pilot_Resp_Space_allKeys.extend(theseKeys)
            if len(_Pilot_Resp_Space_allKeys):
                Pilot_Resp_Space.keys = _Pilot_Resp_Space_allKeys[-1].name  # just the last key pressed
                Pilot_Resp_Space.rt = _Pilot_Resp_Space_allKeys[-1].rt
                Pilot_Resp_Space.duration = _Pilot_Resp_Space_allKeys[-1].duration
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
                currentRoutine=instruct_1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruct_1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruct_1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruct_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_1" ---
    for thisComponent in instruct_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_1
    instruct_1.tStop = globalClock.getTime(format='float')
    instruct_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruct_1.stopped', instruct_1.tStop)
    # check responses
    if Pilot_Resp_Space.keys in ['', [], None]:  # No response was made
        Pilot_Resp_Space.keys = None
    thisExp.addData('Pilot_Resp_Space.keys',Pilot_Resp_Space.keys)
    if Pilot_Resp_Space.keys != None:  # we had a response
        thisExp.addData('Pilot_Resp_Space.rt', Pilot_Resp_Space.rt)
        thisExp.addData('Pilot_Resp_Space.duration', Pilot_Resp_Space.duration)
    thisExp.nextEntry()
    # the Routine "instruct_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_2" ---
    # create an object to store info about Routine instruct_2
    instruct_2 = data.Routine(
        name='instruct_2',
        components=[Pilot_Background_2, Press_Space_3, Pilot_Resp_Space_2],
    )
    instruct_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Pilot_Resp_Space_2
    Pilot_Resp_Space_2.keys = []
    Pilot_Resp_Space_2.rt = []
    _Pilot_Resp_Space_2_allKeys = []
    # store start times for instruct_2
    instruct_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_2.tStart = globalClock.getTime(format='float')
    instruct_2.status = STARTED
    thisExp.addData('instruct_2.started', instruct_2.tStart)
    instruct_2.maxDuration = None
    # keep track of which components have finished
    instruct_2Components = instruct_2.components
    for thisComponent in instruct_2.components:
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
    
    # --- Run Routine "instruct_2" ---
    thisExp.currentRoutine = instruct_2
    instruct_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Pilot_Background_2* updates
        
        # if Pilot_Background_2 is starting this frame...
        if Pilot_Background_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Pilot_Background_2.frameNStart = frameN  # exact frame index
            Pilot_Background_2.tStart = t  # local t and not account for scr refresh
            Pilot_Background_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pilot_Background_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pilot_Background_2.started')
            # update status
            Pilot_Background_2.status = STARTED
            Pilot_Background_2.setAutoDraw(True)
        
        # if Pilot_Background_2 is active this frame...
        if Pilot_Background_2.status == STARTED:
            # update params
            pass
        
        # *Press_Space_3* updates
        
        # if Press_Space_3 is starting this frame...
        if Press_Space_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Press_Space_3.frameNStart = frameN  # exact frame index
            Press_Space_3.tStart = t  # local t and not account for scr refresh
            Press_Space_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Press_Space_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Press_Space_3.started')
            # update status
            Press_Space_3.status = STARTED
            Press_Space_3.setAutoDraw(True)
        
        # if Press_Space_3 is active this frame...
        if Press_Space_3.status == STARTED:
            # update params
            pass
        
        # *Pilot_Resp_Space_2* updates
        waitOnFlip = False
        
        # if Pilot_Resp_Space_2 is starting this frame...
        if Pilot_Resp_Space_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Pilot_Resp_Space_2.frameNStart = frameN  # exact frame index
            Pilot_Resp_Space_2.tStart = t  # local t and not account for scr refresh
            Pilot_Resp_Space_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pilot_Resp_Space_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pilot_Resp_Space_2.started')
            # update status
            Pilot_Resp_Space_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Pilot_Resp_Space_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Pilot_Resp_Space_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Pilot_Resp_Space_2.status == STARTED and not waitOnFlip:
            theseKeys = Pilot_Resp_Space_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Pilot_Resp_Space_2_allKeys.extend(theseKeys)
            if len(_Pilot_Resp_Space_2_allKeys):
                Pilot_Resp_Space_2.keys = _Pilot_Resp_Space_2_allKeys[-1].name  # just the last key pressed
                Pilot_Resp_Space_2.rt = _Pilot_Resp_Space_2_allKeys[-1].rt
                Pilot_Resp_Space_2.duration = _Pilot_Resp_Space_2_allKeys[-1].duration
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
                currentRoutine=instruct_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruct_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruct_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruct_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_2" ---
    for thisComponent in instruct_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_2
    instruct_2.tStop = globalClock.getTime(format='float')
    instruct_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruct_2.stopped', instruct_2.tStop)
    # check responses
    if Pilot_Resp_Space_2.keys in ['', [], None]:  # No response was made
        Pilot_Resp_Space_2.keys = None
    thisExp.addData('Pilot_Resp_Space_2.keys',Pilot_Resp_Space_2.keys)
    if Pilot_Resp_Space_2.keys != None:  # we had a response
        thisExp.addData('Pilot_Resp_Space_2.rt', Pilot_Resp_Space_2.rt)
        thisExp.addData('Pilot_Resp_Space_2.duration', Pilot_Resp_Space_2.duration)
    thisExp.nextEntry()
    # the Routine "instruct_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_3" ---
    # create an object to store info about Routine instruct_3
    instruct_3 = data.Routine(
        name='instruct_3',
        components=[Pilot_Background_3, Press_Space_4, Pilot_Resp_Space_3],
    )
    instruct_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Pilot_Resp_Space_3
    Pilot_Resp_Space_3.keys = []
    Pilot_Resp_Space_3.rt = []
    _Pilot_Resp_Space_3_allKeys = []
    # store start times for instruct_3
    instruct_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruct_3.tStart = globalClock.getTime(format='float')
    instruct_3.status = STARTED
    thisExp.addData('instruct_3.started', instruct_3.tStart)
    instruct_3.maxDuration = None
    # keep track of which components have finished
    instruct_3Components = instruct_3.components
    for thisComponent in instruct_3.components:
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
    
    # --- Run Routine "instruct_3" ---
    thisExp.currentRoutine = instruct_3
    instruct_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Pilot_Background_3* updates
        
        # if Pilot_Background_3 is starting this frame...
        if Pilot_Background_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Pilot_Background_3.frameNStart = frameN  # exact frame index
            Pilot_Background_3.tStart = t  # local t and not account for scr refresh
            Pilot_Background_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pilot_Background_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pilot_Background_3.started')
            # update status
            Pilot_Background_3.status = STARTED
            Pilot_Background_3.setAutoDraw(True)
        
        # if Pilot_Background_3 is active this frame...
        if Pilot_Background_3.status == STARTED:
            # update params
            pass
        
        # *Press_Space_4* updates
        
        # if Press_Space_4 is starting this frame...
        if Press_Space_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Press_Space_4.frameNStart = frameN  # exact frame index
            Press_Space_4.tStart = t  # local t and not account for scr refresh
            Press_Space_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Press_Space_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Press_Space_4.started')
            # update status
            Press_Space_4.status = STARTED
            Press_Space_4.setAutoDraw(True)
        
        # if Press_Space_4 is active this frame...
        if Press_Space_4.status == STARTED:
            # update params
            pass
        
        # *Pilot_Resp_Space_3* updates
        waitOnFlip = False
        
        # if Pilot_Resp_Space_3 is starting this frame...
        if Pilot_Resp_Space_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Pilot_Resp_Space_3.frameNStart = frameN  # exact frame index
            Pilot_Resp_Space_3.tStart = t  # local t and not account for scr refresh
            Pilot_Resp_Space_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Pilot_Resp_Space_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Pilot_Resp_Space_3.started')
            # update status
            Pilot_Resp_Space_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Pilot_Resp_Space_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Pilot_Resp_Space_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Pilot_Resp_Space_3.status == STARTED and not waitOnFlip:
            theseKeys = Pilot_Resp_Space_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Pilot_Resp_Space_3_allKeys.extend(theseKeys)
            if len(_Pilot_Resp_Space_3_allKeys):
                Pilot_Resp_Space_3.keys = _Pilot_Resp_Space_3_allKeys[-1].name  # just the last key pressed
                Pilot_Resp_Space_3.rt = _Pilot_Resp_Space_3_allKeys[-1].rt
                Pilot_Resp_Space_3.duration = _Pilot_Resp_Space_3_allKeys[-1].duration
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
                currentRoutine=instruct_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruct_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruct_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruct_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_3" ---
    for thisComponent in instruct_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruct_3
    instruct_3.tStop = globalClock.getTime(format='float')
    instruct_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruct_3.stopped', instruct_3.tStop)
    # check responses
    if Pilot_Resp_Space_3.keys in ['', [], None]:  # No response was made
        Pilot_Resp_Space_3.keys = None
    thisExp.addData('Pilot_Resp_Space_3.keys',Pilot_Resp_Space_3.keys)
    if Pilot_Resp_Space_3.keys != None:  # we had a response
        thisExp.addData('Pilot_Resp_Space_3.rt', Pilot_Resp_Space_3.rt)
        thisExp.addData('Pilot_Resp_Space_3.duration', Pilot_Resp_Space_3.duration)
    thisExp.nextEntry()
    # the Routine "instruct_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Practice = data.TrialHandler2(
        name='Practice',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Female_pilot.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(Practice)  # add the loop to the experiment
    thisPractice = Practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
    if thisPractice != None:
        for paramName in thisPractice:
            globals()[paramName] = thisPractice[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice in Practice:
        Practice.status = STARTED
        if hasattr(thisPractice, 'status'):
            thisPractice.status = STARTED
        currentLoop = Practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        
        # --- Prepare to start Routine "Fix_Cross" ---
        # create an object to store info about Routine Fix_Cross
        Fix_Cross = data.Routine(
            name='Fix_Cross',
            components=[Background, Fix_cross],
        )
        Fix_Cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fix_Cross
        Fix_Cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fix_Cross.tStart = globalClock.getTime(format='float')
        Fix_Cross.status = STARTED
        thisExp.addData('Fix_Cross.started', Fix_Cross.tStart)
        Fix_Cross.maxDuration = None
        # keep track of which components have finished
        Fix_CrossComponents = Fix_Cross.components
        for thisComponent in Fix_Cross.components:
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
        
        # --- Run Routine "Fix_Cross" ---
        thisExp.currentRoutine = Fix_Cross
        Fix_Cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.75:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Background* updates
            
            # if Background is starting this frame...
            if Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Background.frameNStart = frameN  # exact frame index
                Background.tStart = t  # local t and not account for scr refresh
                Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Background.started')
                # update status
                Background.status = STARTED
                Background.setAutoDraw(True)
            
            # if Background is active this frame...
            if Background.status == STARTED:
                # update params
                pass
            
            # if Background is stopping this frame...
            if Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Background.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Background.tStop = t  # not accounting for scr refresh
                    Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Background.stopped')
                    # update status
                    Background.status = FINISHED
                    Background.setAutoDraw(False)
            
            # *Fix_cross* updates
            
            # if Fix_cross is starting this frame...
            if Fix_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Fix_cross.frameNStart = frameN  # exact frame index
                Fix_cross.tStart = t  # local t and not account for scr refresh
                Fix_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Fix_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Fix_cross.started')
                # update status
                Fix_cross.status = STARTED
                Fix_cross.setAutoDraw(True)
            
            # if Fix_cross is active this frame...
            if Fix_cross.status == STARTED:
                # update params
                pass
            
            # if Fix_cross is stopping this frame...
            if Fix_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Fix_cross.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Fix_cross.tStop = t  # not accounting for scr refresh
                    Fix_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    Fix_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fix_cross.stopped')
                    # update status
                    Fix_cross.status = FINISHED
                    Fix_cross.setAutoDraw(False)
            
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
                    currentRoutine=Fix_Cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Fix_Cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Fix_Cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Fix_Cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fix_Cross" ---
        for thisComponent in Fix_Cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fix_Cross
        Fix_Cross.tStop = globalClock.getTime(format='float')
        Fix_Cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fix_Cross.stopped', Fix_Cross.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fix_Cross.maxDurationReached:
            routineTimer.addTime(-Fix_Cross.maxDuration)
        elif Fix_Cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.750000)
        
        # --- Prepare to start Routine "Blank_Page" ---
        # create an object to store info about Routine Blank_Page
        Blank_Page = data.Routine(
            name='Blank_Page',
            components=[Blank_Background],
        )
        Blank_Page.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Blank_Page
        Blank_Page.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Blank_Page.tStart = globalClock.getTime(format='float')
        Blank_Page.status = STARTED
        thisExp.addData('Blank_Page.started', Blank_Page.tStart)
        Blank_Page.maxDuration = None
        # keep track of which components have finished
        Blank_PageComponents = Blank_Page.components
        for thisComponent in Blank_Page.components:
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
        
        # --- Run Routine "Blank_Page" ---
        thisExp.currentRoutine = Blank_Page
        Blank_Page.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Blank_Background* updates
            
            # if Blank_Background is starting this frame...
            if Blank_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Blank_Background.frameNStart = frameN  # exact frame index
                Blank_Background.tStart = t  # local t and not account for scr refresh
                Blank_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Blank_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank_Background.started')
                # update status
                Blank_Background.status = STARTED
                Blank_Background.setAutoDraw(True)
            
            # if Blank_Background is active this frame...
            if Blank_Background.status == STARTED:
                # update params
                pass
            
            # if Blank_Background is stopping this frame...
            if Blank_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Blank_Background.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Blank_Background.tStop = t  # not accounting for scr refresh
                    Blank_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Blank_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Blank_Background.stopped')
                    # update status
                    Blank_Background.status = FINISHED
                    Blank_Background.setAutoDraw(False)
            
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
                    currentRoutine=Blank_Page,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Blank_Page.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Blank_Page.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Blank_Page.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Blank_Page" ---
        for thisComponent in Blank_Page.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Blank_Page
        Blank_Page.tStop = globalClock.getTime(format='float')
        Blank_Page.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Blank_Page.stopped', Blank_Page.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Blank_Page.maxDurationReached:
            routineTimer.addTime(-Blank_Page.maxDuration)
        elif Blank_Page.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Perspective_cue" ---
        # create an object to store info about Routine Perspective_cue
        Perspective_cue = data.Routine(
            name='Perspective_cue',
            components=[Cue_Persp_Background, Perspective_Cue2],
        )
        Perspective_cue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Perspective_Cue2.setImage(P_Cue)
        # store start times for Perspective_cue
        Perspective_cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Perspective_cue.tStart = globalClock.getTime(format='float')
        Perspective_cue.status = STARTED
        thisExp.addData('Perspective_cue.started', Perspective_cue.tStart)
        Perspective_cue.maxDuration = None
        # keep track of which components have finished
        Perspective_cueComponents = Perspective_cue.components
        for thisComponent in Perspective_cue.components:
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
        
        # --- Run Routine "Perspective_cue" ---
        thisExp.currentRoutine = Perspective_cue
        Perspective_cue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.75:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Cue_Persp_Background* updates
            
            # if Cue_Persp_Background is starting this frame...
            if Cue_Persp_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Cue_Persp_Background.frameNStart = frameN  # exact frame index
                Cue_Persp_Background.tStart = t  # local t and not account for scr refresh
                Cue_Persp_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Cue_Persp_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Cue_Persp_Background.started')
                # update status
                Cue_Persp_Background.status = STARTED
                Cue_Persp_Background.setAutoDraw(True)
            
            # if Cue_Persp_Background is active this frame...
            if Cue_Persp_Background.status == STARTED:
                # update params
                pass
            
            # if Cue_Persp_Background is stopping this frame...
            if Cue_Persp_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Cue_Persp_Background.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Cue_Persp_Background.tStop = t  # not accounting for scr refresh
                    Cue_Persp_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Cue_Persp_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Cue_Persp_Background.stopped')
                    # update status
                    Cue_Persp_Background.status = FINISHED
                    Cue_Persp_Background.setAutoDraw(False)
            
            # *Perspective_Cue2* updates
            
            # if Perspective_Cue2 is starting this frame...
            if Perspective_Cue2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Perspective_Cue2.frameNStart = frameN  # exact frame index
                Perspective_Cue2.tStart = t  # local t and not account for scr refresh
                Perspective_Cue2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Perspective_Cue2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Perspective_Cue2.started')
                # update status
                Perspective_Cue2.status = STARTED
                Perspective_Cue2.setAutoDraw(True)
            
            # if Perspective_Cue2 is active this frame...
            if Perspective_Cue2.status == STARTED:
                # update params
                pass
            
            # if Perspective_Cue2 is stopping this frame...
            if Perspective_Cue2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Perspective_Cue2.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Perspective_Cue2.tStop = t  # not accounting for scr refresh
                    Perspective_Cue2.tStopRefresh = tThisFlipGlobal  # on global time
                    Perspective_Cue2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Perspective_Cue2.stopped')
                    # update status
                    Perspective_Cue2.status = FINISHED
                    Perspective_Cue2.setAutoDraw(False)
            
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
                    currentRoutine=Perspective_cue,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Perspective_cue.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Perspective_cue.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Perspective_cue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Perspective_cue" ---
        for thisComponent in Perspective_cue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Perspective_cue
        Perspective_cue.tStop = globalClock.getTime(format='float')
        Perspective_cue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Perspective_cue.stopped', Perspective_cue.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Perspective_cue.maxDurationReached:
            routineTimer.addTime(-Perspective_cue.maxDuration)
        elif Perspective_cue.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.750000)
        
        # --- Prepare to start Routine "Blank_Page" ---
        # create an object to store info about Routine Blank_Page
        Blank_Page = data.Routine(
            name='Blank_Page',
            components=[Blank_Background],
        )
        Blank_Page.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Blank_Page
        Blank_Page.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Blank_Page.tStart = globalClock.getTime(format='float')
        Blank_Page.status = STARTED
        thisExp.addData('Blank_Page.started', Blank_Page.tStart)
        Blank_Page.maxDuration = None
        # keep track of which components have finished
        Blank_PageComponents = Blank_Page.components
        for thisComponent in Blank_Page.components:
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
        
        # --- Run Routine "Blank_Page" ---
        thisExp.currentRoutine = Blank_Page
        Blank_Page.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Blank_Background* updates
            
            # if Blank_Background is starting this frame...
            if Blank_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Blank_Background.frameNStart = frameN  # exact frame index
                Blank_Background.tStart = t  # local t and not account for scr refresh
                Blank_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Blank_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank_Background.started')
                # update status
                Blank_Background.status = STARTED
                Blank_Background.setAutoDraw(True)
            
            # if Blank_Background is active this frame...
            if Blank_Background.status == STARTED:
                # update params
                pass
            
            # if Blank_Background is stopping this frame...
            if Blank_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Blank_Background.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Blank_Background.tStop = t  # not accounting for scr refresh
                    Blank_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Blank_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Blank_Background.stopped')
                    # update status
                    Blank_Background.status = FINISHED
                    Blank_Background.setAutoDraw(False)
            
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
                    currentRoutine=Blank_Page,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Blank_Page.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Blank_Page.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Blank_Page.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Blank_Page" ---
        for thisComponent in Blank_Page.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Blank_Page
        Blank_Page.tStop = globalClock.getTime(format='float')
        Blank_Page.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Blank_Page.stopped', Blank_Page.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Blank_Page.maxDurationReached:
            routineTimer.addTime(-Blank_Page.maxDuration)
        elif Blank_Page.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Room" ---
        # create an object to store info about Routine Room
        Room = data.Routine(
            name='Room',
            components=[Room_Background, image, key_resp],
        )
        Room.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image.setImage(PICTURE)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for Room
        Room.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Room.tStart = globalClock.getTime(format='float')
        Room.status = STARTED
        thisExp.addData('Room.started', Room.tStart)
        Room.maxDuration = None
        # keep track of which components have finished
        RoomComponents = Room.components
        for thisComponent in Room.components:
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
        
        # --- Run Routine "Room" ---
        thisExp.currentRoutine = Room
        Room.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Room_Background* updates
            
            # if Room_Background is starting this frame...
            if Room_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Room_Background.frameNStart = frameN  # exact frame index
                Room_Background.tStart = t  # local t and not account for scr refresh
                Room_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Room_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Room_Background.started')
                # update status
                Room_Background.status = STARTED
                Room_Background.setAutoDraw(True)
            
            # if Room_Background is active this frame...
            if Room_Background.status == STARTED:
                # update params
                pass
            
            # if Room_Background is stopping this frame...
            if Room_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Room_Background.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Room_Background.tStop = t  # not accounting for scr refresh
                    Room_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Room_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Room_Background.stopped')
                    # update status
                    Room_Background.status = FINISHED
                    Room_Background.setAutoDraw(False)
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['c','n'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
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
                    currentRoutine=Room,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Room.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Room.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Room.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Room" ---
        for thisComponent in Room.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Room
        Room.tStop = globalClock.getTime(format='float')
        Room.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Room.stopped', Room.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        Practice.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            Practice.addData('key_resp.rt', key_resp.rt)
            Practice.addData('key_resp.duration', key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Room.maxDurationReached:
            routineTimer.addTime(-Room.maxDuration)
        elif Room.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[FB_Background, feedback_text],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        feedback_text.setText(feedback_text)
        # Run 'Begin Routine' code from code
        # Ensure that key_resp.keys is not None and process correctly
        if key_resp.keys is not None:
            if isinstance(key_resp.keys, list):  # Check if the response is stored as a list
                response = key_resp.keys[0]  # Get the first response
            else:
                response = key_resp.keys  # Use directly if it's a string
        
            # Convert both response and correct_answer to strings for safe comparison
            if response == str(Correct_Answer):  
                feedback_message = "درست"
            else:
                feedback_message = "نادرست"
        else:
            feedback_message = "نادرست"  # Default to incorrect if no response is given
        
        # Update the text component with the feedback message
        feedback_text.setText(feedback_message)
        
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
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
        
        # --- Run Routine "feedback" ---
        thisExp.currentRoutine = feedback
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FB_Background* updates
            
            # if FB_Background is starting this frame...
            if FB_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FB_Background.frameNStart = frameN  # exact frame index
                FB_Background.tStart = t  # local t and not account for scr refresh
                FB_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FB_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FB_Background.started')
                # update status
                FB_Background.status = STARTED
                FB_Background.setAutoDraw(True)
            
            # if FB_Background is active this frame...
            if FB_Background.status == STARTED:
                # update params
                pass
            
            # if FB_Background is stopping this frame...
            if FB_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FB_Background.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    FB_Background.tStop = t  # not accounting for scr refresh
                    FB_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    FB_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FB_Background.stopped')
                    # update status
                    FB_Background.status = FINISHED
                    FB_Background.setAutoDraw(False)
            
            # *feedback_text* updates
            
            # if feedback_text is starting this frame...
            if feedback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_text.frameNStart = frameN  # exact frame index
                feedback_text.tStart = t  # local t and not account for scr refresh
                feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_text.started')
                # update status
                feedback_text.status = STARTED
                feedback_text.setAutoDraw(True)
            
            # if feedback_text is active this frame...
            if feedback_text.status == STARTED:
                # update params
                pass
            
            # if feedback_text is stopping this frame...
            if feedback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_text.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_text.tStop = t  # not accounting for scr refresh
                    feedback_text.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_text.stopped')
                    # update status
                    feedback_text.status = FINISHED
                    feedback_text.setAutoDraw(False)
            
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
                    currentRoutine=feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        # mark thisPractice as finished
        if hasattr(thisPractice, 'status'):
            thisPractice.status = FINISHED
        # if awaiting a pause, pause now
        if Practice.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Practice.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Practice'
    Practice.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Experiment_section" ---
    # create an object to store info about Routine Experiment_section
    Experiment_section = data.Routine(
        name='Experiment_section',
        components=[Exp_Background, Press_Space_2, Exp_Resp_Space],
    )
    Experiment_section.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Exp_Resp_Space
    Exp_Resp_Space.keys = []
    Exp_Resp_Space.rt = []
    _Exp_Resp_Space_allKeys = []
    # store start times for Experiment_section
    Experiment_section.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Experiment_section.tStart = globalClock.getTime(format='float')
    Experiment_section.status = STARTED
    thisExp.addData('Experiment_section.started', Experiment_section.tStart)
    Experiment_section.maxDuration = None
    # keep track of which components have finished
    Experiment_sectionComponents = Experiment_section.components
    for thisComponent in Experiment_section.components:
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
    
    # --- Run Routine "Experiment_section" ---
    thisExp.currentRoutine = Experiment_section
    Experiment_section.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Exp_Background* updates
        
        # if Exp_Background is starting this frame...
        if Exp_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Exp_Background.frameNStart = frameN  # exact frame index
            Exp_Background.tStart = t  # local t and not account for scr refresh
            Exp_Background.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Exp_Background, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Exp_Background.started')
            # update status
            Exp_Background.status = STARTED
            Exp_Background.setAutoDraw(True)
        
        # if Exp_Background is active this frame...
        if Exp_Background.status == STARTED:
            # update params
            pass
        
        # *Press_Space_2* updates
        
        # if Press_Space_2 is starting this frame...
        if Press_Space_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Press_Space_2.frameNStart = frameN  # exact frame index
            Press_Space_2.tStart = t  # local t and not account for scr refresh
            Press_Space_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Press_Space_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Press_Space_2.started')
            # update status
            Press_Space_2.status = STARTED
            Press_Space_2.setAutoDraw(True)
        
        # if Press_Space_2 is active this frame...
        if Press_Space_2.status == STARTED:
            # update params
            pass
        
        # *Exp_Resp_Space* updates
        waitOnFlip = False
        
        # if Exp_Resp_Space is starting this frame...
        if Exp_Resp_Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Exp_Resp_Space.frameNStart = frameN  # exact frame index
            Exp_Resp_Space.tStart = t  # local t and not account for scr refresh
            Exp_Resp_Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Exp_Resp_Space, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Exp_Resp_Space.started')
            # update status
            Exp_Resp_Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Exp_Resp_Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Exp_Resp_Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Exp_Resp_Space.status == STARTED and not waitOnFlip:
            theseKeys = Exp_Resp_Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Exp_Resp_Space_allKeys.extend(theseKeys)
            if len(_Exp_Resp_Space_allKeys):
                Exp_Resp_Space.keys = _Exp_Resp_Space_allKeys[-1].name  # just the last key pressed
                Exp_Resp_Space.rt = _Exp_Resp_Space_allKeys[-1].rt
                Exp_Resp_Space.duration = _Exp_Resp_Space_allKeys[-1].duration
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
                currentRoutine=Experiment_section,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Experiment_section.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Experiment_section.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Experiment_section.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Experiment_section" ---
    for thisComponent in Experiment_section.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Experiment_section
    Experiment_section.tStop = globalClock.getTime(format='float')
    Experiment_section.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Experiment_section.stopped', Experiment_section.tStop)
    # check responses
    if Exp_Resp_Space.keys in ['', [], None]:  # No response was made
        Exp_Resp_Space.keys = None
    thisExp.addData('Exp_Resp_Space.keys',Exp_Resp_Space.keys)
    if Exp_Resp_Space.keys != None:  # we had a response
        thisExp.addData('Exp_Resp_Space.rt', Exp_Resp_Space.rt)
        thisExp.addData('Exp_Resp_Space.duration', Exp_Resp_Space.duration)
    thisExp.nextEntry()
    # the Routine "Experiment_section" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Loop = data.TrialHandler2(
        name='Loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Female_experimental.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(Loop)  # add the loop to the experiment
    thisLoop = Loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop.rgb)
    if thisLoop != None:
        for paramName in thisLoop:
            globals()[paramName] = thisLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop in Loop:
        Loop.status = STARTED
        if hasattr(thisLoop, 'status'):
            thisLoop.status = STARTED
        currentLoop = Loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop.rgb)
        if thisLoop != None:
            for paramName in thisLoop:
                globals()[paramName] = thisLoop[paramName]
        
        # --- Prepare to start Routine "Fix_Cross" ---
        # create an object to store info about Routine Fix_Cross
        Fix_Cross = data.Routine(
            name='Fix_Cross',
            components=[Background, Fix_cross],
        )
        Fix_Cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Fix_Cross
        Fix_Cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fix_Cross.tStart = globalClock.getTime(format='float')
        Fix_Cross.status = STARTED
        thisExp.addData('Fix_Cross.started', Fix_Cross.tStart)
        Fix_Cross.maxDuration = None
        # keep track of which components have finished
        Fix_CrossComponents = Fix_Cross.components
        for thisComponent in Fix_Cross.components:
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
        
        # --- Run Routine "Fix_Cross" ---
        thisExp.currentRoutine = Fix_Cross
        Fix_Cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.75:
            # if trial has changed, end Routine now
            if hasattr(thisLoop, 'status') and thisLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Background* updates
            
            # if Background is starting this frame...
            if Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Background.frameNStart = frameN  # exact frame index
                Background.tStart = t  # local t and not account for scr refresh
                Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Background.started')
                # update status
                Background.status = STARTED
                Background.setAutoDraw(True)
            
            # if Background is active this frame...
            if Background.status == STARTED:
                # update params
                pass
            
            # if Background is stopping this frame...
            if Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Background.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Background.tStop = t  # not accounting for scr refresh
                    Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Background.stopped')
                    # update status
                    Background.status = FINISHED
                    Background.setAutoDraw(False)
            
            # *Fix_cross* updates
            
            # if Fix_cross is starting this frame...
            if Fix_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Fix_cross.frameNStart = frameN  # exact frame index
                Fix_cross.tStart = t  # local t and not account for scr refresh
                Fix_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Fix_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Fix_cross.started')
                # update status
                Fix_cross.status = STARTED
                Fix_cross.setAutoDraw(True)
            
            # if Fix_cross is active this frame...
            if Fix_cross.status == STARTED:
                # update params
                pass
            
            # if Fix_cross is stopping this frame...
            if Fix_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Fix_cross.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Fix_cross.tStop = t  # not accounting for scr refresh
                    Fix_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    Fix_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fix_cross.stopped')
                    # update status
                    Fix_cross.status = FINISHED
                    Fix_cross.setAutoDraw(False)
            
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
                    currentRoutine=Fix_Cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Fix_Cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Fix_Cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Fix_Cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fix_Cross" ---
        for thisComponent in Fix_Cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fix_Cross
        Fix_Cross.tStop = globalClock.getTime(format='float')
        Fix_Cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fix_Cross.stopped', Fix_Cross.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fix_Cross.maxDurationReached:
            routineTimer.addTime(-Fix_Cross.maxDuration)
        elif Fix_Cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.750000)
        
        # --- Prepare to start Routine "Blank_Page" ---
        # create an object to store info about Routine Blank_Page
        Blank_Page = data.Routine(
            name='Blank_Page',
            components=[Blank_Background],
        )
        Blank_Page.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Blank_Page
        Blank_Page.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Blank_Page.tStart = globalClock.getTime(format='float')
        Blank_Page.status = STARTED
        thisExp.addData('Blank_Page.started', Blank_Page.tStart)
        Blank_Page.maxDuration = None
        # keep track of which components have finished
        Blank_PageComponents = Blank_Page.components
        for thisComponent in Blank_Page.components:
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
        
        # --- Run Routine "Blank_Page" ---
        thisExp.currentRoutine = Blank_Page
        Blank_Page.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisLoop, 'status') and thisLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Blank_Background* updates
            
            # if Blank_Background is starting this frame...
            if Blank_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Blank_Background.frameNStart = frameN  # exact frame index
                Blank_Background.tStart = t  # local t and not account for scr refresh
                Blank_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Blank_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank_Background.started')
                # update status
                Blank_Background.status = STARTED
                Blank_Background.setAutoDraw(True)
            
            # if Blank_Background is active this frame...
            if Blank_Background.status == STARTED:
                # update params
                pass
            
            # if Blank_Background is stopping this frame...
            if Blank_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Blank_Background.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Blank_Background.tStop = t  # not accounting for scr refresh
                    Blank_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Blank_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Blank_Background.stopped')
                    # update status
                    Blank_Background.status = FINISHED
                    Blank_Background.setAutoDraw(False)
            
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
                    currentRoutine=Blank_Page,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Blank_Page.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Blank_Page.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Blank_Page.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Blank_Page" ---
        for thisComponent in Blank_Page.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Blank_Page
        Blank_Page.tStop = globalClock.getTime(format='float')
        Blank_Page.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Blank_Page.stopped', Blank_Page.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Blank_Page.maxDurationReached:
            routineTimer.addTime(-Blank_Page.maxDuration)
        elif Blank_Page.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Perspective_cue" ---
        # create an object to store info about Routine Perspective_cue
        Perspective_cue = data.Routine(
            name='Perspective_cue',
            components=[Cue_Persp_Background, Perspective_Cue2],
        )
        Perspective_cue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Perspective_Cue2.setImage(P_Cue)
        # store start times for Perspective_cue
        Perspective_cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Perspective_cue.tStart = globalClock.getTime(format='float')
        Perspective_cue.status = STARTED
        thisExp.addData('Perspective_cue.started', Perspective_cue.tStart)
        Perspective_cue.maxDuration = None
        # keep track of which components have finished
        Perspective_cueComponents = Perspective_cue.components
        for thisComponent in Perspective_cue.components:
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
        
        # --- Run Routine "Perspective_cue" ---
        thisExp.currentRoutine = Perspective_cue
        Perspective_cue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.75:
            # if trial has changed, end Routine now
            if hasattr(thisLoop, 'status') and thisLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Cue_Persp_Background* updates
            
            # if Cue_Persp_Background is starting this frame...
            if Cue_Persp_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Cue_Persp_Background.frameNStart = frameN  # exact frame index
                Cue_Persp_Background.tStart = t  # local t and not account for scr refresh
                Cue_Persp_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Cue_Persp_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Cue_Persp_Background.started')
                # update status
                Cue_Persp_Background.status = STARTED
                Cue_Persp_Background.setAutoDraw(True)
            
            # if Cue_Persp_Background is active this frame...
            if Cue_Persp_Background.status == STARTED:
                # update params
                pass
            
            # if Cue_Persp_Background is stopping this frame...
            if Cue_Persp_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Cue_Persp_Background.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Cue_Persp_Background.tStop = t  # not accounting for scr refresh
                    Cue_Persp_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Cue_Persp_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Cue_Persp_Background.stopped')
                    # update status
                    Cue_Persp_Background.status = FINISHED
                    Cue_Persp_Background.setAutoDraw(False)
            
            # *Perspective_Cue2* updates
            
            # if Perspective_Cue2 is starting this frame...
            if Perspective_Cue2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Perspective_Cue2.frameNStart = frameN  # exact frame index
                Perspective_Cue2.tStart = t  # local t and not account for scr refresh
                Perspective_Cue2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Perspective_Cue2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Perspective_Cue2.started')
                # update status
                Perspective_Cue2.status = STARTED
                Perspective_Cue2.setAutoDraw(True)
            
            # if Perspective_Cue2 is active this frame...
            if Perspective_Cue2.status == STARTED:
                # update params
                pass
            
            # if Perspective_Cue2 is stopping this frame...
            if Perspective_Cue2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Perspective_Cue2.tStartRefresh + 0.75-frameTolerance:
                    # keep track of stop time/frame for later
                    Perspective_Cue2.tStop = t  # not accounting for scr refresh
                    Perspective_Cue2.tStopRefresh = tThisFlipGlobal  # on global time
                    Perspective_Cue2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Perspective_Cue2.stopped')
                    # update status
                    Perspective_Cue2.status = FINISHED
                    Perspective_Cue2.setAutoDraw(False)
            
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
                    currentRoutine=Perspective_cue,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Perspective_cue.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Perspective_cue.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Perspective_cue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Perspective_cue" ---
        for thisComponent in Perspective_cue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Perspective_cue
        Perspective_cue.tStop = globalClock.getTime(format='float')
        Perspective_cue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Perspective_cue.stopped', Perspective_cue.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Perspective_cue.maxDurationReached:
            routineTimer.addTime(-Perspective_cue.maxDuration)
        elif Perspective_cue.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.750000)
        
        # --- Prepare to start Routine "Blank_Page" ---
        # create an object to store info about Routine Blank_Page
        Blank_Page = data.Routine(
            name='Blank_Page',
            components=[Blank_Background],
        )
        Blank_Page.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Blank_Page
        Blank_Page.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Blank_Page.tStart = globalClock.getTime(format='float')
        Blank_Page.status = STARTED
        thisExp.addData('Blank_Page.started', Blank_Page.tStart)
        Blank_Page.maxDuration = None
        # keep track of which components have finished
        Blank_PageComponents = Blank_Page.components
        for thisComponent in Blank_Page.components:
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
        
        # --- Run Routine "Blank_Page" ---
        thisExp.currentRoutine = Blank_Page
        Blank_Page.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisLoop, 'status') and thisLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Blank_Background* updates
            
            # if Blank_Background is starting this frame...
            if Blank_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Blank_Background.frameNStart = frameN  # exact frame index
                Blank_Background.tStart = t  # local t and not account for scr refresh
                Blank_Background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Blank_Background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Blank_Background.started')
                # update status
                Blank_Background.status = STARTED
                Blank_Background.setAutoDraw(True)
            
            # if Blank_Background is active this frame...
            if Blank_Background.status == STARTED:
                # update params
                pass
            
            # if Blank_Background is stopping this frame...
            if Blank_Background.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Blank_Background.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    Blank_Background.tStop = t  # not accounting for scr refresh
                    Blank_Background.tStopRefresh = tThisFlipGlobal  # on global time
                    Blank_Background.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Blank_Background.stopped')
                    # update status
                    Blank_Background.status = FINISHED
                    Blank_Background.setAutoDraw(False)
            
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
                    currentRoutine=Blank_Page,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Blank_Page.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Blank_Page.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Blank_Page.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Blank_Page" ---
        for thisComponent in Blank_Page.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Blank_Page
        Blank_Page.tStop = globalClock.getTime(format='float')
        Blank_Page.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Blank_Page.stopped', Blank_Page.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Blank_Page.maxDurationReached:
            routineTimer.addTime(-Blank_Page.maxDuration)
        elif Blank_Page.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "Exp_Room" ---
        # create an object to store info about Routine Exp_Room
        Exp_Room = data.Routine(
            name='Exp_Room',
            components=[Room_Back, Image, Resp],
        )
        Exp_Room.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        Image.setImage(PICTURE)
        # create starting attributes for Resp
        Resp.keys = []
        Resp.rt = []
        _Resp_allKeys = []
        # store start times for Exp_Room
        Exp_Room.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Exp_Room.tStart = globalClock.getTime(format='float')
        Exp_Room.status = STARTED
        thisExp.addData('Exp_Room.started', Exp_Room.tStart)
        Exp_Room.maxDuration = None
        # keep track of which components have finished
        Exp_RoomComponents = Exp_Room.components
        for thisComponent in Exp_Room.components:
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
        
        # --- Run Routine "Exp_Room" ---
        thisExp.currentRoutine = Exp_Room
        Exp_Room.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisLoop, 'status') and thisLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Room_Back* updates
            
            # if Room_Back is starting this frame...
            if Room_Back.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Room_Back.frameNStart = frameN  # exact frame index
                Room_Back.tStart = t  # local t and not account for scr refresh
                Room_Back.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Room_Back, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Room_Back.started')
                # update status
                Room_Back.status = STARTED
                Room_Back.setAutoDraw(True)
            
            # if Room_Back is active this frame...
            if Room_Back.status == STARTED:
                # update params
                pass
            
            # if Room_Back is stopping this frame...
            if Room_Back.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Room_Back.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Room_Back.tStop = t  # not accounting for scr refresh
                    Room_Back.tStopRefresh = tThisFlipGlobal  # on global time
                    Room_Back.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Room_Back.stopped')
                    # update status
                    Room_Back.status = FINISHED
                    Room_Back.setAutoDraw(False)
            
            # *Image* updates
            
            # if Image is starting this frame...
            if Image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Image.frameNStart = frameN  # exact frame index
                Image.tStart = t  # local t and not account for scr refresh
                Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Image.started')
                # update status
                Image.status = STARTED
                Image.setAutoDraw(True)
            
            # if Image is active this frame...
            if Image.status == STARTED:
                # update params
                pass
            
            # if Image is stopping this frame...
            if Image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Image.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Image.tStop = t  # not accounting for scr refresh
                    Image.tStopRefresh = tThisFlipGlobal  # on global time
                    Image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Image.stopped')
                    # update status
                    Image.status = FINISHED
                    Image.setAutoDraw(False)
            
            # *Resp* updates
            waitOnFlip = False
            
            # if Resp is starting this frame...
            if Resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Resp.frameNStart = frameN  # exact frame index
                Resp.tStart = t  # local t and not account for scr refresh
                Resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Resp.started')
                # update status
                Resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(Resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if Resp is stopping this frame...
            if Resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Resp.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    Resp.tStop = t  # not accounting for scr refresh
                    Resp.tStopRefresh = tThisFlipGlobal  # on global time
                    Resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Resp.stopped')
                    # update status
                    Resp.status = FINISHED
                    Resp.status = FINISHED
            if Resp.status == STARTED and not waitOnFlip:
                theseKeys = Resp.getKeys(keyList=['c','n'], ignoreKeys=["escape"], waitRelease=False)
                _Resp_allKeys.extend(theseKeys)
                if len(_Resp_allKeys):
                    Resp.keys = _Resp_allKeys[-1].name  # just the last key pressed
                    Resp.rt = _Resp_allKeys[-1].rt
                    Resp.duration = _Resp_allKeys[-1].duration
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
                    currentRoutine=Exp_Room,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Exp_Room.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Exp_Room.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Exp_Room.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Exp_Room" ---
        for thisComponent in Exp_Room.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Exp_Room
        Exp_Room.tStop = globalClock.getTime(format='float')
        Exp_Room.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Exp_Room.stopped', Exp_Room.tStop)
        # check responses
        if Resp.keys in ['', [], None]:  # No response was made
            Resp.keys = None
        Loop.addData('Resp.keys',Resp.keys)
        if Resp.keys != None:  # we had a response
            Loop.addData('Resp.rt', Resp.rt)
            Loop.addData('Resp.duration', Resp.duration)
        # Run 'End Routine' code from CodeComponent_2
        # Get the participant's response
        response = Resp.keys  # Use 'Resp' as the name of your Keyboard component
        
        # Get the correct response from the current trial
        correct_response = Loop.trialList[Loop.thisTrialN]['Correct_Answer']
        
        # Check if the response is correct
        correct = (response == correct_response)
        
        # Save the data
        thisExp.addData('image', Loop.trialList[Loop.thisTrialN]['PICTURE'])
        thisExp.addData('response', response)
        thisExp.addData('correct', correct)
        thisExp.nextEntry()
        
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Exp_Room.maxDurationReached:
            routineTimer.addTime(-Exp_Room.maxDuration)
        elif Exp_Room.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        # mark thisLoop as finished
        if hasattr(thisLoop, 'status'):
            thisLoop.status = FINISHED
        # if awaiting a pause, pause now
        if Loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'Loop'
    Loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[End_Background, image_2],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
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
    
    # --- Run Routine "End" ---
    thisExp.currentRoutine = End
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *End_Background* updates
        
        # if End_Background is starting this frame...
        if End_Background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            End_Background.frameNStart = frameN  # exact frame index
            End_Background.tStart = t  # local t and not account for scr refresh
            End_Background.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(End_Background, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'End_Background.started')
            # update status
            End_Background.status = STARTED
            End_Background.setAutoDraw(True)
        
        # if End_Background is active this frame...
        if End_Background.status == STARTED:
            # update params
            pass
        
        # if End_Background is stopping this frame...
        if End_Background.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > End_Background.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                End_Background.tStop = t  # not accounting for scr refresh
                End_Background.tStopRefresh = tThisFlipGlobal  # on global time
                End_Background.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'End_Background.stopped')
                # update status
                End_Background.status = FINISHED
                End_Background.setAutoDraw(False)
        
        # *image_2* updates
        
        # if image_2 is starting this frame...
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_2.started')
            # update status
            image_2.status = STARTED
            image_2.setAutoDraw(True)
        
        # if image_2 is active this frame...
        if image_2.status == STARTED:
            # update params
            pass
        
        # if image_2 is stopping this frame...
        if image_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_2.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                image_2.tStop = t  # not accounting for scr refresh
                image_2.tStopRefresh = tThisFlipGlobal  # on global time
                image_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.stopped')
                # update status
                image_2.status = FINISHED
                image_2.setAutoDraw(False)
        
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
                currentRoutine=End,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            End.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if End.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if End.maxDurationReached:
        routineTimer.addTime(-End.maxDuration)
    elif End.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
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
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
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
