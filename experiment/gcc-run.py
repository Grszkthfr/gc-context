﻿# !/usr/bin/env python
# -*- coding: utf-8 -*- #

# This code runs a modified gaze cueing paradigma.
# Originally the experiment was designed to run during winter semester 2018 at the university of Würzburg.
# Feel free to reuse any of this code, however, it comes with absolute no warranty. 

# Libraries

import os
import numpy as np
import csv
import random
from itertools import product, chain
import datetime
import cv2
from psychopy import visual, core, event, gui

# 1) Modules

# a) Hardware

screen_size = [1680, 1050]
monitor = "lab011"

full_screen = True

# b) Text

exp_name = "gcc"

blank = ""

instruction_pages = [u"", u"Herzlich Willkommen bei unserer Studie „Blickfang“!\n\nVielen Dank, dass Sie sich dafür entschieden haben an unserem Experiment teilzunehmen.\n\nBitte achten Sie auf die weiteren Anweisungen.", u"Bitte bearbeiten Sie den Versuch möglichst schnell und präzise.\n\nFokussieren Sie bitte immer das Fixationskreuz.\n\nLegen Sie den Zeigefinger und Mittelfinger ihrer dominanten Hand nun auf die markierten Tasten und lassen Sie während des Versuchs die Finger bitte auf der Tastatur.", u"Im Versuch werden Ihnen Bilder dargeboten.\n\nWenn ein \tE\t erscheint, drücken Sie bitte die Taste, die mit dem 'E' markiert ist. \n\nWenn ein \tF\t erscheint, drücken Sie bitte die Taste, die mit dem 'F' markiert ist.\n\n\nBei Fragen wenden Sie sich bitte jetzt an den Versuchsleiter.",
u"Bitte denken Sie daran, das Fixationskreuz zu fokussieren.\n\nSie können den Versuch nun starten."]

text_continue = u"\n\n\n\t\t\t\t\tWeiter mit <Leertaste>"

warningText = u"Bitte schneller reagieren."

pause_text = u"Sie haben den Block nun beendet.\n\nSobald Sie bereit sind können Sie den nächsten Block starten."

goodbye_text = u"Vielen Dank, dass Sie bei unserer Studie mitgemacht haben.\n\nMelden Sie sich bitte leise beim Versuchsleiter und füllen Sie die Fragebögen aus."

# c) Characteristics

session_info = {"subject": "ID of subject",
                "session": "Number of Session",
                "experimenter": "Name of experimenter",
                "computer": os.getenv('COMPUTERNAME')}

max_rt = .800       # maximale time to respond to target in seconds
iti = (1, 1.5)      # jittered time in seconds between two trials
min_reading = 2     # minimal time instructions are present

t_id = ["E", "F"]
t_pos = ["left", "right"]
# cue_dir = ["left", "right"]

soa = .5                   # inter stimulus intervall in seconds

response_key_E = 'u'           # response key for "E"
response_key_F = 'n'           # response key for "F"
continue_key = "space"        # continue key for instructions
quit_key = "q"                # exit key to exit experiment (for experimenter)

# d) Prerequisits
stim_dir = [
    "stimuli" + os.path.sep + "modified" + os.path.sep + "context_present" + os.path.sep,
    "stimuli" + os.path.sep + "modified" + os.path.sep + "context_covered" + os.path.sep]

img_dir = 'original' + os.path.sep + 'imgs' + os.path.sep
roi_dir = 'original' + os.path.sep + 'rois' + os.path.sep

#set n_trials for testing
n_trials = []
#n_trials = list(range(4))


trials_ctx = []
trials_fo = []
trial_count = 1             # count trials, starting with 1
block_count = 1             # count blocks, starting with 1

win = ""
txt = ""

# 2) Functions


def wait(seconds):
    try:
        # Wait presentationTime seconds. 
        number = float(seconds)
        # (?) Maximum accuracy (2nd parameter) is used all the time (i.e. presentationTime seconds).
        core.wait(number, number)  
    except TypeError:
        jitter = random.random() * (seconds[1] - seconds[0]) + seconds[0]
        core.wait(jitter, jitter)
        # print("Jitter: %.3f sec" %(jitter)); #to control for correctness


def getDate(time=core.getAbsTime(), format='%Y-%m-%d'):
    timeNowString = datetime.datetime.fromtimestamp(time).strftime(format)
    return timeNowString

def makeDirectory(path):
    if not os.path.isdir(path):
        os.makedirs(path) # throws an error when failing


def prepareExperiment():
    global win

    # add session info in dialoge
    while True:
        input = gui.DlgFromDict(dictionary=session_info, title=exp_name)
        if input.OK == False:
            core.quit()

        # if Subject field wasn't specified, display Input-win again
        if (session_info['subject'] != ''): break   

    # set global window
    win = visual.Window(screen_size, monitor=monitor, fullscr=full_screen)
    event.Mouse(visible=False)

    # prepare trial list
    getTriallist(stim_dir)


def showText(win, txt):
    msg = visual.TextStim(win, text=txt, height=0.05)
    msg.draw()
    win.flip()
    wait(min_reading)
    event.clearEvents()
    while True:
        # Participant decides when to proceed
        if event.getKeys(continue_key):
            break
        if event.getKeys(quit_key):
            core.quit()


def makeStim(draw_center=False):
    imgs = []

    makeDirectory(stim_dir[0])
    makeDirectory(stim_dir[1])

    print("Please wait, creating stimuli!")

    for imgs in os.listdir(os.path.join(stim_dir, img_dir)):
        # print("%s is being modified" %(imgs))

        img = cv2.imread(os.path.join(stim_dir, img_dir, imgs))
        roi = cv2.imread(os.path.join(stim_dir, roi_dir, imgs))

        # targets color detection
        # define (the list of) & create NumPy arrays from the boundaries
        # BGR: OpenCV represents images as NumPy arrays in reverse order
        lower_t = np.array([0,250,250], dtype="uint8")
        upper_t = np.array([0,255,255], dtype="uint8")

        mask_t = cv2.inRange(roi, lower_t, upper_t)
        roi_t = cv2.bitwise_and(roi, roi, mask=mask_t)

        roi_t_gry = cv2.cvtColor(roi_t, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(roi_t_gry, 127, 255, 0)
        roi_t_gry2, contours, hierachy = cv2.findContours(thresh, 1, 2)

        cnt_t = sorted(contours, key=cv2.contourArea, reverse=True)

        # cue color detection
        lower_c = np.array([0,250,0], dtype="uint8")
        upper_c = np.array([0,255,0], dtype="uint8")

        mask_c = cv2.inRange(roi, lower_c, upper_c)
        roi_c = cv2.bitwise_and(img, img, mask=mask_c)

        roi_c_c = cv2.bitwise_and(roi, roi, mask=mask_c)
        roi_c_c_gry = cv2.cvtColor(roi_c_c, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(roi_c_c_gry, 127, 255, 0)
        roi_c_c_gry2, contours, hierachy = cv2.findContours(thresh, 1, 2)

        cnt_c = sorted(contours, key=cv2.contourArea, reverse=True)

        # check if areas are detected
        if len(cnt_t) < 2 or len(cnt_c) < 1:
            with open("stimFail.txt", 'ab') as saveFile:
                fileWriter = csv.writer(saveFile, delimiter=',')
            
                # write trial
                fileWriter.writerow([imgs])

        else:

            # x1,y1 ------
            # |          |
            # |          |
            # |          |
            # --------x2,y2

            # coords rect 1 & 2
            t1_x1, t1_y1, t1_w, t1_h = cv2.boundingRect(cnt_t[0])
            t2_x1, t2_y1, t2_w, t2_h = cv2.boundingRect(cnt_t[1])

            # draw rect t1
            # t1_x2 = t1_x1+t1_w
            # t1_y2 = t1_y1+t1_h
            # cv2.rectangle(img, (t1_x1,t1_y1), (t1_x2,t1_y2), (128,128,128), -1)

            # draw rect t2
            # t2_x2 = t2_x1+t2_w
            # t2_y2 = t2_y1+t2_h
            # cv2.rectangle(img, (t2_x1,t2_y1), (t2_x2,t2_y2), (128,128,128), -1)

            if t1_h >= t2_h:
                h_max = t1_h
            else:
                h_max = t2_h

            if t1_w >= t2_w:
                w_max = t1_w
            else:
                w_max = t2_w

            # targets center
            cx_t1 = int(t1_x1+t1_w/2)
            cy_t1 = int(t1_y1+t1_h/ 2)

            cx_t2 = int(t2_x1+t2_w/2)
            cy_t2 = int(t2_y1+t2_h/2)

            # target rects
            t1_x1 = cx_t1-int(w_max/2)
            t1_y1 = cy_t1-int(h_max/2)
            t1_x2 = cx_t1+int(w_max/2)
            t1_y2 = cy_t1+int(h_max/2)

            t2_x1 = cx_t2-int(w_max/2)
            t2_y1 = cy_t2-int(h_max/2)
            t2_x2 = cx_t2+int(w_max/2)
            t2_y2 = cy_t2+int(h_max/2)

            # circle around face, only largest area of interest
            (x,y), radius = cv2.minEnclosingCircle(cnt_c[0])
            cx_c = int(x)
            cy_c = int(y)

            # somehow not black into white.
            roi_c[mask_c != 255] = [255,255,255]
            # draw target locations
            cv2.rectangle(roi_c, (t1_x1,t1_y1), (t1_x2,t1_y2), (128,128,128), -1)
            cv2.rectangle(roi_c, (t2_x1,t2_y1), (t2_x2,t2_y2), (128,128,128), -1)

            # draw target locations / cover targets
            cv2.rectangle(img, (t1_x1,t1_y1), (t1_x2,t1_y2), (128,128,128), -1)
            cv2.rectangle(img, (t2_x1,t2_y1), (t2_x2,t2_y2), (128,128,128), -1)

            if draw_center == True:
                # draw center [radius=3, color=(255,255,255), thickness=-1]
                cv2.circle(img, (cx_t1,cy_t1), 3, (0,0,255), -1)
                cv2.circle(img, (cx_t2,cy_t2), 3, (0,0,255), -1)
                cv2.circle(img,(cx_c,cy_c), 3, (0,0,255), -1)

                cv2.circle(roi_c, (cx_t1,cy_t1), 3, (0,0,255), -1)
                cv2.circle(roi_c, (cx_t2,cy_t2), 3, (0,0,255), -1)
                cv2.circle(roi_c,(cx_c,cy_c), 3, (0,0,255), -1)

            # context
            # write image context
            cv2.imwrite(os.path.join(stim_dir[0], imgs), img)

            # face only 
            # write image face_only
            cv2.imwrite(os.path.join(stim_dir[1], imgs), roi_c)

            trials_ctx.append([stim_dir[0], imgs, cx_c, cy_c, cx_t1, cy_t1, cx_t2, cy_t2])
            trials_fo.append([stim_dir[1], imgs, cx_c, cy_c, cx_t1, cy_t1, cx_t2, cy_t2])

    if os.path.isfile("failStim.txt") :
        print("MakeStim() failed for %s images, plese see 'failStim.txt'!"
            %(len(list(open("failStim.txt")))))
    else:
        print("MakeStim() completed!")

    return(trials_ctx, trials_fo)


def getTriallist(stim_dir):
    global trials_ctx, trials_fo, n_trials

    # check if triallist is already computed
    if not os.path.isfile("gcc-trials.csv"):
        # make stimuli
        makeStim()

        # for all trial characteristics
        stim_prod = list(product(t_id, t_pos))

        # stim_list w/o header row
        trials_ctx = list(product(trials_ctx, list(stim_prod)))

        trials_fo = list(product(trials_fo, list(stim_prod))) 

        for trial in range(len(trials_ctx)):
            trials_ctx[trial] = list(chain.from_iterable(trials_ctx[trial]))
            trials_fo[trial] = list(chain.from_iterable(trials_fo[trial]))

        # header
        header = ['stim_dir', 'img', 'c_x', 'c_y', 't1_x', 't1_y', 't2_x', 't2_y', 't_pos', 't_id']

        with open("gcc-trials.csv", "wb") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(trials_ctx)
            writer.writerows(trials_fo)

        # print(">>> new trial list:")
        # print("Context: ", trials_ctx[0], "Length: ", len(trials_ctx))
        # print("Face only: ", trials_fo[0], "Length: ", len(trials_fo))

    else:
        trial_list = []
        with open("gcc-trials.csv", "rb") as file:
            reader = csv.reader(file)
            for row in reader:
                trial_list.append(row)

        trial_list.pop(0)   # remove header

        # print(range(len(trial_list[0])))

        for trial in range(len(trial_list)):

            if trial_list[trial][0] == stim_dir[0]:
                trials_ctx.append(trial_list[trial])

            elif trial_list[trial][0] == stim_dir[1]:
                trials_fo.append(trial_list[trial])

            else:
                print(">>>> Error: Context can't be specified!")
                win.close()
                core.quit()

        # print(">>> read trial list:")
        # print("Context: ", trials_ctx[0], "Length: ", len(trials_ctx))
        # print("Face only: ", trials_fo[0], "Length: ", len(trials_fo))
    

    if len(trials_ctx) == len(trials_fo):
        n_trials = len(trials_ctx)
        #print("%d trials will be shown!" %(n_trials))
    
    else:
        print(">>>> Error: Unequal trial list for context-present (length: %d) and context-covered (length: %d)!" %(len(trials_ctx), len(trials_fo)) )
    
    return trials_ctx, trials_fo, n_trials


def pickTrial(trial_i):

    # for even subject id
    if float(session_info['subject']) % 2 == 0:
        # print(">>> Even subject id")

        # for even block
        if block_count % 2 == 0:    # bei graden Blöcken: context
            return addLables(trials_ctx[trial_i])

        # for odd block
        elif block_count % 2 == 1:
            return addLables(trials_fo[trial_i])

    # for odd subject id
    elif float(session_info['subject']) % 2 == 1:
        # print(">>> Odd subject id")

        # for even block
        if block_count % 2 == 0:    # bei graden Blöcken: face_only
            return addLables(trials_fo[trial_i])

        # for odd block
        elif block_count % 2 == 1:  # bei ungraden Blöcken: context
            return addLables(trials_ctx[trial_i])

    # if subject id neither odd nor even
    else:
        print(">>>> Error: Subject id neither odd nor even")
        win.close()
        core.quit()


def addLables(trial):

    '''
    # trial charactaristics
    c = cue
    t = target
    x,y = coords of center
    id = identifier
    pos = position
    '''

    if len(trial) == 10:
        stim_dir = trial[0]
        img = trial[1]
        c_x = int(trial[2]) - 640
        c_y = 480 - int(trial[3])
        t1_x = int(trial[4]) - 640
        t1_y = 480 - int(trial[5])
        t2_x = int(trial[6]) - 640
        t2_y = 480 - int(trial[7])
        t_id = trial[8]
        t_pos = trial[9]

        return(showTrial(stim_dir, img, c_x, c_y, t1_x, t1_y, t2_x, t2_y, t_id, t_pos))

    else:
        print(">>>> Error: len(Trial) has more then 9 entries: ", len(trial))
        win.close()
        core.quit()


def showTrial(stim_dir, img, c_x, c_y, t1_x, t1_y, t2_x, t2_y, t_id, t_pos):

    FixationCross = visual.TextStim(
        win, text="+", font='Arial', pos=[c_x, c_y],
        height=30, units='pix', color='white')

    SlowWarning = visual.TextStim(
        win, text=warningText, font='Arial', pos=[0, 0],
        height=30, units='pix', color='white')

    #print(stim_dir, img, t_pos, t_id)

    # find target positions:
    if t1_y < t2_y:
        t_pos_r = [t1_x, t1_y]
        t_pos_l = [t2_x, t2_y]

    elif t1_y > t2_y:
        t_pos_l = [t1_x, t1_y]
        t_pos_r = [t2_x, t2_y]

    else:
        print(">>>> Error: Can't find target positions!")
        win.close()
        core.quit()

    # target & position
    if (t_id == "E"):
        correctKey = response_key_E
        wrongKey = response_key_F
        if (t_pos == 'left'):
            target = visual.TextStim(
                win, text='E', font='Arial', units='pix',
                pos=t_pos_l, height=30)
            # print("Target coordinates: ", t_pos_l)
        elif (t_pos == 'right'):  # target right
            target = visual.TextStim(
                win, text='E', font='Arial', units='pix',
                pos=t_pos_r, height=30)
            # print("Target coordinates: ", t_pos_r)

        else:
            print('>>>> Error target E needs to be somewhere!!', t_pos)
            win.close()
            core.quit()

    elif (t_id == "F"):  # target F
        correctKey = response_key_F
        wrongKey = response_key_E
        if (t_pos == 'left'):  # target left
            target = visual.TextStim(
                win, text='F', font='Arial', units='pix',
                pos=t_pos_l, height=30)
            #print("Target coordinates: ", t_pos_l)

        elif (t_pos=='right'):  # target right
            target = visual.TextStim(
                win, text='F', font='Arial', units='pix',
                pos=t_pos_r, height=30)
            # print("Target coordinates: ", t_pos_r)
        else:
            print('>>>> Error target F needs to be somewhere!', t_pos)
            win.close()
            core.quit()

    else:
        print('>>>> Error, target id needs to be something! ', t_id)
        win.close()
        core.quit()

    reactionTime = core.Clock()
    FixationCross.draw()
    win.flip()

    #  wait iti before trial start
    wait(iti)
    FixationCross.draw()

    stimulus = visual.ImageStim(
        win, units='pix', size=[1280, 960], pos=(0, 0))
    stimulus.setImage(stim_dir + os.path.sep + img)
    stimulus.draw()

    win.flip()  # draw stimulus
    wait(soa)   # wait interStimuliIntervall milliseconds

    # # show display coords
    # top_left = visual.TextStim(
    #     win, text='[-640, 480]', font='Arial', units='pix',
    #     pos=[-640, 480], height=30)
    # top_right = visual.TextStim(
    #     win, text='[640, 480]', font='Arial', units='pix',
    #     pos=[640, 480], height=30)
    # bottom_right = visual.TextStim(
    #     win, text='[640, -480]', font='Arial', units='pix',
    #     pos=[640, -480], height=30)
    # bottom_left = visual.TextStim(
    #     win, text='[-640, -480]', font='Arial', units='pix',
    #     pos=[-640, -480], height=30)
    # center = visual.TextStim(
    #     win, text='X', font='Arial', units='pix',
    #     pos=[0, 0], height=30)

    # top_left.draw()
    # top_right.draw()
    # bottom_right.draw()
    # bottom_left.draw()
    # center.draw()


    stimulus.draw()
    target.draw()   # draw target

    # FixationCross.draw() # show fixation cross during target presentation

    win.flip()  # flip all faces with target
    reactionTime.reset()    # reaction time resets
    event.clearEvents()

    # Participant decides when to proceed, after onset of target!
    while (reactionTime.getTime() <= max_rt):
        if event.getKeys(correctKey):
            response = "correct"
            rt = reactionTime.getTime()
            # print('correct')
            # print(reactionTime.getTime())
            break

        elif event.getKeys(wrongKey):
            response = "incorrect"
            rt = reactionTime.getTime()
            # print('NOT correct')
            # print(reactionTime.getTime())
            break

        if event.getKeys(quit_key):
            print('quit')
            core.quit()

    while (reactionTime.getTime() > max_rt):    # after max_rt runs out
        if event.getKeys(correctKey):
            response = "correct"
            rt = reactionTime.getTime()
            # print('correct')
            # print(reactionTime.getTime())

            SlowWarning.draw()
            win.flip()
            wait(1.5)
            break

        elif event.getKeys(wrongKey):
            response = "incorrect"
            rt = reactionTime.getTime()
            # print('NOT correct')
            # print(reactionTime.getTime())

            SlowWarning.draw()
            win.flip()
            wait(1.5)

            break

        if event.getKeys(quit_key):
            print('quit')
            core.quit()

    return writeLog(stim_dir, img, c_x, c_y, t1_x, t1_y, t2_x, t2_y, t_id, t_pos, response, rt)


def writeLog(stim_dir, img, c_x, c_y, t1_x, t1_y, t2_x, t2_y, t_id, t_pos, correct_response, reaction_time):

    # check if file and folder already exist
    makeDirectory('log_files')
    fileName = 'log_files' + os.path.sep + exp_name + '_' + getDate() + '_' + session_info['subject'].zfill(2) + '.csv' 

    # generate file name with name of the experiment and subject
    # open file
    # 'a' = append; 'w' = writing; 'b' = in binary mode
    with open(fileName, 'ab') as saveFile:
        fileWriter = csv.writer(saveFile, delimiter='\t') # tab separated
        if os.stat(fileName).st_size == 0:  # if file is empty, insert header
            fileWriter.writerow(('exp', 'subject', 'session', 'experimenter', 'computer', 'date', 'block', 'trial', 'stim_dir', 'img', 'c_x', 'c_y', 't1_x', 't1_y', 't2_x', 't2_y', 't_pos', 't_id', 'soa', 'correct_response', 'reaction_time'))

        # write trial
        fileWriter.writerow((exp_name, session_info['subject'].zfill(2), session_info['session'], session_info['experimenter'], session_info['computer'], getDate(), block_count, trial_count, stim_dir, img, c_x, c_y, t1_x, t1_y, t2_x, t2_y, t_pos, t_id, soa, correct_response, reaction_time))


def runInstructions(instruction):

    for page in (range(len(instruction))):
        # print(instruction[page])
        instruction_complete = instruction[page]+text_continue
        showText(win, instruction_complete)


def runTrials(randomize=True):
    global trial_count, block_count
    
    while block_count <= 2:
        if (randomize):
            print("Trials randomized!")
            random.shuffle(trials_fo)
            random.shuffle(trials_ctx)

        for trial_i in range(n_trials):
            print("Picked trial %d of %d per block." %(trial_i+1, n_trials))
            pickTrial(trial_i)     # run trial i
            #print('> trial_count: %d' %(trial_count))
            trial_count += 1
        
        showText(win, pause_text + text_continue)
        #print('>> block_count: %d' %(block_count))
        block_count += 1
        trial_count = 1  # restart trial_count each block

    print("All trials shown!")

def runExperiment():
    prepareExperiment()

    print("Subject ID is: %s" %(session_info['subject']))
    print("Experimenter is: %s" %(session_info['experimenter']))
    print("Computer is: %s" %(session_info['computer']))

    runInstructions(instruction_pages)
    runTrials()
    showText(win, goodbye_text)

    win.close()
    core.quit()


# Execution
runExperiment()
