from BaumWelch import *
from BaumWelchLR import *
import numpy as np
import pandas as pd
import csv
import math
from numpy import linspace,exp
from numpy.random import randn
from scipy.interpolate import UnivariateSpline
import sys
model = sys.argv[1]
m = int(sys.argv[2])
tol = float(sys.argv[3])
m2 = int(sys.argv[4])
D = int(sys.argv[5])
print(model, m, tol, m2, D)


def encode(x, code):
    output = np.array([int(np.where(code == x[i])[0]) for i in range(0, len(x))])
    return output

def decode(x, code):
    output = np.zeros(len(x))
    for i in range(0, len(x)):
        output[i] = code[x[i]]
    return output

def hmm(n, pi, phi, Tmat, T2mat, T3mat, code, model):
    m = Tmat.shape[0]
    k = phi.shape[1]
    zstates = np.arange(0, m)
    xstates = np.arange(0, k)
    z = np.zeros(n, dtype = int)
    x = np.zeros(n, dtype = int)
    z[0] = np.random.choice(zstates, size = 1, p = pi)
    if model == 'first_order':
        for j in range(1, n):
            z[j] = np.random.choice(zstates, size = 1, p = Tmat[z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size = 1, p = phi[z[i], :])
     
    if model == 'second_order':
        z[1] = np.random.choice(zstates, size = 1,  p = Tmat[z[0], :])
        for j in range(2, n):
            z[j] = np.random.choice(zstates, size = 1,  p = T2mat[z[j-2],z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size =1, p = phi[z[i], :])
    if model == 'third_order':
        z[1] = np.random.choice(zstates, size = 1,  p = Tmat[z[0], :])
        z[2] = np.random.choice(zstates, size = 1,  p = T2mat[z[0],z[1], :])
        for j in range(3, n):
            z[j] = np.random.choice(zstates, size = 1,  p = T3mat[z[j-3],z[j-2],z[j-1], :])
        for i in range(0, n):
            x[i] = np.random.choice(xstates, size =1, p = phi[z[i], :])
    output = decode(x, code)
    return(output,z)

def hmm_2hidden(n, pi, phi, Tmat, A, B, code):
    N = A.shape[0]
    M = B.shape[1]
    k = phi.shape[1]
    zstates = np.arange(0, N*M)
    rstates = np.arange(0,N)
    sstates = np.arange(0,M)
    xstates = np.arange(0, k)
    R = np.zeros(n, dtype = int)
    S = np.zeros(n, dtype = int)
    x = np.zeros(n, dtype = int)
    z = np.random.choice(zstates, size = 1, p = pi)
    S[0] = z % M
    R[0] = int((z - z%M)/M)
    for j in range(1, n):
        R[j] = np.random.choice(rstates, size = 1, p = A[R[j-1], :])
        S[j] = np.random.choice(sstates, size = 1, p = B[R[j], S[j-1],:])
    for i in range(0, n):
        x[i] = np.random.choice(xstates, size = 1, p = phi[S[i], :])
    output = decode(x, code)
    return(output)

def hmmARHMM(n, pi, phi, Tmat, psi, code):
    m = Tmat.shape[0]
    k = phi.shape[1]
    zstates = np.arange(0, m)
    xstates = np.arange(0, k)
    z = np.zeros(n, dtype = int)
    x = np.zeros(n, dtype = int)
    z[0] = np.random.choice(zstates, size = 1, p = pi)
    x[0] = np.random.choice(xstates, size = 1, p = phi[z[0], :])
    
    for j in range(1, n):
        z[j] = np.random.choice(zstates, size = 1, p = Tmat[z[j-1], :])
    for i in range(0, n):
        x[i] = np.random.choice(xstates, size = 1, p = psi[z[i], :, x[i-1]])
    output = decode(x, code)
    return(output)

def hmmSCFG(n, m, k, t, e, code):
    ##Assumes binary tree structure
    zstates = range(0, m)
    xstates = range(0, k)
    zOld = []
    x = np.zeros(n)
    zProbs = np.sum(t, axis = (1,2))
    t = t/np.sum(t, axis = (1,2))[:, None, None]
    e = e/np.sum(e, axis = 1)[:, None]
    zOld.append(np.random.choice(zstates, size = 1, p = zProbs/np.sum(zProbs)))
    levels = math.ceil(np.log2(n))
    count = 0
    for i in range(1, levels+1):
        zNew = []
        for j in zOld:
            non_prob = np.random.choice(t[j, :, :].ravel(), size = 1, p = t[j, :, :].ravel())
            ztemp = j
            zNew.append(np.where(t[ztemp, :, :] == non_prob)[0])
            zNew.append(np.where(t[ztemp, :, :] == non_prob)[1])
        zOld = zNew
    zOld = np.asarray(zOld).flatten() #.reshape((1,n))
    for i in zOld:
        x[count] = np.random.choice(xstates, size = 1, p = e[i,:].ravel())
        count+=1
        if count >= n:
            break
    output = decode(x, code)
    return output

import scipy 
import editdistance
import sklearn.metrics
import statsmodels.api as sm
# Input a pandas series 
def ent(data):
    p_data= np.unique(data, return_counts = True)[1]/len(data) # calculates the probabilities
    entropy=scipy.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy

def musicality(tn, time):
    dissonance = 0
    usual_dis = 0
    octave = 0
    dis_ints = np.array([1,-1,2,-2,10,-10,11,-11])
    usual_dis_ints = np.array([5,-5,6,-6])
    for t in np.unique(time):
        consid_notes = tn[tn[:,0] == t,1]
        if len(consid_notes) > 0:
            if np.any(consid_notes[1:] - consid_notes[0] in dis_ints):
                dissonance +=1
            if np.any(consid_notes[1:] - consid_notes[0] in usual_dis_ints):
                usual_dis +=1

    for t in range(len(np.unique(time))-3):
        consid_notes = tn[tn[:,0] == np.unique(time)[t],1] 
        consid_notes_low = consid_notes[np.where(consid_notes < 60)[0]]
        consid_notes_high = consid_notes[np.where(consid_notes >= 60)[0]]
        step_ahead = tn[tn[:,0] == np.unique(time)[t+1],1] 
        step_ahead_low = step_ahead[np.where(step_ahead < 60)[0]]
        step_ahead_high = step_ahead[np.where(step_ahead >= 60)[0]]
        if len(consid_notes_low) > 0 and len(step_ahead_low) > 0:
            for q in step_ahead_low:
                if np.any(q - consid_notes_low > 12) or np.any(q - consid_notes_low < -12):
                    octave +=1
        if len(consid_notes_high) > 0 and len(step_ahead_high) > 0:
            for q in step_ahead_high:
                if np.any(q - consid_notes_high > 12) or np.any(q - consid_notes_high < -12):
                    octave +=1

    return(dissonance/tn.shape[0],usual_dis/tn.shape[0], octave/tn.shape[0] )


def music_metrics(notes, newNotes, possibleNotes, time, n, k):
    entropy = ent(newNotes)
    edit_dist = editdistance.eval(notes, newNotes.astype(int))/n
    mutual_info = sklearn.metrics.mutual_info_score(notes, newNotes.astype(int))
    unique_new_notes, note_counts = np.unique(newNotes, return_counts = True) 
    if len(unique_new_notes) != k:
        add_notes = list(set(possibleNotes) - set(unique_new_notes))
        for i in add_notes:
            if np.where(possibleNotes == i)[0] > len(note_counts):
                note_counts = np.append(note_counts, np.where(possibleNotes == i)[0], 0)
            else:
                note_counts = np.insert(note_counts, np.where(possibleNotes == i)[0], 0)
    note_counts = note_counts/n
    note_acf = sm.tsa.stattools.acf(newNotes)
    note_pacf = sm.tsa.stattools.pacf(newNotes)
    indicies = []
    for i in np.unique(newNotes):
        indicies.append(np.where(newNotes == i)[0][::2])
    tn = np.zeros((len(np.concatenate(indicies, axis=0)),2))
    tn[:,0] = time[np.sort(np.concatenate(indicies, axis=0))]
    tn[:,1] = newNotes[np.sort(np.concatenate(indicies, axis=0))]
    dissonance, usual_dissonance, octaves = musicality(tn, time)
    return(np.concatenate((np.array([entropy, edit_dist, mutual_info, dissonance, usual_dissonance, 
                          octaves]), note_counts, note_acf, note_pacf), axis = 0))


def hmm_compose(input_filename, output_filename, line_skip, model, m,  tol, m2 = None, D = 1):
    with open(input_filename,encoding = "ISO-8859-1") as fd:
    #with open(input_filename,'rU') as fd:
        reader=csv.reader(fd)
        rows= [row for idx, row in enumerate(reader)]
    song = pd.DataFrame(rows)

    #Find when notes and velocities begin
    end_song = np.where(song.ix[:,2] == ' End_track')[0]
    end_track = end_song[np.where(np.array(end_song > line_skip))[0][0]]-1

    #Remove control_c values
    control_c = song[song.ix[:,2] == ' Control_c']
    tempo = song[song.ix[:,2] == ' Tempo']
    key = song[song.ix[:,2] == ' Key_signature']

    # Select header and footer
    header = song.ix[:line_skip,:]
    footer = song.ix[end_track+1:,:]
    
    song_mod = song[(song.ix[:,2] == ' Note_on_c') | (song.ix[:,2] == ' Note_off_c')] 
    #song_mod = song[song.ix[:,2] != ' Control_c']


    notes = np.array(song_mod.ix[line_skip:end_track,4])
    notes = notes.astype(int)
    velocity = np.array(song_mod.ix[line_skip:end_track,5])
    velocity = velocity.astype(int)
    commands = np.array(song_mod.ix[line_skip:end_track, 2])
    time = np.array(song_mod.ix[line_skip:end_track, 1])
    time = time.astype(int)
    #Find possible unique notes and velocities
    possibleNotes = np.unique(notes)
    possibleVelocities =  np.unique(velocity)

    k = len(possibleNotes)
    xNotes = encode(notes, possibleNotes)
    n = len(xNotes)
    
    newNotes = np.zeros(shape = (n,5),dtype = int)
    newZ = np.zeros(shape = (n,5),dtype = int)
    metrics = np.zeros(shape = (1001, 3+k+40+40+3))
    orig_metrics = music_metrics(notes, notes, possibleNotes, time, n, k)
    metrics = np.zeros(shape = (1001, len(orig_metrics)))
    metrics[0,:] = orig_metrics
    it_save = np.random.choice(np.arange(1000), size = 2)
    save_count = 0
    

    #Run BaumWelch for specified model
    if model == 'random':
        vals = np.random.rand(m)
        pi1 = vals/np.sum(vals)
        Tmat1 = np.zeros(shape = (m, m))
        phi1 = np.zeros(shape = (m, k))
        vals1 = np.random.rand(m,m)
        vals2 = np.random.rand(m,k)
        Tmat1 = vals1/np.sum(vals1, axis=1)[:,None]
        phi1 = vals2/np.sum(vals2, axis = 1)[:,None]
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
        
    if model == 'first_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
       
        
    if model == 'first_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
        
        
    if model == 'second_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        it2, T2mat = second_orderLR(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, T2mat, None,
                                                                 possibleNotes,'second_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
        
    
    if model == 'second_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, T2mat = second_order(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, T2mat, None,
                                                                 possibleNotes,'second_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)

    if model == 'third_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        it2, T2mat = second_orderLR(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        it3, T3mat = third_orderLR(n, m, k, xNotes, pi1, Tmat1, T2mat, phi1, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat,
                                                                 possibleNotes,'third_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
        
    
    if model == 'third_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, T2mat = second_order(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        it3, T3mat = third_order(n, m, k, xNotes, pi1, Tmat1, T2mat, phi1, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat,
                                                                 possibleNotes,'third_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)

    if model == 'two_hidden_states':
        it1, p1, pi1, phi1, Tmat1, A1, B1 = two_hidden_states(n, m, m2, k, xNotes, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count] = hmm_2hidden(n, pi1, phi1, Tmat1, A1, B1, possibleNotes)
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm_2hidden(n, pi1, phi1, Tmat1, A1, B1, possibleNotes)
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
       
        
    if model == 'HMMSDO':
        it1, p1, pi1, phi1, Tmat1 = HMMSDO(n, m, k, xNotes, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'HMMSDO')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'HMMSDO')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
        
     
    if model == 'HSMM':
        it1, p1, pi1, phi1, Tmat1 =  HSMM(n, m, k, xNotes, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
        
    
    if model == 'ARHMM':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, p2, psi = first_orderARHMM(n, m, k, np.log(pi1), np.log(Tmat1), np.log(phi1), xNotes,  tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count] = hmmARHMM(n, pi1, phi1, Tmat1, psi, possibleNotes)
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmmARHMM(n, pi1, phi1, Tmat1, psi, possibleNotes)
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
        
    
    if model == 'SCFG':
        it1, p1, t, e = SCFG(n,m, k, xNotes,tol)
        newNotes = hmmSCFG(n,m,k,t,e, possibleNotes)
    
    if model == 'NSHMM':
        #for i in range(10):
        temp_notes = NSHMM( n,  m,  k,  D, xNotes, tol, 1000)
        for i in range(1000):
            temp_notes[i,:] = decode(temp_notes[i,:], possibleNotes)
            print(i)
            if i in it_save:
                newNotes[:,save_count] = temp_notes[i,:]
                save_count += 1
            metrics[i+1,:] = music_metrics(notes, temp_notes[i,:], possibleNotes, time, n, k)
        
    
    if model == 'ANN_HMM':
        H = 5 
        it1, p1, pi1, phi1, Tmat1 =  ANN_HMM( n,  m,  k,  H,  xNotes, tol)
        for i in range(1000):
            print(i)
            if i in it_save:
                newNotes[:,save_count], newZ[:,save_count] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
                temp_notes = newNotes[:,save_count]
                save_count += 1
            else:
                temp_notes = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')[0]
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
    
        
    #### EDITED 15,10,5###
    if model == 'factorial':
        xstates = range(0, k)
        noteArray = np.zeros(shape = (3, n))
        it1, p1, pi1, phi15, Tmat1 = first_order(n, 15, k, xNotes, tol)
        zStar15 = Viterbi(n, 15, k, np.log(pi1), np.log(Tmat1), np.log(phi15), xNotes)
        zStar15 = np.array(zStar15).astype(int)
        it1, p1, pi1, phi10, Tmat1 = first_order(n, 10, k, xNotes, tol)
        zStar10 = Viterbi(n, 10, k, np.log(pi1), np.log(Tmat1), np.log(phi10), xNotes)
        zStar10 = np.array(zStar10).astype(int)
        it1, p1, pi1, phi5, Tmat1 = first_order(n, 5, k, xNotes, tol)
        zStar5 = Viterbi(n, 5, k, np.log(pi1), np.log(Tmat1), np.log(phi5), xNotes)
        zStar5 = np.array(zStar5).astype(int)
        
        for i in range(1000):
            for j in range(0, n):
                noteArray[0,j] = np.random.choice(xstates, size = 1, p = phi15[zStar15[j], :])
                noteArray[1,j] = np.random.choice(xstates, size = 1, p = phi10[zStar10[j], :])
                noteArray[2,j] = np.random.choice(xstates, size = 1, p = phi5[zStar5[j], :])
            temp_notes = np.rint(np.mean(noteArray, axis=0)).astype(int)
            temp_notes = decode(temp_notes, possibleNotes)   
            print(i)
            if i in it_save:
                newNotes[:,save_count] = temp_notes
                save_count += 1
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
    
    
    if model == 'layered':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        zStar1 = Viterbi(n, m, k, np.log(pi1), np.log(Tmat1), np.log(phi1), xNotes)
        zStar1 = np.array(zStar1).astype(int)
        it2, p2, pi2, phi2, Tmat2 = first_order(n, m, m, zStar1, tol)
        zStar2 = Viterbi(n, m, m, np.log(pi2), np.log(Tmat2), np.log(phi2), zStar1)
        zStar2 = np.array(zStar2).astype(int)
        it3, p3, pi3, phi3, Tmat3 = first_order(n, m, m, zStar2, tol)
        zStar3 = Viterbi(n, m, m, np.log(pi3), np.log(Tmat3), np.log(phi3), zStar2)
        zStar3 = np.array(zStar3).astype(int)
        output = np.zeros(shape = (3,n), dtype = int)
        xstates = range(0, k)
        zstates = range(0, m)
        for i in range(1000):
            for j in range(0,n):
                output[2, j] = np.random.choice(zstates, size = 1, p = phi3[zStar3[j], :])
                output[1, j] = np.random.choice(zstates, size = 1, p = phi2[output[2, j], :])
                output[0, j] = np.random.choice(xstates, size = 1, p = phi1[output[1, j], :])
            temp_notes = decode(output[0,:], possibleNotes).astype(int)
            print(i)
            if i in it_save:
                newNotes[:,save_count] = temp_notes
                save_count += 1
            metrics[i+1,:] = music_metrics(notes, temp_notes, possibleNotes, time, n, k)
    
    # Use splines to interpolate the velocities
    for song_num in range(5):
        newVelocities = np.zeros(len(newNotes[:, song_num]))
        y = velocity[np.nonzero(velocity)]
        indicies = []
        for i in np.unique(newNotes[:, song_num]):
            indicies.append(np.where(newNotes[:, song_num] == i)[0][::2])

        unlist = [item for sublist in indicies for item in sublist]
        unlist.sort()
        X = np.array(range(0,len(y)))
        s = UnivariateSpline(X, y, s=300) #750
        xs = np.linspace(0, len(y), len(unlist), endpoint = True)
        ys = s(xs)    
        newVelocities[np.array(unlist)] = np.round(ys).astype(int)
        #Fix entries that are too small or too large due to spline overfitting
        newVelocities[np.where(newVelocities < 0)[0]] = y[-1]
        newVelocities = newVelocities.astype(int)    

    #     newNotes_vals, newNotes_counts = np.unique(newNotes[:, song_num].astype(int), return_counts=True)
    #     for i in set(possibleNotes) - set(newNotes_vals):
    #         newNotes_vals = np.append(newNotes_vals, i)
    #         newNotes_counts = np.append(newNotes_counts, 0)
    #     #     print(newNotes_vals)
    #     #     print(newNotes_counts)
    #     #     print(Tmat1)
    #     #     print(T2mat)
    #     # Fix notes that are on for too long
    #     for i in range(0,len(possibleNotes)):
    #         if newNotes_counts[i] < 25:
    #             max_duration = np.max(np.ediff1d(time[np.where(notes == newNotes_vals[i])[0]].astype(int))[::2])
    #             new_duration = np.ediff1d(time[np.where(newNotes == newNotes_vals[i])[0]].astype(int))[::2]
    #             too_long = np.where(new_duration > max_duration)[0]
    #             inter = time[np.where(newNotes[:, song_num] == newNotes_vals[i])[0]]
    #             inter[2*too_long] = max_duration + inter[2*too_long]
    #             time[np.where(newNotes == newNotes_vals[i])[0]] = inter

        # Fix note on/off commands
        commands[np.where(newVelocities !=0)[0]] = ' Note_on_c'
        commands[np.where(newVelocities ==0)[0]] = ' Note_off_c'

        # Add new notes and velocities to original CSV
        newsong = song_mod
        inter_df = pd.DataFrame([time, commands, newNotes[:, song_num], newVelocities]).transpose()
        inter_df.columns = [ 'time', 'commands',  'notes', 'vel']
        inter_df.time = inter_df.time.astype(int)
        inter_df = inter_df.sort(columns = 'time')
        newsong.ix[line_skip:end_track,2] = np.array(inter_df.commands)
        newsong.ix[line_skip:end_track,4] = np.array(inter_df.notes)
        newsong.ix[line_skip:end_track,5] = np.array(inter_df.vel)
        newsong.ix[line_skip:end_track,1] = np.array(inter_df.time)

        newsong = newsong.append(control_c)
        newsong = newsong.append(tempo)
        newsong = newsong.append(key)
        newsong = newsong.append(header)
        newsong = newsong.append(footer)
        newsong = newsong.sort()
        split = output_filename.split('.')
        name = output_filename.split('/')[1].split('.')[0]
        output_f1 = split[0] + str(song_num)  + '__'+ model + '_' + str(m)+ '_D' + str(D)+ '_m2-'+str(m2)+  '-tol' +str(tol)+'.' + split[1]
        newsong.to_csv(output_f1, header = None, index = False)


        # Save z values
        output_f2 = 'z/'+ name + str(song_num)  + '__'+ model + '_' + str(m)+ '_D' + str(D)+ '_m2-'+str(m2)+  '-tol' +str(tol)+'.' + split[1]
        pd.DataFrame(newZ).to_csv(output_f2, header = None, index = False)


        # Save acf/pacf plots
        #output_f3 = 'plots/'+ name + str(song_num)  + '__'+ model + '_' + str(m)+ '_D' + str(D)+ '_m2'+str(m2)+  '-tol' +str(tol)+'.png' 
        #fig = plt.figure(figsize=(12,8))
        #ax1 = fig.add_subplot(211)
        #fig = sm.graphics.tsa.plot_acf(newNotes[:, song_num], lags=40, ax=ax1)
        #ax2 = fig.add_subplot(212)
        #fig = sm.graphics.tsa.plot_pacf(newNotes[:, song_num], lags=40, ax=ax2)
        #plt.savefig(output_f3, bbox_inches='tight')


    # Save metrics values
    output_f4 = 'metrics/'+ name  + '__'+ model + '_' + str(m)+ '_D' + str(D)+ '_m2-'+str(m2)+  '-tol' +str(tol)+'.' + split[1]
    pd.DataFrame(metrics).to_csv(output_f4, header = None, index = False)


#np.random.seed(77)
#print('Baroque\n')
#hmm_compose('OriginalCSV/book1-prelude02.csv', 'NewCSV/book1-prelude02.csv', 56, model, m, tol, m2, D)    
#print('Bach Prelude')
#hmm_compose('OriginalCSV/book1-fugue02.csv', 'NewCSV/book1-fugue02.csv', 47, model, m, tol, m2, D)    
#print('Bach Fugue')
#hmm_compose('OriginalCSV/invention2part-no4.csv', 'NewCSV/invention2part-no4.csv', 22, model, m, tol, m2, D)    
#print('Bach Invention')

#print('Classical')
#hmm_compose('OriginalCSV/haydn-piano-sonata-31-1.csv', 'NewCSV/haydn-piano-sonata-31-1.csv', 132, model, m, tol, m2, D)    
#print('Haydn')
#hmm_compose('OriginalCSV/alla-turca.csv', 'NewCSV/alla-turca.csv', 126, model, m, tol, m2, D)    
#print('Mozart')
#hmm_compose('OriginalCSV/beethoven-minuet-in-G.csv', 'NewCSV/beethoven-minuet-in-G.csv', 105, model, m, tol, m2, D)    
#print('Beethoven Minuet')
#hmm_compose('OriginalCSV/moonlight-movement1.csv', 'NewCSV/moonlight-movement1.csv', 96, model, m, tol, m2, D)
#print('moonlight-movement1 done')

#print('Romantic')
#hmm_compose('OriginalCSV/brahms-intermezzo-op118-no2.csv', 'NewCSV/brahms-intermezzo-op118-no2.csv', 
#            214, model, m, tol, m2, D)    
#print('Brahms Intermezzo')
#hmm_compose('OriginalCSV/brahms-waltz-15.csv', 'NewCSV/brahms-waltz-15.csv', 70, model, m, tol, m2, D)    
#print('Brahms Waltz')
#hmm_compose('OriginalCSV/chopin-funeral-march.csv', 'NewCSV/chopin3.csv', 132, model, m, tol, m2, D)          
#print("funeral Done")
#hmm_compose('OriginalCSV/chopin-etude-op10-no4.csv', 'NewCSV/chopin-etude-op10-no4.csv', 139, model, m, tol, m2, D)    
#print('Chopin etude')
#hmm_compose('OriginalCSV/frederic-chopin-nocturne-no20.csv', 'NewCSV/frederic-chopin-nocturne-no20.csv', 
#            133, model, m, tol, m2, D)    
#print('Chopin nocturne')
#hmm_compose('OriginalCSV/chpn_op7_2_format0.csv', 'NewCSV/chpn_op7_2_format0.csv', 24, model, m, tol, m2, D)    
#print('Chopin Mazurka')
#hmm_compose('OriginalCSV/muss_1_format0.csv', 'NewCSV/muss_1_format0.csv', 30, model, m, tol, m2, D)    
#print('Pictures at an Exhibition')
#hmm_compose('OriginalCSV/Songwithoutwords06.csv', 'NewCSV/Songwithoutwords06.csv', 42, model, m, tol, m2, D)    
#print('Songs 6')
#hmm_compose('OriginalCSV/Songwithoutwords53.csv', 'NewCSV/Songwithoutwords53.csv', 39, model, m, tol, m2, D)    
#print('Songs 53')
#hmm_compose('OriginalCSV/ty_november_format0.csv', 'NewCSV/ty_november_format0.csv', 19, model, m, tol, m2, D)    
#print('Seasons')
#hmm_compose('OriginalCSV/lszt_hr2.csv', 'NewCSV/lszt_hr2.csv', 9, model, m, tol, m2, D)
#print('lszt_hr2 done')

#print('Late Romantic/Impressionist')
#hmm_compose('OriginalCSV/rac_op32_1_format0.csv', 'NewCSV/rac_op32_1_format0.csv', 27, model, m, tol, m2, D)
#print('rac_op32_1_format0 done')
#hmm_compose('OriginalCSV/rac_op33_8_format0.csv', 'NewCSV/rac_op33_8_format0.csv', 12, model, m, tol, m2, D)
#print('lszt_hr2 done')
#hmm_compose('OriginalCSV/clair-de-lune.csv', 'NewCSV/clair-de-lune.csv', 192, model, m, tol, m2, D)
#print('clair-de-lune done')
#hmm_compose('OriginalCSV/Golliwogs-Cakewalk.csv', 'NewCSV/Golliwogs-Cakewalk.csv', 144, model, m, tol, m2, D)
#print('lszt_hr2 done')
#hmm_compose('OriginalCSV/edward-elgar-enigma-variations-nimrod-for-piano.csv', 'NewCSV/elgar-nimrod.csv', 
#            54, model, m, tol, m2, D)
#print('Elgar done')
#hmm_compose('OriginalCSV/ravotoc.csv', 'NewCSV/ravotoc.csv', 10, model, m, tol, m2, D)
#print('Ravel done')

#print('Contemporary/Modern')
#hmm_compose('OriginalCSV/prkson61.csv', 'NewCSV/prkson61.csv', 173, model, m, tol, m2, D)
#print('prkson61 done')
#hmm_compose('OriginalCSV/Westworld_Theme.csv', 'NewCSV/Westworld_Theme.csv', 22, model, m, tol, m2, D)          
#print("Westworld Done") 

print('Hymns/Carols')
#hmm_compose('OriginalCSV/Jupiter.csv', 'NewCSV/Jupiter.csv', 39, model, m, tol, m2, D)
#print('Jupiter done')
#hmm_compose('OriginalCSV/pachelbel.csv', 'NewCSV/pachelbel.csv', 27, model, m, tol, m2, D)
#print('pachelbel done')
#hmm_compose('OriginalCSV/beethoven-symphony9-4-ode-to-joy-piano-solo.csv', 'NewCSV/ode-to-joy.csv', 23, model, m, tol, m2, D)    
#print('ode to joy done')
#hmm_compose('OriginalCSV/carol-of-the-bells.csv', 'NewCSV/carol-of-the-bells.csv', 26, model, m, tol, m2, D)              
#print('carol-of-the-bells done')
#hmm_compose('OriginalCSV/deutschlandlied.csv', 'NewCSV/deutschlandlied.csv', 30, model, m, tol, m2, D)                          
#print('deutschlandlied done')
hmm_compose('OriginalCSV/hark-the-herald-angels-sing.csv', 'NewCSV/hark-the-herald-angels-sing.csv', 47, model, m, 
            tol, m2, D)      
print('hark-the-hearald-angels-sing done')
#hmm_compose('OriginalCSV/il-est-ne-le-divin-enfant-keyboard.csv', 'NewCSV/il-est-ne.csv', 23, model, m, tol, m2, D)           
#print('il est ne done')
#hmm_compose('OriginalCSV/in-the-bleak-midwinter.csv', 'NewCSV/in-the-bleak-midwinter.csv', 19, model, m, tol, m2, D)           
#print('in-the-bleak-midwinter done')
#hmm_compose('OriginalCSV/old-100th.csv', 'NewCSV/old-100th.csv', 63, model, m, tol, m2, D)                                     
#print('old 100th done')
#hmm_compose('OriginalCSV/Third-Mode-Melody.csv', 'NewCSV/Tallis.csv', 73, model, m, tol, m2, D)                                 
#print('Tallis done')
hmm_compose('OriginalCSV/we-three-kings-keyboard.csv', 'NewCSV/we-three-kings.csv', 42, model, m, tol, m2, D)                    
print('we three kings done')
#hmm_compose('OriginalCSV/swing-low-sweet-chariot.csv', 'NewCSV/swing-low-sweet-chariot.csv', 24, 
#            model, m, tol, m2, D)                    
#print('swing-low-sweet-chariot done')
#hmm_compose('OriginalCSV/greensleeves.csv', 'NewCSV/greensleeves.csv', 53, model, m, tol, m2, D)                        
##print('greensleeves done')
#hmm_compose('OriginalCSV/when-johnny-comes-marching-home.csv', 'NewCSV/when-johnny-comes-marching-home.csv', 
#            24, model, m, tol, m2, D)                    
#print('when-johnny-comes-marching-home done')
#hmm_compose('OriginalCSV/shall-we-gather-at-the-river.csv', 'NewCSV/shall-we-gather-at-the-river.csv', 
#            25, model, m, tol, m2, D)                    
#print('shall-we-gather-at-the-river done')

  
  
#hmm_compose('OriginalCSV/Dvorak9Largo.csv', 'NewCSV/Dvorak9Largo.csv', 98, model, m, tol, m2)
#print('Dvorak9Largo done')
#hmm_compose('OriginalCSV/chpson3b-sonata-scherzo.csv', 'NewCSV/chpson3b-sonata-scherzo.csv', 11, 
#            model, m, tol)
#print('chpson3b-sonata-scherzo done')
#hmm_compose('OriginalCSV/dansa-villa-lobos.csv', 'NewCSV/dansa-villa-lobos.csv', 32, model, m, tol, m2)
#print('dansa-villa-lobos done')
#hmm_compose('OriginalCSV/furelise.csv', 'NewCSV/furelise.csv', 80, model, m, tol, m2)
#print('furelise done')
#hmm_compose('OriginalCSV/beethoven-short.csv', 'NewCSV/ode-to-joy-short.csv', 23, model, m, tol)
#print('ode to joy done')
#hmm_compose('OriginalCSV/beethoven-symphony9-4-ode-to-joy-piano-solo.csv', 'NewCSV/ode-to-joy.csv', 23, model, m, tol, m2)      
#print('ode to joy done')
#hmm_compose('OriginalCSV/beethoven-symphony9-short.csv', 'NewCSV/ode-to-joy-short2.csv', 23, model, m, tol, m2)      
#print('ode to joy done')
#hmm_compose('OriginalCSV/carol-of-the-bells.csv', 'NewCSV/carol-of-the-bells.csv', 26, model, m, tol, m2)                       
#print('carol-of-the-bells done')
#hmm_compose('OriginalCSV/coventry-carol-keyboard.csv', 'NewCSV/coventry-carol.csv', 27, model, m, tol, m2)
#print('coventry carol done')
#hmm_compose('OriginalCSV/deutschlandlied.csv', 'NewCSV/deutschlandlied.csv', 30, model, m, tol, m2)                          
#print('deutschlandlied done')
#hmm_compose('OriginalCSV/edvard-grieg-peer-gynt1-morning-mood-piano.csv', 'NewCSV/morning-mood.csv', 60, model, m, tol, m2)     
#print('morning mood done')
#hmm_compose('OriginalCSV/god-rest-you-merry-gentlemen-short.csv', 'NewCSV/god-rest-you-merry-gentlemen-short.csv', 30, model, m, tol, m2)   
#print('god-rest-you-merry-gentleman done')
#hmm_compose('OriginalCSV/greensleeves-short.csv', 'NewCSV/greensleeves-short.csv', 53, model, m, tol, m2)                        
#print('greensleeves done')
#hmm_compose('OriginalCSV/gregorian-dies-irae.csv', 'NewCSV/gregorian-dies-irae.csv', 56, model, m, tol, m2)                     
#print('gregorian-dies-irae done')
#hmm_compose('OriginalCSV/handel-water-music-hornpipe-piano.csv', 'NewCSV/water-music.csv', 35, model, m, tol, m2)               
#print('water music done')
#hmm_compose('OriginalCSV/hark-the-herald-angels-sing.csv', 'NewCSV/hark-the-herald-angels-sing.csv', 47, model, m, tol, m2)      
#print('hark-the-hearald-angels-sing done')
#hmm_compose('OriginalCSV/il-est-ne-le-divin-enfant-keyboard.csv', 'NewCSV/il-est-ne.csv', 23, model, m, tol, m2)               
#print('il est ne done')
#hmm_compose('OriginalCSV/in-the-bleak-midwinter.csv', 'NewCSV/in-the-bleak-midwinter.csv', 19, model, m, tol, m2)               
#print('in-the-bleak-midwinter done')
#hmm_compose('OriginalCSV/deutschlandlied.csv', 'NewCSV/deutschlandlied.csv', 30, model, m, tol, m2)                              
#print('deutschlandlied done')
#hmm_compose('OriginalCSV/leaning-on-the-everlasting-arms.csv', 'NewCSV/leaning-on-the-everlasting-arms.csv', 32, model, m, tol, m2)                                       
#print('leaning-on-the-everlasting-arms done')
#hmm_compose('OriginalCSV/morning-has-broken.csv', 'NewCSV/morning-has-broken.csv', 22, model, m, tol, m2)                       
#print('morning-has-broken done')
#hmm_compose('OriginalCSV/old-100th.csv', 'NewCSV/old-100th.csv', 63, model, m, tol, m2)                                       
#print('old 100th done')
#hmm_compose('OriginalCSV/once-in-royal.csv', 'NewCSV/once-in-royal.csv', 23, model, m, tol, m2)                                 
#print('once in royal done')
#hmm_compose('OriginalCSV/sugar-plum-fairy-piano.csv', 'NewCSV/sugar-plum-fairy.csv', 28, model, m, tol, m2)                      
#print('sugar plum fairy done')
#hmm_compose('OriginalCSV/the-blue-danube.csv', 'NewCSV/the-blue-danube.csv', 32, model, m, tol, m2)                              
#print('the-blue-danube done')
#hmm_compose('OriginalCSV/Third-Mode-Melody.csv', 'NewCSV/Tallis.csv', 73, model, m, tol, m2)                                    
#print('Tallis done')
#hmm_compose('OriginalCSV/we-three-kings-keyboard.csv', 'NewCSV/we-three-kings.csv', 42, model, m, tol, m2)                      
#print('we three kings done')
#hmm_compose('OriginalCSV/Westworld_Theme.csv', 'NewCSV/Westworld_Theme.csv', 22, model, m, tol, m2)          
#print("Westworld Done") 
#hmm_compose('OriginalCSV/chopin-funeral-march.csv', 'NewCSV/chopin3.csv', 132, model, m, tol, m2)          
#print("Chopin Done") 


#hmm_compose('OriginalCSV/book1-prelude01.csv', 'NewCSV/book1-prelude01.csv', 83, model, m, tol, m2)    
#hmm_compose('OriginalCSV/book1-prelude02.csv', 'NewCSV/book1-prelude02.csv', 56, model, m, tol, m2)          
#hmm_compose('OriginalCSV/book1-prelude03.csv', 'NewCSV/book1-prelude03.csv', 61, model, m, tol, m2)          

