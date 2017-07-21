from BaumWelch import *
from BaumWelchLR import *
from TVAR import *
import numpy as np
import pandas as pd
import csv
import math
from numpy import linspace,exp
from numpy.random import randn
from scipy.interpolate import UnivariateSpline
import scipy
import sys
model = sys.argv[1]
m = int(sys.argv[2])
tol = float(sys.argv[3])
num_it = int(sys.argv[4])
m2 = int(sys.argv[5]) 
print(model, m, tol, num_it, m2)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def encode(x, code):
    output = np.array([int(np.where(code == x[i])[0]) for i in range(0, len(x))])
    return output

def decode(x, code):
    output = np.zeros(len(x))
    for i in range(0, len(x)):
        output[i] = code[x[i]]
    return output

## Generate New Notes ##

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
    return(output)

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


## Main Function to generate new pieces ##
## Input and Output files should be CSV (MIDI files converged to CSV via MIDI-CSV, http://www.fourmilab.ch/webtools/midicsv/)
## Line-skip is line number for first occuring note, number of header lines to skip 
## m = number of hidden states 
## tol = tolerance for convergence of EM Algorithm
## num_it = number of new pieces to generate (5 generated pieces saved as CSV)
## m2 = number of second level of hidden states for Two Hidden State HMM

def hmm_compose(input_filename, output_filename, line_skip, model, m,  tol, num_it = 1000, m2 = None):
    #with open(input_filename,encoding = "ISO-8859-1") as fd:
    with open(input_filename,'rU') as fd:
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
    metrics = np.zeros(shape = (n,num_it + 2))
    metrics[:,0] = time
    metrics[:,1] = notes
    
    it_save = np.random.choice(np.arange(num_it), size = 2)
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
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
        
    if model == 'first_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
       
        
    if model == 'first_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
        
        
    if model == 'second_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        it2, T2mat = second_orderLR(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
        
    
    if model == 'second_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, T2mat = second_order(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, T2mat, None, possibleNotes,'second_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1

    if model == 'third_order-LR':
        it1, p1, pi1, phi1, Tmat1 = first_orderLR(n, m, k, xNotes, tol)
        it2, T2mat = second_orderLR(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        it3, T3mat = third_orderLR(n, m, k, xNotes, pi1, Tmat1, T2mat, phi1, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
        
    
    if model == 'third_order':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, T2mat = second_order(n, m, k, xNotes, pi1, Tmat1, phi1, tol)
        it3, T3mat = third_order(n, m, k, xNotes, pi1, Tmat1, T2mat, phi1, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, T2mat, T3mat, possibleNotes,'third_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1

    if model == 'two_hidden_states':
        it1, p1, pi1, phi1, Tmat1, A1, B1 = two_hidden_states(n, m, m2, k, xNotes, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm_2hidden(n, pi1, phi1, Tmat1, A1, B1, possibleNotes)
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1

             
    if model == 'HSMM':
        it1, p1, pi1, phi1, Tmat1 =  HSMM(n, m, k, xNotes, tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmm(n, pi1, phi1, Tmat1, None, None, possibleNotes,'first_order')
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
        
    
    if model == 'ARHMM':
        it1, p1, pi1, phi1, Tmat1 = first_order(n, m, k, xNotes, tol)
        it2, p2, psi = first_orderARHMM(n, m, k, np.log(pi1), np.log(Tmat1), np.log(phi1), xNotes,  tol)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = hmmARHMM(n, pi1, phi1, Tmat1, psi, possibleNotes)
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1

    
    if model == 'MCMC':
        tempNotes = MCMC( n,  m,  k, xNotes, tol, num_it)
        for i in range(num_it):
            print(i)
            metrics[:,i+2] = decode(tempNotes[:,i], possibleNotes)
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
        
    
    if model == 'TVAR':
        # Find parameters that maximize likelihood
        x = notes - np.mean(notes)
        T = n
        pvals=np.array([7, 15]) 
        p=pvals[1]  
        dn=np.arange(0.94, 0.975,.005) 
        bn=np.arange(0.85, 0.915, 0.005) 
        m0=np.zeros(shape = (p,1)); n0=1; s0=0.01; C0=np.identity(p); 
        [popt,delopt,likp] = tvar_lik(x,pvals,dn,bn,m0,C0,s0,n0);
        print(popt)
        
        # Fit TVAR
        p=popt; m0=np.zeros(shape = (p,1)); n0=1; s0=0.01; C0=np.identity(p);  # initial priors 
        delta=delopt
        [m,C,n,s,e,mf,Cf,sf,nf,ef,qf] = tvar(x,p,delta,m0,C0,s0,n0);
        
        # Simulate from TVAR
        N=num_it; # MC sample size
        times=range(T);
        phis = tvar_sim(m,C,n,s,times,N);
        
        # Generate new notes
        
        err_term = np.random.normal(0, np.sqrt(s)) 
        for i in range(num_it):
            metrics[:, i+2] = x
            for t in range(p, T):
                if t == p:
                    metrics[t, i+2] = np.dot(x[t-1::-1], phis[:,t,17]) + err_term[t]
                else:
                    metrics[t, i+2] = np.dot(x[t-1:t-p-1:-1], phis[:,t,17]) + err_term[t]

            metrics[:,i+2] = np.round(metrics[:, i+2] + np.mean(notes))
            for j in range(len(metrics[:,i+2])):
                if metrics[j,i+2] not in possibleNotes:
                    metrics[j,i+2] = find_nearest(possibleNotes, metrics[j,i+2])
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
        m = popt
    
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
        
        for i in range(num_it):
            for j in range(0, n):
                noteArray[0,j] = np.random.choice(xstates, size = 1, p = phi15[zStar15[j], :])
                noteArray[1,j] = np.random.choice(xstates, size = 1, p = phi10[zStar10[j], :])
                noteArray[2,j] = np.random.choice(xstates, size = 1, p = phi5[zStar5[j], :])
            temp_notes = np.rint(np.mean(noteArray, axis=0)).astype(int)
            temp_notes = decode(temp_notes, possibleNotes)   
            print(i)
            metrics[:,i+2] = temp_notes
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
    
    
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
        for i in range(num_it):
            for j in range(0,n):
                output[2, j] = np.random.choice(zstates, size = 1, p = phi3[zStar3[j], :])
                output[1, j] = np.random.choice(zstates, size = 1, p = phi2[output[2, j], :])
                output[0, j] = np.random.choice(xstates, size = 1, p = phi1[output[1, j], :])
            temp_notes = decode(output[0,:], possibleNotes).astype(int)
            print(i)
            metrics[:,i+2] = temp_notes
            if i in it_save:
                newNotes[:,save_count] = metrics[:,i+2]
                save_count += 1
    
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
        output_f1 = split[0] + str(song_num)  + '__'+ model + '_' + str(m)+ '_m2-'+str(m2)+  '-tol' +str(tol)+'.' + split[1]
        newsong.to_csv(output_f1, header = None, index = False)



    # Save metrics values
    output_f4 = 'metrics/'+ name  + '__'+ model + '_' + str(m)+ '_m2-'+str(m2)+  '-tol' +str(tol)+'.' + split[1]
    pd.DataFrame(metrics).to_csv(output_f4, header = None, index = False)

## Example input for Pachelbel's Canon 
hmm_compose('OriginalCSV/pachelbel.csv', 'NewCSV/pachelbel.csv', 27, model, m, tol, num_it, m2)

