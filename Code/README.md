hmm_compose.py is the main script used to train the various HMMs and generate new pieces.  

The code for model implementation is in BaumWelch.pyx, BaumWelchLR.pyx (both in Cython and need to be compiled before use) and TVAR.py.

Usage: hmm_compose(input_filename, output_filename, line_skip, model, m,  tol, num_it = 1000, m2 = None)

Input and Output files should be CSV (MIDI files converged to CSV via MIDI-CSV, http://www.fourmilab.ch/webtools/midicsv/)

Line_skip is line number for first occuring note, number of header lines to skip 

m = number of hidden states 

tol = tolerance for convergence of EM Algorithm

num_it = number of new pieces to generate (5 generated pieces saved as CSV)

m2 = number of second level of hidden states for Two Hidden State HMM


Example: hmm_compose('OriginalCSV/pachelbel.csv', 'NewCSV/pachelbel.csv', 27, model, m, tol, num_it, m2)
