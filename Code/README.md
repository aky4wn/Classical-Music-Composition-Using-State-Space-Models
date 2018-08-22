Musical pieces are first converted from MIDI to CSV using: http://www.fourmilab.ch/webtools/midicsv/.  The generated CSVs are in the folder `OriginalCSV`.  

The notebook `JASA_Results.ipynb` contains all relevant code.  It can be run from top to bottom as is to reproduce the results presented in the paper.  The files `BaumWelch.pyx` and `BaumWelchLR.pyx` contain functions for inference, written in Cython.

The only inputs needed to analyze different pieces are the path and filename of the new piece and the length of the shortest note in the new piece (in MIDI clicks).  This information can be found in the header of the generated CSV; see http://www.fourmilab.ch/webtools/midicsv/.

Generated CSVs are in `NewCSV` and can be converted back to MIDI using the same software as above.  Other software is needed to convert these generated pieces to MP3 or other audio format.  Metrics calculations on generated pieces are saved in the `metrics` folder.


