Musical pieces are first converted from MIDI to CSV using: http://www.fourmilab.ch/webtools/midicsv/.  The generated CSVs are in the folder `OriginalCSV`.  

The notebook `JASA_Results.ipynb` contains all relevant code.  

## Instructions

The inference algorithms for this code are in Cython.  To run the notebook `JASA_Results.ipynb`, first run: `np.get_include()` in the notebook, which will output a `sample-path`.  Then, in the terminal set **export CFLAGS="-I sample-path $CFLAGS"**.  Finally, in the terminal, run 
- `python setup.py build_ext --inplace`
- `python setup-left_right.py build_ext --inplace`

to compile the Cython modules.  The notebook can then run from top to bottom.  

Required software and versions include:

| Software        | Version           | 
| :------------- |-------------:| 
| Python     | 3.6.5 | 
| IPython     | 6.4.0    |   
| numpy	|1.14.3|
|scipy|	1.1.0|
|seaborn	|0.8.1|
|pandas|	0.23.0|
|csv	|1.0|
|matplotlib|	2.2.2|
|editdistance|	0.4|
|sklearn	|0.19.1|
|statsmodels|	0.9.0|

The only inputs needed to analyze different pieces are the path and filename of the new piece and the length of the shortest note in the new piece (in MIDI clicks).  This information can be found in the header of the generated CSV; see http://www.fourmilab.ch/webtools/midicsv/.

Generated CSVs are in `NewCSV` and can be converted back to MIDI using the same software as above.  Other software is needed to convert these generated pieces to MP3 or other audio format.  Metrics calculations on generated pieces are saved in the `metrics` folder.

### References
- MIDICSV code: http://www.fourmilab.ch/webtools/midicsv/
- Original Training Pieces:
  - https://www.mfiles.co.uk/classical-midi.htm
  - http://www.midiworld.com/classic.htm/beethoven.htm
  - http://www.piano-midi.de/midi_files.htm



