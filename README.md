# 575_summarization

Please create conda environment from env.yml file:

`conda env create -f env.yml --prefix ./env`

__________________________

D1: Form your team, Organize Repo, Outline Report

Group Members:

Rachel Hantz,
Yi-Chien (Nica) Lin,
Yian Wang,
Tashi Tsering,
Chenxi Li

___________________________

D2: Process a DocSet

Script `proc_docset.py` is under `src/`

To run the script:

`cd src/`
 
and run either:

`./proc_docset.sh`

or 

`condor_submit ../D2.cmd`

Preprocessed and tokenized documents are stored under `outputs`

___________________________

D3: Initial System - Content Selection

Scripts `tf_idf.py`, `ilp.py`, and `run_rouge_eval.py` are under `src/`

To run the initial system:

`cd src/`

and run either:

`./D3_run.sh`

or 

`condor_submit ../D3.cmd`

__________________________
