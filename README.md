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

Download Code at Release D2

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

Download Code at Release D3

D3: Initial System - Content Selection

Scripts `tf_idf.py`, `ilp.py`, and `run_rouge_eval.py` are under `src/`

To run the initial system:

`cd src/`

and run either:

`./D3_run.sh`

or 

`condor_submit ../D3.cmd`

Summaries will print under `outputs/D3`

Results will print under `results/D3`

__________________________

Download Code at Release D4

D4: Improved System - Information Ordering

Scripts `cluster.py`, `build_theme_graph.py`, `baseline.py`, and `io_cos_jac.py` are under `src/`

To run the improved system with Information Ordering methods included:

`cd src`

and run either:

`./D4_run.sh`

or 

`condor_submit ../D4.cmd`

Summaries will print under `outputs/D4` along with an indermediate directory `ilp_cos`

Results will print under  `results/D4`

___________________________

Download Code at Release D5

D5: Final System - Content Realization

Scripts `solve_PPN.py` and `find_people.py` are under src/

To run the final system on the development dataset with Content Realization method included:

`cd src`

and run:

`./D5_run_dev.sh`

Summaries will print under `outputs/D5_devtest`

Results will print under `results`

To run the final system on the evaluation dataset with Content Realization method included:

`cd src`

and run either:

`./D5_run.sh`

or

`condor_submit ../D5.cmd`

Summaries will print under `outputs/D5_evaltest`

Results will print under `results`
