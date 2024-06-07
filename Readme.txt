Please locate the folder ADHS_RP in an arbitrary directory on your computer and follow the follwoing instruction.

The python environment should cover the following imports from libraries:

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import time
import sys
import pathlib
import stlcg


The MATLAB environment should cover:

 - Deep learning toolbox, 
 - Statistics and Machine Learning toolbox,

Please make it certain you python is added to the system path. For example in windows, python.exe should be added to system variable environments.
we use python 3.8 and pytorch 2.0.1 + cu 11.7

In this package we check repeatability of the experimental section of the paper.

The first part of codes are for case study 1 in the paper.
  
  - Please open the folder "ADHS_RP\Case_study1". You will see a folder called "ADHS_RP\Case_study1\1".
  - Please run the content of this file.
  - This file firstly generates the LB4TL for the specification proposed in Case study1.
  - Then it runs the python file for training the controller when uses LB4TL for backpropagation.
  - Once the python file is executed a .mat file called traininfo.mat will be generated that we use for verification.
  - We firstly try risk measure RM_verifier() for verification of controller. However it takes m = 4*10^6 different i.i.d residuals 
    for verification which takes weeks for datageneration. Thus we set m = 10,000 here and we show the largest residual
    is still negative which is the requirement for verification.
  - However m = 10,000 is not enough for risk measure but it is enough for conformal prediction to give us the guarantee that is provided
    in the paper. Thus we also provided a new function called CP_verifier() that provides us the mentioned guarantee in the paper with 
    m = 10,000. By the way, we didnt discuss conformal prediction in the paper. 
  - The last step of the script is to generate the figure proposed in case study1 that presents the simulation for trained controller.

  - **** The script calls a python function in MATLAB. for those reviewers who can not call .py files from MATLAB, we have provided 
    another folder called "ADHS_RP\Case_study1\Use_this_if_1_does_not_work" which segments the script in folder "ADHS_RP\Case_study1\1" 
    in several scripts.Whenever the reviewer needs to run the .py file he can run it manually in his desired IDE and once he generated 
    "traininfo.mat" he can continue the rest of the code that is written in the next script. **** 


The second part of the codes are for Case_study 2 in the paper.

  - Please open the folder "ADHS_RP\Case_study2".
  - Please run the content of this file.
  - This file firstly generates the LB4TL for the specification proposed in Case study2.
  - Then it runs an mfile called Test_STL() for training the controller when uses LB4TL for backpropagation.
  - After training, we firstly try risk measure RM_verifier() for verification of controller. However it takes m = 4*10^6 different i.i.d residuals 
    for verification which takes weeks for datageneration. Thus, we set m = 10,000 here and we show the largest residual
    is still negative which is the requirement for verification.
  - However m = 10,000 is not enough for risk measure but it is enough for conformal prediction to give us the guarantee that is provided
    in the paper. Thus we also provided a new function called CP_verifier() that provides us the mentioned guarantee in the paper with 
    m = 10,000. By the way, we didnt discuss conformal prediction in the paper. 
  - The last step of the script is to generate the figure proposed in case study2 that presents the simulation for trained controller.


The third part of the codes are for proving that training with LB4TL is at least 10 times faster than STLCG, Gilpin et al. and Pant et al. on a complex specification.

  *** The results of the regenerated table are not supposed to be equal to what is in the paper. But the average runtime for LB4TL
  should be abouth 10x faster than the others. The difference in proposed runtimes between Repeatability package and submitted document, is mostly related to the fact
  that we didn't include torch.manual_seed(0) in our training code, at the time of submission. But it is included in the Repeatability package.  ***

  - Please open the folder "ADHS_RP\Table1". You will see a folder called "ADHS_RP\Table1\1".
  - Please run the content of this file.
  - This file utilizes 5 different initial guess for control parameters.
  - For every one of those guesses, this file trains the controller 4 times, via the objective functions provided in 
    Pant et al. , Gilpin et al. ,  STLCG and LB4TL respectively. The training process tries to provide a positive robustness 
    for all the sampled initial states for us.
  - In case an objective function was unable to provide us positive robustness for all the initial states within 3600 seconds, 
    we terminate the process and return 3600 as the runtime. We need to show that LB4TL succesfully finishes the training for all of those 4 intial guesses for control parameters
    and our runtime for LB4TL is still 10x faster even in this unfair condition.  
  - ***  The script calls several python functions in MATLAB. for those reviewers who can not call .py files from MATLAB, we have provided 
    another folder called   "ADHS_RP\Table1\use_this_if_1_does_not_work"   which segments the script in folder "ADHS_RP\Table1\1" in several scripts. Whenever the reviewer
    needs to run the .py file he can run it manually in his desired IDE and once he generated "training_info.mat" he can continue the rest of
    the code that is written in the next script. ****

   
