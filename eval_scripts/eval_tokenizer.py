import os, sys, numpy as np 
import torch
from pathlib import Path
from datetime import date


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from data.tokenizers.default_garmentcode import DefaultGarmentCode

eval_outputs = Path('./eval_outputs')
today = str(date.today())
os.mkdir(eval_outputs / today, exist_ok=True)

import code; code.interact(local=locals())