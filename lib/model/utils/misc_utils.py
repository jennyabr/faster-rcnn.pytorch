import os
import re

def get_epoch_num_from_ckpt(ckpt_path):
    filename = os.path.basename(ckpt_path)
    fname_without_ext = filename.split('.')[0]
    epoch_num = re.findall('\d+', fname_without_ext)[-1]
    return epoch_num
