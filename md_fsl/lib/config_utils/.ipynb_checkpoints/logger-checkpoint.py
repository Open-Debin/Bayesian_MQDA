import time
import os, sys
from pathlib import Path
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

class Logger(object):

  def __init__(self, log_dir, title):
    """Create a summary writer logging to log_dir."""
    self.log_dir = Path("{:}".format(str(log_dir)))
    if not self.log_dir.exists(): os.makedirs(str(self.log_dir))
    self.title = title
    self.log_file = '{:}/{:}_rename_date-{:}.txt'.format(self.log_dir,title, time_for_file())
    self.file_writer = open(self.log_file, 'a')

  def checkpoint(self, name):
    return self.log_dir / name

  def print(self, string, fprint=True, is_pp=False):
    if is_pp: pp.pprint (string)
    else:     print(string)
    if fprint:
      self.file_writer.write('{:}\n'.format(string))
      self.file_writer.flush()
        
  def write(self, string):
    self.file_writer.write('{:}\n'.format(string))
    self.file_writer.flush()  
        
  def rename(self, name):
        new_name = self.log_file.replace('rename',name)
        os.rename(self.log_file, new_name)
        
  def close(self):
    self.file_writer.close()


def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string