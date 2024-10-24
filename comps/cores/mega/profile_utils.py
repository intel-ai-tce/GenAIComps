#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys

os.environ['DNNL_VERBOSE'] = '1'
os.environ['DNNL_VERBOSE_TIMESTAMP'] = '1'
import psutil

import json
import time
import threading
from contextlib import contextmanager

try:
  from queue import Queue
except ImportError:
  from Queue import Queue


def to_microseconds(s):
  return 1000000 * float(s)

class PlatformUtils:

    def __init_(self):
        self.cpufreq = ''
        self.cpu_socket_count = ''
        self.svmem = ''
        return

    def dump_platform_info(self):
        # let's print CPU information
        print("=" * 20, "CPU Info", "=" * 20)
        # number of cores
        print("Physical cores:", psutil.cpu_count(logical=False))
        print("Total cores:", psutil.cpu_count(logical=True))
        # CPU frequencies
        cpufreq = psutil.cpu_freq()
        print("Max Frequency:", cpufreq.max)
        print("Min Frequency:", cpufreq.min)
        #cpu_socket_count = int(subprocess.check_output(
        #    'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l'))
        #print("Socket Number:", cpu_socket_count)
        print("=" * 20, "Memory Information", "=" * 20)
        # get the memory details
        svmem = psutil.virtual_memory()
        print("Total: ", int(svmem.total / (1024 ** 3)), "GB")
        self.cpufreq = cpufreq
        #self.cpu_socket_count = cpu_socket_count
        self.svmem = svmem


class FileUtils:

    def __init_(self):
        return
    def replace_string_in_file(self, filename, oldstring, newstring):
        fin = open(filename, "rt")
        #output file to write the result to
        fout = open('tmp.txt', "wt")
        #for each line in the input file
        for line in fin:
            #read replace the string and write to output file
            fout.write(line.replace(oldstring, newstring))
        #close input and output files
        fin.close()
        fout.close()
        os.remove(filename)
        os.rename('tmp.txt',filename)


class oneDNNLog:

    def __init_(self):
        self.filename = ''
        self.data = None
        self.exec_data = None
        self.with_timestamp = True
        return

    def load_log(self, log):
        self.filename = log

        fn_t_list = [self.load_log_dnnl_timestamp_backend, self.load_log_dnnl_timestamp]
        fn_not_list = [self.load_log_dnnl_backend, self.load_log_dnnl, self.load_log_mkldnn]

        fn_list = fn_not_list
        self.with_timestamp = False

        data = fn_t_list[0](log)
        for d in data['timestamp']:
            if self.is_float(d) is True:
                self.with_timestamp = False
                fn_list = fn_t_list


        for index, fn in enumerate(fn_list):
            data = fn(log)
            count = data['time'].count()
            if count > 2:
                print(index)
                break

        exec_data = data[data['exec'] == 'exec']
        self.data = data
        self.exec_data = exec_data.copy()

        if self.with_timestamp is True:
            import io
            with io.open('./oneDNN.json', mode='wb') as fh:
                tp = TraceProfiler(output=fh)
                tp.install()
                for index, row in self.data.iterrows():
                    if row["time"] != None and row["time"].find('.') != -1:
                        tp.fire_event(
                            event_type='exec',
                            event_name=row["type"],
                            event_cat='DNNL_Op',
                            kernel_name=row["jit"],
                            timestamp=str(float(row["timestamp"])*1000),
                            duration=str(float(row["time"])*1000),
                            pass_type=row["pass"],
                        )
                tp.shutdown()
        return

    def load_log_dnnl(self, log):
        import pandas as pd
        # dnnl_verbose,exec,cpu,convolution,jit:avx2,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,,alg:convolution_direct,mb1_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,1.21704
        data = pd.read_csv(log, names=[ 'dnnl_verbose','exec','arch','type', 'jit', 'pass', 'fmt', 'opt', 'alg', 'shape', 'time', 'dummy'], engine='python')
        return data


class TraceWriter(threading.Thread):

  def __init__(self, terminator, input_queue, output_stream):
    threading.Thread.__init__(self)
    self.daemon = True
    self.terminator = terminator
    self.input = input_queue
    self.output = output_stream

  def _open_collection(self):
    """Write the opening of a JSON array to the output."""
    self.output.write(b'[')

  def _close_collection(self):
    """Write the closing of a JSON array to the output."""
    self.output.write(b'{}]')  # empty {} so the final entry doesn't end with a comma

  def run(self):
    self._open_collection()
    while not self.terminator.is_set() or not self.input.empty():
      item = self.input.get()
      self.output.write((json.dumps(item) + ',\n').encode('ascii'))
    self._close_collection()


class TraceProfiler(object):
  """A python trace profiler that outputs Chrome Trace-Viewer format (about://tracing).

     Usage:

        from pytracing import TraceProfiler
        tp = TraceProfiler(output=open('/tmp/trace.out', 'wb'))
        with tp.traced():
          ...

  """
  TYPES = {'call': 'B', 'return': 'E', 'exec': 'X'}

  def __init__(self, output, clock=None):
    self.output = output
    self.clock = clock or time.time
    self.pid = os.getpid()
    self.queue = Queue()
    self.terminator = threading.Event()
    self.writer = TraceWriter(self.terminator, self.queue, self.output)

  @property
  def thread_id(self):
    return threading.current_thread().name

  @contextmanager
  def traced(self):
    """Context manager for install/shutdown in a with block."""
    self.install()
    try:
      yield
    finally:
      self.shutdown()

  def install(self):
    """Install the trace function and open the JSON output stream."""
    self.writer.start()               # Start the writer thread.

  def shutdown(self):
    self.terminator.set()              # Stop the writer thread.
    self.writer.join()                 # Join the writer thread.

  def fire_event(self, event_type, event_name, event_cat, timestamp, duration):
    """Write a trace event to the output stream."""
    # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
    event = dict(
      name=event_name,                 # Event Name.
      cat=event_cat,               # Event Category.
      tid=self.thread_id,             # Thread ID.
      ph=self.TYPES[event_type],      # Event Type.
      pid=self.pid,                   # Process ID.
      ts=timestamp,                   # Timestamp.
      dur=duration
      )
    self.queue.put(event)


