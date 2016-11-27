import re
import time
import itertools
import numpy as np
import pandas as pd
from IPython.display import display

def flatten(list_of_lists):
  """Flatten a list-of-lists into a single list."""
  return list(itertools.chain.from_iterable(list_of_lists))

def pretty_print_matrix(M, rows=None, cols=None, dtype=float):
  """Pretty-print a matrix using Pandas.
  
  Args:
    M : 2D numpy array
    rows : list of row labels
    cols : list of column labels
    dtype : data type (float or int)
  """
  display(pd.DataFrame(M, index=rows, columns=cols, dtype=dtype))

def pretty_timedelta(fmt="%d:%02d:%02d", since=None, until=None):
  """Pretty-print a timedelta, using the given format string."""
  since = since or time.time()
  until = until or time.time()
  delta_s = until - since
  hours, remainder = divmod(delta_s, 3600)
  minutes, seconds = divmod(remainder, 60)
  return fmt % (hours, minutes, seconds)


##
# Word processing functions
def canonicalize_digits(word):
  if any([c.isalpha() for c in word]): return word
  word = re.sub("\d", "DG", word)
  if word.startswith("DG"):
    word = word.replace(",", "") # remove thousands separator
  return word

def canonicalize_word(word, wordset=None, digits=True):
  word = word.lower()
  if digits:
    if (wordset != None) and (word in wordset): return word
    word = canonicalize_digits(word) # try to canonicalize numbers
  if (wordset == None) or (word in wordset): return word
  else: return "<unk>" # unknown token

def canonicalize_words(words, **kw):
  return [canonicalize_word(word, **kw) for word in words]


##
# Utility functions for processing CoNLL 2003 data
##
def load_dataset(fname):
  docs = []
  with open(fname) as fd:
    cur = []
    for line in fd:
      # new sentence on -DOCSTART- or blank line
      if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
        if len(cur) > 0:
          docs.append(cur)
        cur = []
      else: # read in tokens
        cur.append(line.strip().split("\t",1))
    # flush running buffer
    docs.append(cur)
  return docs

def pad_sequence(seq, left=1, right=1):
  return left*[("<s>", "O")] + seq + right*[("</s>", "O")]

def preprocess_docs(docs, tag_to_num, vocab, k):
  # Add sentence boundaries with empty tags
  docs = flatten([pad_sequence(seq, left=k, right=k) 
                        for seq in docs])

  words, tags = zip(*docs)
  ids = vocab.words_to_ids(
    canonicalize_words(words, wordset=vocab.word_to_id))
  y = [tag_to_num[t.split("|")[0]] for t in tags]
  return ids, y

def build_windows(ids, y, vocab, k, shuffle=True, drop_boundary=True):
  C = 2*k + 1
  windows = np.zeros((len(ids)-2*k, C), dtype=int)
  y = np.array(y)[k:-k]
  for i in range(C):
    windows[:,i] = ids[i:i+len(windows)]
  if shuffle:
    # Randomize order of training data
    idx = np.random.permutation(len(windows))
    windows, y = windows[idx], y[idx]
  if drop_boundary:
    # Drop rows where the center word is <s> or </s>
    mask = (windows[:,k] != vocab.START_ID) * (windows[:,k] != vocab.END_ID)
    windows = windows[mask]
    y = y[mask]
  return windows, y


def docs_to_windows(docs, tag_to_num, vocab, k):
  w,y = preprocess_docs(docs, tag_to_num, vocab, k)
  return build_windows(w, y, vocab, k, shuffle=False, drop_boundary=True)


