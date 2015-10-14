#Functions with input utilities
#verify format, convert

#Check format
#1 columns - could be binary
#2 columns - raise exception (not a valid format)
#3 columns - should be binary

def transform_target_variable(y):
  '''
  1. Determines y format (binary, single-column or unknown)
  2. If format is not correct, raises exception
  3. If format is correct determines if it's binary or single-column
  4. If binary, raises exception
  '''
  pass

def is_in_single_column_format(y):
  dimensions = len(y.shape)
  return dimensions == 1

def is_in_binary_format(X):
  '''
      This function verifies if a given numpy array is in binary format
      e.g. a = np.array([1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [1,0,0])
      a is a binary formatted input where there are three different classes,
      and we have two observations for the first class and one observation for
      the second and third classes.
  '''
  #Check there's only 1's and 0's
  pass


def convert_to_single_column_format(y):
    pass
