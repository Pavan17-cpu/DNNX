"""Fast correctness checks; does not run the long training sweeps."""
from problem1.core import test_adding_generator, verify_cells
from problem2.core import test_attention, three_token_example
print('Adding generator:', test_adding_generator())
print('Scratch cell max errors:', verify_cells())
print('Attention tests:', test_attention())
print('3-token output:', three_token_example()[-1])
