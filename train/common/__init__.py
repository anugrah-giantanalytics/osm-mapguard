"""Shared machinery for every stage under train/.

Stage scripts put `train/` and `src/` at the front of sys.path in their first two lines,
then import from here. Nothing in this package reads data except through
`data_contract`, so when the dataset is finalised only the contract has to change.
"""
