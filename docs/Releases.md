## 0.0.8 (released 8/3/19)
- Minor bug fixes (see Issue #70)

## 0.0.7 (released 8/3/19)
- Implementation of FLUSS algorithm for determining the Corrected Arc Curve
- Implementation of algorithm to extract regimes from the Corrected Arc Curve
- Fix for handling missing NaN/inf values
- Ability to read in non-numpy array data types (lists, tuples)
- Bug fixes for SCRIMP++ implementation


## 0.0.6 (released 6/13/19)
- Fixed bug by requiring a later version of numpy

## 0.0.5 (released 6/9/19)
- SCRIMP++ implementation
- Algorithm performance benchmarking and use case descriptions
- Improved self-join handling


## 0.0.4 (released 2/11/19)
- Python 2 compatibility
- Added parallel STAMP function
- Updated docstrings for better interpretability
- Refactored self-join logic

## 0.0.3 (released 1/2/19)
- Implementation of STOMP for faster matrix profile calculation
- Fixed bugs in testing sute (note that this does not impact performance for matrixprofile-ts 0.0.2)
- Functionality for applying annotation vector ("apply_av")
- Functionality for locating top K discords ("discords")
