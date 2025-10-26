# Pipe Resonance Estimation

# Laundary List:
* [x] sort out images size and stuff
  - [x] work with the crop from the selected backdrop
* [x] adaptive_binarization_threshold
* [x] refactor _process_frame, it's a mess!
* [x] define pixel as a class with getter for row, col and x,y
* [x] move types into one place, like units or user_defined_types?
* [x] draw crossed circle at scene config points
* [x] draw border around images
* [x] draw deviation distribution
* [x] any assumption of a vertical line is invalid, overhaul!
* [x] we still have issue with detecting points! "points from skeleton" needs to be
  - robust,
  - more deterministic, and
  - have better parametrization.
* [ ] disable strobe for now
* [ ] write usage instruction
* [ ] how to distinguish points at different height of the pipe?
* [ ] define what is a "deviation", from which the "resonance" is to be estimated
* [ ] include formatter in the repo's CI


## not sure if actually needed (atleast for now now)
* [ ] find checker board
* - [ ] get pixel size
* - [ ] block out checker board
* [ ] detect if stobe is on/off  
      draw a red circle around strobe light when it is detected to be on