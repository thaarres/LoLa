// Example for a low_pt working point
// Written for the HEPTopTagger2.0 FastJet Plugin

#ifndef __LOWPT_HH__
#define __LOWPT_HH__

#include <math.h>

#include "fastjet/PseudoJet.hh"

#include "HTT.hh"
#include "FWM.hh"

/// Example low transverse momentum working point
class LowPt {
public:
  LowPt();

  /// Is the HTT result consistent with the working point? 
  const bool is_tagged(PseudoJet htt_result) const;
};
#endif // __LOWPT_HH__
