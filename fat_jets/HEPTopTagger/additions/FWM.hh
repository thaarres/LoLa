// Fox-Wolfram-Moments for the HEPTopTagger2.0 FastJet Plugin

#ifndef __FWM_HH__
#define __FWM_HH__

#include <math.h>
#include <cmath>

#include "gsl/gsl_math.h"
#include "gsl/gsl_sf_legendre.h"

#include "HTT.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/tools/Boost.hh"

using namespace std;
using namespace fastjet;  

/// Class to calculate Fox-Wolfram moments of unit or pT weight
class FWM {
public:
  /// empty creator
  FWM();
  /// creator for an arbitrary vector of PseudoJets
  FWM(vector<PseudoJet> jets);
  /// creator for selection of jets from a HTT result 
  FWM(PseudoJet htt_result, int selection);
  /// creator for selection of jets from a HTTStructure result 
  FWM(HTTStructure htt, int selection); 

  /// Unit weight Fox-Wolfram moment of requested order with 
  const double U(unsigned order) const;
  /// pT weight Fox-Wolfram moment of requested order with 
  const double Pt(unsigned order) const;
  /// pT weight Fox-Wolfram moment of requested order with pT relative to a reference jet
  const double Pt(unsigned order, PseudoJet ref_pj) const;
 
private:
  double cos_Omega(PseudoJet jet1, PseudoJet jet2) const;
  inline double ATan2(double x, double y) const;
  double Theta(PseudoJet j) const;
  double legendre(int l, double x) const {return gsl_sf_legendre_Pl(l,x);};
  double perp(PseudoJet v_pj, PseudoJet ref_pj) const;

  vector<PseudoJet> _jets;
};

#endif // __FWM_HH__
