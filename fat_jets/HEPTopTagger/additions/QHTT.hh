// QJets for the HEPTopTagger2.0 FastJet Plugin

#ifndef __QHTT__HH__
#define __QHTT__HH__

#include "HTT.hh"
#include "qjets/QjetsPlugin.h"

///QJet mode for the HEPTopTagger
class QHTT {
public:
  /// default creator
  QHTT();
  /// set number of QJet iterations
  void set_iterations(int niter) {_niter = niter;};
  /// run the QJet mode
  void run(HTT htt, PseudoJet jet);
  /// HEPTopTagger result with highest weight  
  const PseudoJet leading() const {return _htt_q1;};
  ///HEPTopTagger result with second highest weight
  const PseudoJet subleading() const {return _htt_q2;};
  /// weight of the HEPTopTagger result with highest weight 
  const double weight_leading() const {return _weight_q1;};
  /// weight of the HEPTopTagger result with second highest weight 
  const double weight_subleading() const {return _weight_q2;};
  /// QJet efficiency
  const double eps_q() const {return _eps_q;};
  /// unweighted mean mass
  const double m_mean() const {return _qtags > 0 ? _m_sum/float(_qtags) : 0. ;}
  /// unweighted mean squared mass
  const double m2_mean() const {return _qtags > 0 ? _m2_sum/float(_qtags) : 0.;}
  /// set jet modifier 
  PseudoJet (*_qjet_fun)(PseudoJet);
  
private:
  int _niter;
  int _qtags;
  double _weight_q1, _weight_q2;
  PseudoJet _htt_q, _htt_q1, _htt_q2;
  double _m_sum, _m2_sum;
  double _eps_q;
};
#endif // __QHTT_HH__
