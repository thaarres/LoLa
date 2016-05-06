// QJets for the HEPTopTagger2.0 FastJet Plugin

#include "QHTT.hh"

PseudoJet qjet_default(PseudoJet injet) {
  double _q_zcut(0.1), _q_dcut_fctr(0.5), _q_exp_min(0.), _q_exp_max(0.), _q_rigidity(0.1), _q_truncation_fctr(0.0);
  QjetsPlugin _qjet_plugin(_q_zcut, _q_dcut_fctr, _q_exp_min, _q_exp_max, _q_rigidity, _q_truncation_fctr);
  JetDefinition _qjet_def = fastjet::JetDefinition(&_qjet_plugin);
  vector<fastjet::PseudoJet> _q_constits = injet.associated_cluster_sequence()->constituents(injet);
  ClusterSequence* _qjet_seq = new ClusterSequence(_q_constits, _qjet_def);      
  PseudoJet _qjet = sorted_by_pt(_qjet_seq->inclusive_jets())[0];
  _qjet_seq->delete_self_when_unused();
  return _qjet;
}


QHTT::QHTT() : _niter(100), _qjet_fun(&qjet_default) {
};

void QHTT::run(HTT htt, PseudoJet jet) {
  _weight_q1 = -1.;
  _weight_q2 = -1.;
  _m_sum = 0.;
  _m2_sum = 0.;
  _eps_q = 0.;
  _qtags = 0;
  htt.set_jet_modifier(_qjet_fun);
  for (int iq = 0; iq < _niter; iq++) {
    PseudoJet _htt_q = htt(jet);
    if (_htt_q.has_structure_of<HTT>()) {
      _qtags++;
      _m_sum += _htt_q.structure_of<HTT>().t().m();
      _m2_sum += _htt_q.structure_of<HTT>().t().m() * _htt_q.structure_of<HTT>().t().m();
      const fastjet::ClusterSequence* _qjet_seq3 = _htt_q.structure_of<HTT>().fat().associated_cluster_sequence();
      const QjetsBaseExtras* ext =
	dynamic_cast<const QjetsBaseExtras*>(_qjet_seq3->extras());
      double _qweight = ext->weight();
      if (_qweight > _weight_q1) {
      	_weight_q2 = _weight_q1; _htt_q2 = _htt_q1;
      	_weight_q1 =_qweight; _htt_q1 = _htt_q;
      } else if (_qweight > _weight_q2) {
      	_weight_q2 = _qweight; _htt_q2 = _htt_q;
      }
    }
  }
  _eps_q = float(_qtags)/float(_niter);
};
