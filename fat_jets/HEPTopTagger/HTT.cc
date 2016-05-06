// HEPTopTagger2.0 as FastJet Plugin

#include "./HTT.hh"

FASTJET_BEGIN_NAMESPACE

//default function for R_opt_calc
double HTT::R_opt_calc_default(double pt_filt) const{
  return 327./pt_filt;
}

bool HTT::_first_time = true;
LimitedWarning HTT::_warnings_nonca;

void HTT::print_banner() {
  if (!_first_time) {return;}
  _first_time = false;

  cout << "#--------------------------------------------------------------------------\n";
  cout << "#                  HEPTopTagger 2.0 -- FastJet Contrib                     \n";
  cout << "#                                                                          \n";
  cout << "# Please cite JHEP 1506 (2015) 203 [arXiv:1503.05921 [hep-ph]]             \n";
  cout << "# and JHEP 1010 (2010) 078 [arXiv:1006.2833 [hep-ph]].                     \n";
  cout << "# This code is provided without warranty.                                  \n";
  cout << "#--------------------------------------------------------------------------\n";
}

HTT::HTT(): _do_optimalR(true),
	    _modify_jet(false),
	    _returntype(TAG),
	    _mass_drop_threshold(0.8), _max_subjet_mass(30.), _mode(EARLY_MASSRATIO_SORT_MASS),
	    _mtmass(172.3), _mwmass(80.4),
	    _mtmin(150.), _mtmax(200.), _rmin(0.85*80.4/172.3), _rmax(1.15*80.4/172.3),
	    _m23cut(0.35), _m13cutmin(0.2), _m13cutmax(1.3), _minpt_tag(200.),
	    _nfilt(5), _Rfilt(0.3),  _minpt_subjet(0.),
	    _zcut(0.1), _rcut_factor(0.5),
	    _jet_algorithm_filter(cambridge_algorithm),
	    _jet_algorithm_recluster(cambridge_algorithm),
	    _min_fatjet_R(0.5), _step_R(0.1), _optimalR_threshold(0.2),
	    _R_filt_optimalR_calc(0.2), _N_filt_optimalR_calc(10),
	    _mod_r_min_exp_function(false),
	    _optimalR_mmin(150.), _optimalR_mmax(200.), _optimalR_fw(0.175),
	    _R_opt_diff(0.3),
	    _R_filt_optimalR_pass(0.2), _N_filt_optimalR_pass(5),
	    _R_filt_optimalR_fail(0.3), _N_filt_optimalR_fail(3)
{
  print_banner();
};

HTT::~HTT(){};

string HTT::description() const{ 
  return "HEPTopTagger 2.0. FastJet Contrib.";
}

PseudoJet HTT::result(const PseudoJet & jet) const {
  //warning if not C/A
  if ((! jet.has_associated_cluster_sequence()) ||
      (jet.validated_cs()->jet_def().jet_algorithm() != cambridge_algorithm))
    _warnings_nonca.warn("HEPTopTagger should be applied on jets from a Cambridge/Aachen clustering.");

  //prepare input jet
  PseudoJet _jet(jet);
  if (_modify_jet)
    _jet = _modified_jet(jet);

  //fixed R run
  if (!_do_optimalR) {
    PseudoJet direct = run(_jet, _returntype);
    if (direct.has_structure_of<HTT>()) {
      HTTStructure *ds = (HTTStructure*) direct.structure_non_const_ptr();
      ds -> _do_optimalR = _do_optimalR;
    }
    return direct;
  }

  //optimalR run
  vector<PseudoJet> big_fatjets, small_fatjets;
  big_fatjets.push_back(_jet);
  const ClusterSequence* _seq(jet.validated_cluster_sequence());
  int _Ropt(-1);
  int maxR = int(jet.validated_cluster_sequence()->jet_def().R() * 10);
  int minR = int(_min_fatjet_R * 10);
  int stepR = int(_step_R * 10);
  map<int,PseudoJet> _HEPTopTagger;
  PseudoJet _HEPTopTagger_opt;

  //optimal R loop
  for (int R = maxR; R >= minR; R -= stepR) {
    small_fatjets = UnclusterFatjets(big_fatjets, *_seq, R / 10.);
    double dummy_pt(-1.);
    PseudoJet  result_R_best;

    //find hardest subjet at R
    for (vector<PseudoJet>::iterator small_fatjet = small_fatjets.begin(); small_fatjet !=  small_fatjets.end(); ++small_fatjet) {
      PseudoJet  result_R_tmp = run(*small_fatjet, BEST_CANDIDATE);
      if (result_R_tmp.perp() > dummy_pt) {
  	dummy_pt = result_R_tmp.perp();
	result_R_best = result_R_tmp;
      }
    } //End of loop over small_fatjets
    _HEPTopTagger[R] = result_R_best;
    big_fatjets = small_fatjets;
    small_fatjets.clear();
    
    //Check optimalR criterion
    if (_Ropt < 0 && R < maxR) {                 
      if (_HEPTopTagger[R].m() < (1-_optimalR_threshold)*_HEPTopTagger[maxR].m()){
  	_Ropt = R + stepR;
      }
    }
  }//End of loop over R

  //if we did not find Ropt in the loop, pick the last value
  if (_Ropt < 0 && _HEPTopTagger[maxR].m() > 0)
    _Ropt = minR;

  //for the case that there is no tag at all (< 3 hard substructures)
  if (_Ropt < 0 && _HEPTopTagger[maxR].m() == 0)
    _Ropt = maxR;

  _HEPTopTagger_opt = _HEPTopTagger[_Ropt];

  //claculate R_opt
  Filter filter_optimalR_calc(_R_filt_optimalR_calc, SelectorNHardest(_N_filt_optimalR_calc));
  double R_opt_calc(-1);
  if (_mod_r_min_exp_function) {
    R_opt_calc =  _r_min_exp_function(filter_optimalR_calc(_jet).pt());
  } else {
    R_opt_calc = R_opt_calc_default(filter_optimalR_calc(_jet).pt());
  }

  //determine optimalR type and filter accordingly
  int optR_type(-1);
  PseudoJet _filt_fat;
  if(_HEPTopTagger_opt.has_structure_of<HTT>()) {
    double frec = _HEPTopTagger_opt.structure_of<HTT>().frec();
    optR_type = optimalR_type(_HEPTopTagger_opt, _Ropt, frec, R_opt_calc);
    Filter filter_optimalR_pass(_R_filt_optimalR_pass, SelectorNHardest(_N_filt_optimalR_pass));
    Filter filter_optimalR_fail(_R_filt_optimalR_fail, SelectorNHardest(_N_filt_optimalR_fail));
    PseudoJet _fat(_HEPTopTagger_opt.structure_of<HTT>().fat());
    if(optR_type == 1) {
      _filt_fat = filter_optimalR_pass(_fat);
    } else {
      _filt_fat = filter_optimalR_fail(_fat);
    }
  }

  //combine information to result 
  PseudoJet result =  _HEPTopTagger_opt; 
  if(result.has_structure_of<HTT>()) {
    if (_returntype == BEST_CANDIDATE || (_returntype == TAG && result.structure_of<HTT>().is_tagged())) {
      HTTStructure *s = (HTTStructure*) result.structure_non_const_ptr();
      s -> _do_optimalR = _do_optimalR;
      s -> _optimalR_mmin =  _optimalR_mmin;
      s -> _optimalR_mmax = _optimalR_mmax;
      s -> _optimalR_fw = _optimalR_fw;
      s -> _R_opt_diff = _R_opt_diff;
      s -> _initial_jet = _jet;
      s -> _Ropt = _Ropt ;
      s -> _Ropt_calc = R_opt_calc ;
      s -> _optimalRtype = optR_type;
      s -> _HEPTopTagger = _HEPTopTagger ;
      s -> _filt_fat = _filt_fat;
      s -> _R_filt_optimalR_pass = _R_filt_optimalR_pass;
      s -> _N_filt_optimalR_pass = _N_filt_optimalR_pass;
      s -> _R_filt_optimalR_fail = _R_filt_optimalR_fail;
      s -> _N_filt_optimalR_fail = _N_filt_optimalR_fail;
      return result;
    }
  }  
  return PseudoJet();
}

//core module HTT at fixed R
PseudoJet HTT::run(const PseudoJet & jet, ReturnType _returntype) const {
  PseudoJet fat(jet);
  PseudoJet _top_candidate;
  vector<PseudoJet> _top_subs, _top_subjets;
  double _delta_top(-1.), _djsum(-1.);
  double _Rprun = fat.validated_cluster_sequence()->jet_def().R();
  bool _is_maybe_top(false), _is_masscut_passed(false), _is_ptmincut_passed(false);
  double _pruned_mass(-1.), _unfiltered_mass(-1.), _fw(-1.); 

  //Find hard substructure
  vector<PseudoJet> _top_parts;
 _top_parts = HTT::FindHardSubst(fat, _top_parts);
  if (_top_parts.size() < 3)
    return PseudoJet();
  
  // loop over triples
  for (vector<PseudoJet>::iterator rr = _top_parts.begin(); rr != _top_parts.end(); ++rr) {
    for (vector<PseudoJet>::iterator ll = rr+1; ll != _top_parts.end(); ++ll) {
      for (vector<PseudoJet>::iterator kk = ll+1; kk != _top_parts.end(); ++kk) {

	if((_mode==TWO_STEP_FILTER) && rr!=_top_parts.begin())
	  continue;
	if((_mode==TWO_STEP_FILTER) && ll!=_top_parts.begin()+1)
	  continue;
	if((_mode==TWO_STEP_FILTER) && kk!=_top_parts.begin()+2)
	  continue;

	PseudoJet triple = join(*rr, *ll, *kk);

	//Filter triple -> topcandidate
	double R_min_parts = min(  min(rr->delta_R(*ll),
				       rr->delta_R(*kk)),
				   kk->delta_R(*ll)
				   );
	double filt_top_R = min( _Rfilt, 0.5*R_min_parts);
	JetDefinition filtering_def(_jet_algorithm_filter, filt_top_R);
	Filter filter(filtering_def, SelectorNHardest(_nfilt) * SelectorPtMin(_minpt_subjet));
	PseudoJet topcandidate = filter(triple);
	
	//mass window cut
  	if (topcandidate.m() < _mtmin || _mtmax < topcandidate.m())
	  continue;

	// Recluster to 3 subjets and apply mass plane cuts
	if (topcandidate.pieces().size() < 3)
	  continue;
	JetDefinition reclustering(_jet_algorithm_recluster, 3.14);
	ClusterSequence*  cs_top_sub = new ClusterSequence(topcandidate.pieces(), reclustering);
        vector <PseudoJet> top_subs = sorted_by_pt(cs_top_sub->exclusive_jets(3));         
	cs_top_sub->delete_self_when_unused();

	// Require the third subjet to be above the pT threshold
	if (top_subs[2].perp() < _minpt_subjet)
	  continue;

	// Modes with early 2d-massplane cuts
	if (_mode == EARLY_MASSRATIO_SORT_MASS      && !check_mass_criteria(top_subs)) {continue;}
	if (_mode == EARLY_MASSRATIO_SORT_MODDJADE  && !check_mass_criteria(top_subs)) {continue;}
	
	//is this candidate better than the other?
	double deltatop = fabs(topcandidate.m() - _mtmass);
	double djsum = djademod(top_subs[0], top_subs[1], topcandidate) 
	  + djademod(top_subs[0], top_subs[2], topcandidate)
	  + djademod(top_subs[1], top_subs[2], topcandidate);
	bool better(false);
	if ((_mode == EARLY_MASSRATIO_SORT_MASS) || (_mode == LATE_MASSRATIO_SORT_MASS)) {
	  if (_delta_top < 0. || deltatop < _delta_top) 
	    better = true;
	} else if ((_mode == EARLY_MASSRATIO_SORT_MODDJADE) || (_mode == LATE_MASSRATIO_SORT_MODDJADE)) {
	  if (djsum > _djsum) 
	    better = true;
	} else if (_mode == TWO_STEP_FILTER) {
	  better = true;
	} else {
	  cout << "ERROR: UNKNOWN MODE (IN DISTANCE MEASURE SELECTION)" << endl;
	  return PseudoJet();
	}

	//if this candidate is better, update
	if (better) {
	  _is_maybe_top = true;
	  _is_masscut_passed = false;
	  _is_ptmincut_passed = false;
	  
	  _delta_top = deltatop; 
	  _djsum = djsum;
	  _top_candidate = topcandidate;
	  _top_subs = HTT::subjets(top_subs);
	  _top_subjets = HTT::subjets(top_subs);

	  // Pruned and unfiltered triple mass
	  JetDefinition jet_def_prune(cambridge_algorithm, _Rprun);
	  Pruner pruner(jet_def_prune, _zcut, _rcut_factor);
	  PseudoJet prunedjet = pruner(triple);
	  _pruned_mass = prunedjet.m();
	  _unfiltered_mass = triple.m();
	  
	  //are all criteria fulfilled?
	  if (check_mass_criteria(top_subs))
	    _is_masscut_passed = true;
	  if (_top_candidate.pt() > _minpt_tag)
	    _is_ptmincut_passed = true;
	} //end better
      } //end kk
    } //end ll
  } //end rr

  //combine information
  bool _is_tag = _is_masscut_passed && _is_ptmincut_passed;
  if ((_returntype == TAG && _is_tag) || (_returntype == BEST_CANDIDATE && _is_maybe_top)){
    HTTStructure * s = new HTTStructure(_top_candidate);
    s->_mode = _mode;
    s->_modify_jet = _modify_jet;
    s->_mtmass = _mtmass;
    s->_mwmass  = _mwmass;
    s->_mass_drop_threshold=_mass_drop_threshold;
    s->_max_subjet_mass=_max_subjet_mass;
    s->_mtmin=_mtmin; s->_mtmax=_mtmax;
    s->_rmin=_rmin; s->_rmax=_rmax;
    s->_m23cut=_m23cut; s->_m13cutmin=_m13cutmin; s->_m13cutmax=_m13cutmax;
    s->_minpt_tag = _minpt_tag ;
    s-> _nfilt = _nfilt ;
    s->_Rfilt = _Rfilt ;
    s->_jet_algorithm_filter = _jet_algorithm_filter ;
    s->_minpt_subjet = _minpt_subjet ;
    s->_jet_algorithm_recluster = _jet_algorithm_recluster ;
    s->_zcut = _zcut ;
    s->_rcut_factor = _rcut_factor;

    s->_is_masscut_passed = _is_masscut_passed ;
    s->_is_ptmincut_passed = _is_ptmincut_passed ;
    s->_is_maybe_top = _is_maybe_top ;
  
    s->_optimalR_jet = fat;
    s->_delta_top = _delta_top ;
    s->_djsum = _djsum ;
    s->_pruned_mass = _pruned_mass ;
    s->_unfiltered_mass = _unfiltered_mass ;
    s->_fw =  f_rec(_top_subs) ;

    s->_top_candidate = _top_candidate ;
    s->_top_subs = _top_subs ;
    s->_top_subjets = _top_subjets ;
    s->_top_hadrons =  _top_candidate.constituents() ;
    s->_top_parts = _top_parts ;

    _top_candidate.set_structure_shared_ptr(SharedPtr<PseudoJetStructureBase>(s));
    return _top_candidate;
  } else {
    return PseudoJet();
  }
};
//---------------------------------------------------------------------

//find hard substructures
vector<PseudoJet> HTT::FindHardSubst(const PseudoJet & this_jet, vector<PseudoJet>& t_parts_o) const {
  vector<PseudoJet> t_parts(t_parts_o);
  PseudoJet parent1(0, 0, 0, 0), parent2(0, 0, 0, 0);
  if (this_jet.m() < _max_subjet_mass || !this_jet.validated_cs()->has_parents(this_jet, parent1, parent2)) {
    t_parts.push_back(this_jet);
  } else {
    if (parent1.m() < parent2.m()) 
      std::swap(parent1, parent2);
    t_parts = FindHardSubst(parent1, t_parts);
    if (parent1.m() < _mass_drop_threshold * this_jet.m()) {
      t_parts = FindHardSubst(parent2, t_parts);
    }
  }
  t_parts = sorted_by_pt(t_parts);
  return t_parts;
};

bool HTT::check_mass_criteria(const vector<PseudoJet> & top_subs) const {
  bool is_passed = false;
  double m12 = (top_subs[0] + top_subs[1]).m();
  double m13 = (top_subs[0] + top_subs[2]).m();
  double m23 = (top_subs[1] + top_subs[2]).m();
  double m123 = (top_subs[0] + top_subs[1] + top_subs[2]).m();
  if (
      (atan(m13/m12) > _m13cutmin && _m13cutmax > atan(m13/m12)
       && (m23/m123 > _rmin && _rmax > m23/m123))
      ||
      (((m23/m123) * (m23/m123) < 1 - _rmin * _rmin* (1 + (m13/m12) * (m13/m12)))
       &&
       ((m23/m123) * (m23/m123) > 1 - _rmax * _rmax * (1 + (m13/m12) * (m13/m12)))
       && 
       (m23/m123 > _m23cut))
      ||
      (((m23/m123) * (m23/m123) < 1 - _rmin * _rmin * (1 + (m12/m13) * (m12/m13)))
       &&
       ((m23/m123) * (m23/m123) > 1 - _rmax * _rmax * (1 + (m12/m13) * (m12/m13)))
       && 
       (m23/m123 > _m23cut))
      ) { 
    is_passed = true;
  }
  return is_passed;
}

//order subjets to b, W1, W2
vector<PseudoJet> HTT::subjets(const vector<PseudoJet>& top_subs) const {
  vector<PseudoJet> _top_subjets;
  double m12 = (top_subs[0] + top_subs[1]).m();
  double m13 = (top_subs[0] + top_subs[2]).m();
  double m23 = (top_subs[1] + top_subs[2]).m();
  double dm12 = fabs(m12 - _mwmass);
  double dm13 = fabs(m13 - _mwmass);
  double dm23 = fabs(m23 - _mwmass);
  
  if (dm23 <= dm12 && dm23 <= dm13) {
    _top_subjets.push_back(top_subs[0]); 
    _top_subjets.push_back(top_subs[1]); 
    _top_subjets.push_back(top_subs[2]);	
  } else if (dm13 <= dm12 && dm13 < dm23) {
    _top_subjets.push_back(top_subs[1]);
    _top_subjets.push_back(top_subs[0]);
    _top_subjets.push_back(top_subs[2]);
  } else if (dm12 < dm23 && dm12 < dm13) {
    _top_subjets.push_back(top_subs[2]);
    _top_subjets.push_back(top_subs[0]);
    _top_subjets.push_back(top_subs[1]);
  }
  return  _top_subjets;
}

//modified Jade distance
double HTT::djademod (const PseudoJet& subjet_i, const PseudoJet& subjet_j, const PseudoJet& ref) const {
  double dj = -1.0;
  double delta_phi = subjet_i.delta_phi_to(subjet_j);
  double delta_eta = subjet_i.eta() - subjet_j.eta();
  double delta_R = sqrt(delta_eta * delta_eta + delta_phi * delta_phi);	
  dj = perp(subjet_i, ref) * perp(subjet_j, ref) * pow(delta_R, 4.);
  return dj;
}

//pt wrt a reference vector
double HTT::perp(const PseudoJet & vec, const PseudoJet & ref) const {
  double ref_ref = ref.px() * ref.px() + ref.py() * ref.py() + ref.pz() * ref.pz();
  double vec_ref = vec.px() * ref.px() + vec.py() * ref.py() + vec.pz() * ref.pz();
  double per_per = vec.px() * vec.px() + vec.py() * vec.py() + vec.pz() * vec.pz();
  if (ref_ref > 0.) 
    per_per -= vec_ref * vec_ref / ref_ref;
  if (per_per < 0.) 
    per_per = 0.;
  return sqrt(per_per);
}


vector<PseudoJet> HTT::UnclusterFatjets(const vector<PseudoJet> & big_fatjets, 
					const ClusterSequence & cseq, 
					const double small_radius) const {
  vector<PseudoJet> small_fatjets;
  for (vector<PseudoJet>::const_iterator big_fatjet = big_fatjets.begin(); big_fatjet !=  big_fatjets.end(); ++big_fatjet) {
    PseudoJet this_jet = *big_fatjet;
    PseudoJet parent1(0, 0, 0, 0), parent2(0, 0, 0, 0);
    bool test = cseq.has_parents(this_jet, parent1, parent2);
    double dR(-1.);

    if(test) dR = sqrt(parent1.squared_distance(parent2));

    if (!test || (dR>0 && dR<small_radius)) {
      small_fatjets.push_back(this_jet);
    } else {
      vector<PseudoJet> parents;
      parents.push_back(parent1);
      parents.push_back(parent2);
      UnclusterFatjets(parents, cseq, small_radius);
    }
  }
  return small_fatjets;
}

//optimal_R type
int HTT::optimalR_type(PseudoJet jet, int R_opt, double frec, double R_opt_calc) const {
  if(jet.m() < _optimalR_mmin || jet.m() > _optimalR_mmax)
    return 0;
  if(frec > _optimalR_fw)
    return 0;
  if(R_opt/10. - R_opt_calc > _R_opt_diff)
    return 0;
  return 1;
}

//minimal |(m_ij / m_123) / (m_w/ m_t) - 1|
double HTT::f_rec(vector<PseudoJet> top_subs) const {
  double m12 = (top_subs[0] + top_subs[1]).m();
  double m13 = (top_subs[0] + top_subs[2]).m();
  double m23 = (top_subs[1] + top_subs[2]).m();
  double m123 = (top_subs[0] + top_subs[1] + top_subs[2]).m();

  double fw12 = fabs( (m12/m123) / (_mwmass/_mtmass) - 1.);
  double fw13 = fabs( (m13/m123) / (_mwmass/_mtmass) - 1.);
  double fw23 = fabs( (m23/m123) / (_mwmass/_mtmass) - 1.);
  
  return min(fw12, min(fw13, fw23));  
}

void HTT::get_setting() const {
  cout << "#--------------------------------------------------------------------------\n";
  cout << "#                         HEPTopTagger Settings" << endl;
  cout << "#" << endl;
  cout << "# input jet modified by external function: " << _modify_jet << endl;
  cout << "# mode: " << _mode << " (0 = EARLY_MASSRATIO_SORT_MASS) " << endl;
  cout << "#        "         << " (1 = LATE_MASSRATIO_SORT_MASS)  " << endl;
  cout << "#        "         << " (2 = EARLY_MASSRATIO_SORT_MODDJADE)  " << endl;
  cout << "#        "         << " (3 = LATE_MASSRATIO_SORT_MODDJADE)  " << endl;
  cout << "#        "         << " (4 = TWO_STEP_FILTER)  " << endl;
  cout << "# top mass: " << _mtmass << "    ";
  cout << "W mass: " << _mwmass << endl;
  cout << "# top mass window: [" << _mtmin << ", " << _mtmax << "]" << endl;
  cout << "# W mass ratio: [" << _rmin << ", " << _rmax << "] (["
       <<_rmin*_mtmass/_mwmass<< "%, "<< _rmax*_mtmass/_mwmass << "%])"<< endl;
  cout << "# mass plane cuts: (m23cut, m13min, m13max) = (" 
       << _m23cut << ", " << _m13cutmin << ", " << _m13cutmax << ")" << endl;
  cout << "# mass_drop_threshold: " << _mass_drop_threshold << "    ";
  cout << "max_subjet_mass: " << _max_subjet_mass << endl;
  cout << "# R_filt: " << _Rfilt << "    ";
  cout << "n_filt: " << _nfilt << endl;
  cout << "# minimal subjet pt: " << _minpt_subjet << endl;
  cout << "# minimal reconstructed pt: " << _minpt_tag << endl;
  cout << "# internal jet algorithms (0 = kt, 1 = C/A, 2 = anti-kt): " << endl; 
  cout << "#   filtering: "<< _jet_algorithm_filter << endl;
  cout << "#   reclustering: "<< _jet_algorithm_recluster << endl;
  cout << "#--------------------------------------------------------------------------\n";
  return;
}

FASTJET_END_NAMESPACE
