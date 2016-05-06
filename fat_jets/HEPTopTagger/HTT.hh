// HEPTopTagger2.0 as FastJet Plugin
// under construction -- far from being released

#ifndef __HTT_HH__
#define __HTT_HH__

#include <fastjet/tools/Transformer.hh>
#include <fastjet/LimitedWarning.hh>
#include <fastjet/WrappedStructure.hh>
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"
#include "fastjet/contrib/Nsubjettiness.hh"

using namespace std;
using namespace fastjet;

FASTJET_BEGIN_NAMESPACE
class HTTStructure;

/// HEPTopTagger class as FastJet Transformer. For a detailed description of the parameters have a look at the appendix of the [HEPTopTagger2 release paper](http://inspirehep.net/record/1353673?ln=en").
class HTT : public Transformer{
public:
  /// return type of the tagger
  enum ReturnType {
    TAG, ///< if there is a tag return the reconstructed top otherwise an empty PseudoJet is returned
      BEST_CANDIDATE ///< returns the best candidate that fulfills at least the mass window cut. Additional requirements depend on the selected modus. To check if the returned candidate is a tag use the is_tagged() method of HTTStructure Class. If there is no candidate an empty PseudoJet is returned.
      };
  
  /// HEPTopTagger modus
  enum Mode {
    EARLY_MASSRATIO_SORT_MASS, ///< apply the mass plane cuts before the mass window cut and minimize the deviation from the top mass to select the best candidate 
      LATE_MASSRATIO_SORT_MASS, ///< among the triples that pass the mass window cut minimize the deviation from the top mass to select the best candidate and check mass plane cuts later.
      EARLY_MASSRATIO_SORT_MODDJADE, ///< apply the mass plane cuts before the mass window cut and maximize the summed modified Jade distance to select the best candidate
      LATE_MASSRATIO_SORT_MODDJADE, ///< among the triples that pass the mass window cut maximize the summed modified Jade distance to select the best candidate and check mass plane cuts later.
      TWO_STEP_FILTER /// use the hardest hard substructures as best candidate
      };

  /// default constructor
  HTT();
  /// default destructor
  ~HTT();

  /// print describtion
  virtual string description() const;
  /// result of an HTT instance applied to a PseudoJet.
  virtual PseudoJet result(const PseudoJet & jet) const;
  /// print settings
  void get_setting() const;

  typedef HTTStructure StructureType;

  //settings
  /// set the return type of the tagger 
  void set_return_type(enum ReturnType returntype) {_returntype = returntype;}

  /// (un)select optimalR mode
  void do_optimalR(bool optimalR) {_do_optimalR = optimalR;}

  /// set mass drop threshold used for the extraction of hard substructures
  void set_mass_drop_threshold(double x) {_mass_drop_threshold = x;}
  /// set the maximal mass of hard substructures
  void set_max_subjet_mass(double x) {_max_subjet_mass = x;}

  /// set the number of hardest subjets used for the filtering of the triplet
  void set_filtering_n(unsigned nfilt) {_nfilt = nfilt;}
  /// set the jet size used for the filtering of the triplet
  void set_filtering_R(double Rfilt) {_Rfilt = Rfilt;}
  /// set the minimal transverse momentum of the used subjets
  void set_filtering_minpt_subjet(double x) {_minpt_subjet = x;}
  /// set the jet algorithm for the filtering of the triplet
  void set_filtering_jetalgorithm(JetAlgorithm jet_algorithm) {_jet_algorithm_filter = jet_algorithm;}

  /// set the jet algorithm used for reclustering to three subjets after filtering
  void set_reclustering_jetalgorithm(JetAlgorithm jet_algorithm) {_jet_algorithm_recluster = jet_algorithm;}

  /// set the mode of the tagger
  void set_mode(enum Mode mode) {_mode = mode;}
  /// set the used top mass in the tagger
  void set_mt(double x) {_mtmass = x;}
  /// set the used W mass in the tagger
  void set_mw(double x) {_mwmass = x;}
  /// set the top mass window
  void set_top_mass_range(double xmin, double xmax) {_mtmin = xmin; _mtmax = xmax;}
  /// set the width of the A-shaped bands for the mass plane cuts
  void set_fw(double fw) {_rmin = (1.-fw)*_mwmass/_mtmass; _rmax=(1.+fw)*_mwmass/_mtmass;}
  /// set the top to W mass ratio range for the mass plane cuts 
  void set_mass_ratio_range(double rmin, double rmax) {_rmin = rmin; _rmax = rmax;}
  /// set the ends of the mass plane cuts
  void set_mass_ratio_cut(double m23cut, double m13cutmin,double m13cutmax) {_m23cut = m23cut; _m13cutmin = m13cutmin; _m13cutmax = m13cutmax;}
  /// set the minimal transverse momentum of the reconstructed top (consistency cut) 
  void set_top_minpt(double x) {_minpt_tag = x;}

  /// set maximum jet size used in the optimalR mode  
  void set_optimalR_max(double x) {_max_fatjet_R = x;}
  /// set minimum jet size used in the optimalR mode
  void set_optimalR_min(double x) {_min_fatjet_R = x;}
  /// set step size used in the optimalR mode
  void set_optimalR_step(double x) {_step_R = x;}
  /// set threshold for the optimalR criterion
  void set_optimalR_threshold(double x) {_optimalR_threshold = x;}

  /// set the jet size used for the filtering used to determine R_opt_calc
  void set_filtering_optimalR_calc_R(double x) { _R_filt_optimalR_calc = x;}
  /// set the number of hardest subjets used for the filtering used to determine R_opt_calc
  void set_filtering_optimalR_calc_n(unsigned x) {_N_filt_optimalR_calc = x;}
  /// set the function used to determine R_opt_calc
  void set_optimalR_calc_fun(double (*f)(double)) {_mod_r_min_exp_function = true; _r_min_exp_function = f;}

  /// set top mass window for optimalR type
  void set_optimalR_type_top_mass_range(double x, double y) {_optimalR_mmin = x; _optimalR_mmax = y;}
  /// set width of A-shaped bands for optimalR type
  void set_optimalR_type_fw(double x) {_optimalR_fw = x;}
  /// set maximal difference between R_opt and R_opt_calc for optimalR type
  void set_optimalR_type_max_diff(double x) {_R_opt_diff = x;}

  /// set the jet size used for the filtering for fat jet that pass the optimalR cuts 
  void set_filtering_optimalR_pass_R(double x) {_R_filt_optimalR_pass = x;}
  /// set the number of hardest subjets used for the filtering for fats jet that pass the optimalR cuts 
  void set_filtering_optimalR_pass_n(unsigned x) {_N_filt_optimalR_pass = x;}
  /// set the jet size used for the filtering for fat jet that fail the optimalR cuts 
  void set_filtering_optimalR_fail_R(double x) {_R_filt_optimalR_fail = x;}
  /// set the number of hardest subjets used for the filtering for fat jets that fail the optimalR cuts 
  void set_filtering_optimalR_fail_n(unsigned x) {_N_filt_optimalR_fail = x;}

  /// set z_cut for pruning
  void set_pruning_zcut(double zcut) {_zcut = zcut;}
  /// set r_cut factor for pruning
  void set_pruning_rcut_factor(double rcut_factor) {_rcut_factor = rcut_factor;}

  ///set external jet modifier, e.g. for QJets
  void set_jet_modifier(PseudoJet (*f)(PseudoJet)) { _modify_jet=true; _modified_jet = f;}

private:
  ReturnType _returntype;
  bool _do_optimalR;
  Mode _mode;
  double _mass_drop_threshold, _max_subjet_mass;
  double _mtmass, _mwmass;
  double  _mtmin, _mtmax, _rmin, _rmax, _m23cut, _m13cutmin, _m13cutmax, _minpt_tag;
  double _zcut, _rcut_factor;
  unsigned _nfilt;
  double _Rfilt;
  double _minpt_subjet;
  JetAlgorithm _jet_algorithm_filter, _jet_algorithm_recluster;
  double _max_fatjet_R, _min_fatjet_R, _step_R, _optimalR_threshold;
  double  _R_filt_optimalR_calc, _N_filt_optimalR_calc;
  double (*_r_min_exp_function)(double);
  double _optimalR_mmin, _optimalR_mmax, _optimalR_fw, _R_opt_calc, _R_opt_diff;
  double _R_filt_optimalR_pass, _N_filt_optimalR_pass, _R_filt_optimalR_fail, _N_filt_optimalR_fail;
  bool _modify_jet,  _mod_r_min_exp_function;
  PseudoJet (*_modified_jet)(PseudoJet);

  static bool _first_time;
  static LimitedWarning _warnings_nonca;

  //internal functions
  PseudoJet run(const PseudoJet & jet, ReturnType _returntype = TAG) const;
  void print_banner();
  vector<PseudoJet> FindHardSubst(const PseudoJet& this_jet, vector<PseudoJet>& t_parts_o) const;
  bool check_mass_criteria(const vector<PseudoJet>& top_subs) const;
  vector<PseudoJet> subjets(const vector<PseudoJet>& top_subs) const;
  double djademod (const PseudoJet& subjet_i, const PseudoJet& subjet_j, const PseudoJet& ref) const;
  double perp(const PseudoJet & vec, const PseudoJet & ref) const;
  vector<PseudoJet> UnclusterFatjets(const vector<PseudoJet> & big_fatjets, const ClusterSequence & cs, const double small_radius) const;
  int optimalR_type(PseudoJet jet, int R_opt, double frec, double R_opt_calc) const;
  double f_rec(vector<PseudoJet> top_subs) const;
  double R_opt_calc_default(double pt_filt) const;
};

/// Structure Class holding all information from the tagging
class HTTStructure : public WrappedStructure{
public:
  /// default creator
  HTTStructure(const PseudoJet & result_jet) :
    WrappedStructure(result_jet.structure_shared_ptr()){}

  /// print description
  string description() const{ 
    return "Structure for HEPTopTagger 2.0 results.";
  }
  
  //get information
  /// top mass used in the tagger
  const double mtmass() const{return _mtmass;}
  /// W mass used in the tagger
  const double mwmass() const {return _mwmass;}
  /// fat jet that was put in the tagger
  const PseudoJet & original() const {return _initial_jet;}

  /// valid topcandidate with at least 3 hard substructures?
  const bool is_maybe_top() const {return _is_maybe_top;}
  /// mass plane cuts passed?
  const bool is_masscut_passed() const {return _is_masscut_passed;}
  /// reconstructed top exceeds minimal transverse momentum cut (consistency cut)?  
  const bool is_minptcut_passed() const {return _is_ptmincut_passed;}
  /// is it tagged? 
  const bool is_tagged() const {return (_is_masscut_passed && _is_ptmincut_passed);}

  /// deviation from the top mass
  const double delta_top() const {return _delta_top;}
  /// summed modified Jade distance
  const double djsum() const {return _djsum;}
  /// pruned fat jet mass
  const double pruned_mass() const {return _pruned_mass;}
  /// unfiltered fat jet mass
  const double unfiltered_mass() const {return _unfiltered_mass;}

  /// reconstructed top
  const PseudoJet & t() const {return _top_candidate;}
  /// reconstructed b
  const PseudoJet & b() const {return _top_subjets[0];}
  /// reconstructed W
  const PseudoJet W() const {PseudoJet _W = join( _top_subjets[1], _top_subjets[2]); return _W;}
  /// leading reconstructed W decay jet
  const PseudoJet & W1() const {return _top_subjets[1];}
   /// subleading reconstructed W decay jet
  const PseudoJet & W2() const {return _top_subjets[2];}
  //const vector<PseudoJet> & top_subjets() const {return _top_subjets;}
  //const vector<PseudoJet> & top_subs() const {return _top_subs;}
  /// leading subjet
  const PseudoJet & j1() const {return _top_subs[0];}
  /// subleading subjet
  const PseudoJet & j2() const {return _top_subs[1];}
  /// subsubleading subjet
  const PseudoJet & j3() const {return _top_subs[2];}
  /// constituents of the initial fat jet used to build the top
  const vector<PseudoJet> & top_hadrons() const {return _top_hadrons;}
  /// hard parts found inside the fat jet
  const vector<PseudoJet> & hardparts() const {return _top_parts;}
  /// fat jet put in the tagger
  const PseudoJet & fat_initial() const {return _initial_jet;}
  /// fat jet reduced to R_opt
  const PseudoJet & fat_optR() const {return _optimalR_jet;}
  /// processed fat jet (R_opt, external jet modifiers, ...)  
  const PseudoJet & fat() const {return _optimalR_jet;}
  /// obtained f_rec
  const double & frec() const {return _fw;}

  /// N-subjettiness values calculate from the unfiltered processed fat jet (at R_opt) 
  const double nsub_unfiltered(int order, fastjet::contrib::Njettiness::AxesMode axes = fastjet::contrib::Njettiness::kt_axes, double beta = 1., double R0 = 1.) const {
    contrib::Nsubjettiness nsub(order, axes, beta, R0);
    return nsub.result(_optimalR_jet);
  }
  /// N-subjettiness values calculate from the optimalR type specifically filtered processed fat jet (at R_opt) 
  const double nsub_filtered(int order, fastjet::contrib::Njettiness::AxesMode axes = fastjet::contrib::Njettiness::kt_axes, double beta = 1., double R0 = 1.) const {
    contrib::Nsubjettiness nsub(order, axes, beta, R0);
    return nsub.result(_filt_fat);
  }

  /// print information
  void get_info() const {  
    cout << "#--------------------------------------------------------------------------\n";
    cout << "#                          HEPTopTagger Result" << endl;
    cout << "#" << endl;
    cout << "# mass plane cuts passed: " << _is_masscut_passed << endl;
    cout << "# top candidate mass: " << _top_candidate.m() << endl;
    cout << "# top candidate (pt, eta, phi): (" 
	 << _top_candidate.perp() << ", "
	 << _top_candidate.eta() << ", "
	 << _top_candidate.phi_std() << ")" << endl;
    cout << "# top hadrons: " << _top_hadrons.size() << endl;
    cout << "# hard substructures: " << _top_parts.size() << endl;
    cout << "# |m - mtop| : " << _delta_top << endl; 
    cout << "# djsum : " << _djsum << endl;
    cout << "# fw : " << _fw << endl;
    if (_do_optimalR) {
      cout << "# optimalR mode activated: " << endl;
      cout << "#   Ropt, Ropt_calc: " << _Ropt/10. << ", " << _Ropt_calc << endl;
      cout << "#   optimalR type : " << _optimalRtype << endl;
    }
    cout << "#--------------------------------------------------------------------------\n";
    return;
  }

  /// print setings used in the tagger 
  void get_setting() const {
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
    if (_do_optimalR) {
      cout << "# optimalR mode activated: " << endl;
      cout << "#   top mass window: [" << _optimalR_mmin << ", " << _optimalR_mmax << "]" << endl;
      cout << "#   fW: " << _optimalR_fw << endl;
      cout << "#   R_opt_diff : " <<  _R_opt_diff << endl;
      cout << "#   pass  R_filt , N_filt: " <<  _R_filt_optimalR_pass << ", " << _N_filt_optimalR_pass << endl;
      cout << "#   fail  R_filt , N_filt: " <<  _R_filt_optimalR_fail << ", " << _N_filt_optimalR_fail << endl;
    }
    cout << "#--------------------------------------------------------------------------\n";
    return;
  }

  /// constituents of a reconstruced used jet (b, W1, W2, W, t, j1, j2, j3). Due to the reclustering inside the tagger, the default PseudoJet member consituents() does not work. 
  const vector<PseudoJet> subjet_constituents(PseudoJet subjet) const {
    if (subjet == _top_candidate)
      return _top_candidate.constituents();
    vector<PseudoJet> used_pieces(subjet.constituents()), constituents;
    for (vector<PseudoJet>::iterator used_piece = used_pieces.begin(); used_piece != used_pieces.end(); ++used_piece) {
      for (unsigned i = 0; i < _top_candidate.pieces().size(); i++) {
      	PseudoJet piece =  _top_candidate.pieces()[i];
    	if (used_piece->px() == piece.px() && used_piece->py() == piece.py() &&
	    used_piece->pz() == piece.pz() && used_piece->e() == piece.e()) {
	  vector<PseudoJet> piece_constituents  = piece.constituents();
	  for (vector<PseudoJet>::iterator constituent = piece_constituents.begin(); constituent != piece_constituents.end(); ++constituent) {
	    constituents.push_back(*constituent) ;
	  }
	}
      }
    }
    if (constituents.size() == 0) {
      cout << "This works only for the HTT objects b, W1, W2, W, t, j1, j2, j3" << endl;
    }
    return constituents;
  }
    

protected:
  bool _do_optimalR;
  PseudoJet _initial_jet;
  PseudoJet _optimalR_jet;

  HTT::Mode _mode;
  bool _modify_jet;
  bool  _mod_r_min_exp_function;
  double _mass_drop_threshold;
  double _max_subjet_mass;
  double _mtmass, _mwmass;
  double _mtmin, _mtmax;
  double _rmin, _rmax;
  double _m23cut, _m13cutmin, _m13cutmax;
  double _minpt_tag;
  unsigned _nfilt;
  double _Rfilt;
  JetAlgorithm _jet_algorithm_filter;
  double _minpt_subjet;
  JetAlgorithm _jet_algorithm_recluster;
  double _zcut;
  double _rcut_factor;
  
  bool _is_masscut_passed;
  bool _is_ptmincut_passed;
  bool _is_maybe_top;

  double _delta_top;
  double _djsum;

  double _pruned_mass;
  double _unfiltered_mass;

  double _fw;

  PseudoJet _top_candidate;
  vector<PseudoJet> _top_subs;
  vector<PseudoJet> _top_subjets;
  vector<PseudoJet> _top_hadrons;
  vector<PseudoJet> _top_parts;

  double _optimalR_mmin, _optimalR_mmax, _optimalR_fw, _R_opt_diff;
  double _R_filt_optimalR_pass, _N_filt_optimalR_pass, _R_filt_optimalR_fail, _N_filt_optimalR_fail;

  
  double _Ropt, _Ropt_calc;
  int _optimalRtype;
  map<int,PseudoJet> _HEPTopTagger;
  PseudoJet _filt_fat;

private:
  const double nsub(PseudoJet jet, int order, contrib::Njettiness::AxesMode axes, double beta, double R0) const {
    contrib::Nsubjettiness nsub(order, axes, beta, R0);
    return nsub.result(jet);
  }

  friend class HTT;
};

FASTJET_END_NAMESPACE

#endif // __HTT_HH__
