bool debug = false;

#include <iostream>
#include <fstream>
#include <iomanip>
#include <utility>
#include <vector>

#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TString.h"

#include "TH2.h"
#include "THStack.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TClonesArray.h"
#include "TLorentzVector.h"


#include "TMath.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
using namespace ROOT::Math;


//HEPTopTagger
#include "./HEPTopTagger/HTT.hh"


//FastJet
#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>
#include <fastjet/Selector.hh>
#include <fastjet/Tools/Filter.hh>
#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/SoftDrop.hh"

#include "classes/DelphesClasses.h"

#include "ExRootAnalysis/ExRootTreeReader.h"
#include "ExRootAnalysis/ExRootTreeWriter.h"
#include "ExRootAnalysis/ExRootTreeBranch.h"
#include "ExRootAnalysis/ExRootUtilities.h"
#include "ExRootAnalysis/ExRootResult.h"


using namespace std;
using namespace fastjet;

void print(PseudoJet p) {
  cout.precision(3);
  cout << scientific << "(" << p.px() << ", " << p.py() << ", " << p.pz() << ", "<< p.e() << ")" << endl;
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  int counter(0);
  if (argc != 3) {
    cout << "ERROR: give input_file out_file as arguments" << endl;
    return 0;
  }

  TString inputFile(argv[1]);
  TString outfile(argv[2]);
  TFile fout(outfile, "recreate");

  // Create chain of root trees
  TChain chain("Delphes");
  chain.Add(inputFile);

  // Create object of class ExRootTreeReader
  ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
  Long64_t numberOfEntries = treeReader->GetEntries();
  TClonesArray *branchTower = treeReader->UseBranch("Tower");
  TClonesArray *branchPartons = treeReader->UseBranch("Partons");
  
  cout << "** Chain contains " <<  numberOfEntries << " events" << endl;
  Tower *tower;
  GenParticle *parton;
  vector<PseudoJet> towers, partons;
  towers.clear();
  partons.clear();

  TTree t("tree","tree"); 
  TLorentzVector *fatjet=new TLorentzVector(0.,0.,0.,0.); 
  TLorentzVector *filtered=new TLorentzVector(0.,0.,0.,0.); 
  TLorentzVector *top=new TLorentzVector(0.,0.,0.,0.); 
  TLorentzVector *softdropjet=new TLorentzVector(0.,0.,0.,0.); 
  t.Branch("fatjet","TLorentzVector",&fatjet);
  t.Branch("filtered","TLorentzVector",&filtered);
  t.Branch("top","TLorentzVector",&top);
  t.Branch("softdropjet","TLorentzVector",&softdropjet);

  std::vector<float> veta;
  std::vector<float> vphi;
  std::vector<float> ve;
  t.Branch("veta",&veta);
  t.Branch("vphi",&vphi);
  t.Branch("ve",&ve);

  double tau2(-1.), tau3(-1.);
  t.Branch("tau2",&tau2);
  t.Branch("tau3",&tau3);

  double tau2_filt(-1.), tau3_filt(-1.);
  t.Branch("tau2_filt",&tau2_filt);
  t.Branch("tau3_filt",&tau3_filt);
  
  double tau2_sd(-1.), tau3_sd(-1.);
  t.Branch("tau2_sd",&tau2_sd);
  t.Branch("tau3_sd",&tau3_sd);

  
  bool htt_tagged(false);
  t.Branch("htt_tagged",&htt_tagged);

  // parameters
  double R_fat(1.5), dRmatch(1.2);
  JetDefinition prejet_def(antikt_algorithm, R_fat);
  JetDefinition jet_def(cambridge_algorithm, R_fat);
  
  Selector select_akt = SelectorAbsEtaMax(1.0);
  Selector select_ca = SelectorPtRange(350.,450.) && SelectorAbsEtaMax(1.0); 

  // groomers
  Filter filter(0.3, SelectorNHardest(5));
  contrib::SoftDrop sd(1.,0.2);

  //Nsubjettiness
  contrib::Nsubjettiness nsub2(2, fastjet::contrib::Njettiness::kt_axes, 1., R_fat);
  contrib::Nsubjettiness nsub3(3, fastjet::contrib::Njettiness::kt_axes, 1., R_fat);

  //HTT
  HTT tagger;

   // Loop over all events
  for(int entry = 0; entry < numberOfEntries; ++entry) {

    if (debug) cout << " -------- " << endl;
    if (debug) cout << "event: " <<  entry << endl;

    // read event
    treeReader->ReadEntry(entry);

    // towers
    if (debug) cout << "towers  " << endl;
    towers.clear();
    for (int i = 0; i < branchTower->GetEntriesFast(); i++) {
      tower = (Tower*) branchTower->At(i);
      PseudoJet tmp1((tower->P4()).Px(), 
		    (tower->P4()).Py(), 
		    (tower->P4()).Pz(), 
		    (tower->P4()).E());

      //minimal tower energy of 1 GeV, is already applied by Delphes. Check this
       if (tmp1.E() > 1. ) { 
	towers.push_back(tmp1);
       }
    }
    towers = sorted_by_pt(towers);

    // partons
    if (debug) cout << "partons  " << endl;
    for (int i = 0; i < branchPartons->GetEntriesFast(); i++) {
      parton = (GenParticle*) branchPartons->At(i);
      PseudoJet tmp1((parton->P4()).Px(), 
		    (parton->P4()).Py(), 
		    (parton->P4()).Pz(), 
		    (parton->P4()).E());
      partons.push_back(tmp1);
    }
    partons = sorted_by_pt(partons);

    //before C-A fat jets, precluster towers into anti-KT 1.5 jets
    ClusterSequence precluster(towers, prejet_def);
    vector<PseudoJet> antikT_fat_jets = sorted_by_pt(select_akt(precluster.inclusive_jets())); 

    for (unsigned jak = 0; jak < antikT_fat_jets.size(); jak++) {
    
      //run CA15 on AK15 constituents. If there are several CA15 found, take the hardest one.
      ClusterSequence clust_seq(antikT_fat_jets[jak].constituents(), jet_def); 
      PseudoJet CA_fat_jet = sorted_by_pt(clust_seq.inclusive_jets())[0];

      if (!select_ca.pass(CA_fat_jet))
	continue;

      //match to parton
      double dRmin(100.);
      int pmin(-1);
      for (unsigned p = 0; p < partons.size(); p++) {
	double dR = CA_fat_jet.delta_R(partons[p]);
	if (dR < dRmin) {
	  dRmin = dR;
	  pmin = p;
	}
      }
    
      htt_tagged = false;
      veta.clear();
      vphi.clear();
      ve.clear();
      
      PseudoJet partontop(0.,0.,0.,0.);
      if(partons.size() != 0) {
	if (dRmin > dRmatch)
	  continue;
	partontop = partons[pmin];
      }
      
      counter++;
      PseudoJet f = CA_fat_jet;
      PseudoJet filt = filter(f);
      PseudoJet rec_top =  tagger(CA_fat_jet);
      PseudoJet sd_jet = sd(CA_fat_jet);
      fatjet->SetPxPyPzE(f.px(),f.py(),f.pz(),f.e());
      filtered->SetPxPyPzE(filt.px(),filt.py(),filt.pz(),filt.e());
      softdropjet->SetPxPyPzE(sd_jet.px(),sd_jet.py(),sd_jet.pz(),sd_jet.e());
      top->SetPxPyPzE(partontop.px(),partontop.py(),partontop.pz(),partontop.e());
      tau2 = nsub2.result(f);
      tau3 = nsub3.result(f);
      tau2_filt = nsub2.result(filt);
      tau3_filt = nsub3.result(filt);
      tau2_sd = nsub2.result(sd_jet);
      tau3_sd = nsub3.result(sd_jet);

      if (rec_top.has_structure_of<HTT>()) {
	htt_tagged = rec_top.structure_of<HTT>().is_tagged();
      }
      
      for (unsigned c = 0; c < CA_fat_jet.constituents().size(); c++) {
	PseudoJet p = CA_fat_jet.constituents()[c];
	veta.push_back(p.eta());
	vphi.push_back(p.phi());
	ve.push_back(p.e());
      }
      t.Fill();
    }
  }
  t.Write();
  fout.Close();
}
//------------------------------------------------------------------------------
