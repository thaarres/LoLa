// Example distributed with HEPTopTagger2.0 FastJet Plugin

#include <iostream>
#include <fstream>

#include "fastjet/PseudoJet.hh"

#include "./HTT.hh"

using namespace std;
using namespace fastjet;

int main(int argc, char *argv[]) {
  // Read input and convert MeV->GeV
  ifstream fin("input.dat", ifstream::in);  
  vector<PseudoJet> input_clusters(0);
  while (!fin.eof()) {
    double x, y, z, e;
    fin >> x >> y >> z >> e;
    PseudoJet p(x/1000., y/1000., z/1000., e/1000.);
    input_clusters.push_back(p);
  }
  cout << "Read event: " << input_clusters.size() << " particles are read" << endl;
  
  //  jet definition 
  double conesize(1.5);
  JetDefinition jet_def(cambridge_algorithm, conesize);

  // run the jet finding; find the hardest jet with pt > 200 GeV
  ClusterSequence clust_seq(input_clusters, jet_def);  
  double ptmin_jet(200.);
  vector<PseudoJet> jets = sorted_by_pt(clust_seq.inclusive_jets(ptmin_jet));

  //create HTT instance and print settings
  HTT tagger;
  tagger.get_setting();

  for(unsigned ijet = 0; ijet<jets.size(); ijet++) {
    //run tagger on jet
    PseudoJet top = tagger(jets[ijet]);
    
    if (top.has_structure_of<HTT>()) {
      cout << "tag!" << endl;
      cout << "Input fatjet: " << ijet << "  pT = " << jets[ijet].perp() << endl;      
      cout << "Output: pT = " << top.perp() << " Mass = " << top.m() << endl;
      top.structure_of<HTT>().get_info();
    } else {
      cout << "no tag!" << endl;

      }
  } // end of top tagger
  return 0;
}// end of main
