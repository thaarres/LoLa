TARGET = fatjets
####################################
CC = $(shell root-config --cxx)
OBJS = $(TARGET).o HEPTopTagger/HTT.o
######### ROOT #####################
ROOTCFLAGS =  $(shell root-config --cflags)
ROOTLIBS   = $(shell root-config --libs)
CXXFLAGS   = $(shell fastjet-config --cxxflags)
######### HEPMC ####################
HepMCdir = ${HEPMC}
HepMClib = -L$(HepMCdir)/lib -lHepMC
######### FASTJET ##################
FASTJETLIB  = -L${FASTJET}/lib -lfastjet -lNsubjettiness
######### DELPHES ######################
DELPHESlib = -L${DELPHES} -lDelphes
#-------- INCLUDE -----------------#
INCLUDE=./include
INCLUDES = -I${INCLUDE}
INCLUDES += -I${HepMCdir}/include 
INCLUDES += -I${FASTJET}/include 
INCLUDES += -I${ROOTSYS}/include
INCLUDES += -I${ROOTSYS}/include/root
INCLUDES += -I${DELPHES} -I${DELPHES}/external
INCLUDES += -I./
#########  FLAGS, LIBS  ############
CXXFLAGS += ${INCLUDES}
CXXLIBS = ${HepMClib} ${FASTJETLIB} $(ROOTLIBS) $(DELPHESlib)
#########################################################

all:
	cd HEPTopTagger; make
	make fatjets

fatjets: $(OBJS)
	$(CC) $(CXXFLAGS) $(ROOTCFLAGS) $(OBJS) $(CXXLIBS) -o $@

clean:
	rm -f $(TARGET) $(OBJS)
