SOURCES = QHTT.cc FWM.cc LowPt.cc
OBJECTS = $(SOURCES:.cc=.o)
CC = g++ 

# PATH
HTT_DIR = ../
GSL_DIR = ${GSL}

# FLAGS
CXXFLAGS += $(shell fastjet-config --cxxflags) 

# INCLUDES
HTT_INCLUDES = -I$(HTT_DIR)
GSL_INCLUDES = -I$(GSL_DIR)/include

# LIBS 
# QJETS_LIBS = -L./qjets/lib -lQjets
# GSL_LIBS = -L$(GSL_DIR)/lib -lgsl

# RULES
all: $(OBJECTS)

FWM.o: FWM.cc
	$(CC) $(CXXFLAGS) $(HTT_INCLUDES) $(GSL_INCLUDES) -c $<

LowPt.o: LowPt.cc FWM.o
	$(CC) $(CXXFLAGS) $(HTT_INCLUDES) $(GSL_INCLUDES) -c $<

QHTT.o: QHTT.cc
	cd qjets; make
	$(CC) $(CXXFLAGS) $(HTT_INCLUDES) -c $<

clean:
	cd qjets; make clean
	rm -f $(OBJECTS)
