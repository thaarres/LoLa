SOURCES=HTT.cc example.cc
OBJECTS=$(SOURCES:.cc=.o)
CC = g++ 

#FLAGS
CXXFLAGS += $(shell fastjet-config --cxxflags) 

# LIBS
FASTJET_LIBS  = $(shell fastjet-config --libs) -lNsubjettiness
CXXLIBS = $(FASTJET_LIBS)

# RULES
example: $(OBJECTS)
	$(CC) $(CXXFLAGS) $(OBJECTS) $(CXXLIBS) -o $@

.cc.o:
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $<

clean:
	rm -f $(OBJECTS) example
