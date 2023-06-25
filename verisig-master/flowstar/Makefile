CXX = g++
include makefile.local
LIBS = -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lyaml-cpp -ljsoncpp
CFLAGS = -I . -I $(GMP_HOME) -I /opt/homebrew/include -I ./ctpl -g -std=c++11
LINK_FLAGS = -g -L $(GMP_LIB_HOME) -pthread -L /opt/homebrew/lib
OBJS = Interval.o Matrix.o Polynomial.o Monomial.o Constraints.o Geometry.o TaylorModel.o Continuous.o expression.o DNNResets.o DNN.o AddedResets.o Hybrid.o NNTaylorModel.o NNPolynomial.o NNMonomial.o

all: flowstar lib

flowstar: $(OBJS) lex.yy.o modelParser.tab.o modelParser.o
	g++ -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

lib: $(OBJS) lex.yy.o modelParser.tab.o
	ar rcs libflowstar.a $^

%.o: %.cc
	$(CXX) -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -c $(CFLAGS) -o $@ $<

modelParser.tab.c: modelParser.y
	bison -d -v modelParser.y

lex.yy.c: modelLexer.l modelParser.tab.c
	flex modelLexer.l

clean: 
	rm -f flowstar *.o libflowstar.a *~ modelParser.tab.c modelParser.tab.h modelParser.output lex.yy.c
