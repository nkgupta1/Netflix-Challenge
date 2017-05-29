CC          = g++
OPTIMIZE    = -O2
CFLAGS      = -Wall -g -ansi -pedantic -ggdb -std=c++11 $(OPTIMIZE)
OBJS        = util.o

all: clean preprocess rbm

preprocess: $(OBJS) preprocess.o
	$(CC) -o $@ $^

rbm: $(OBJS) rbm.o
	$(CC) -o $@ $^

cond_rbm: $(OBJS) cond_rbm.o
	$(CC) -o $@ $^

%.o: %.cpp
	$(CC) -c $(CFLAGS) -x c++ $< -o $@

clean:
	rm -f *.o util preprocess rbm cond_rbm

.PHONY: all clean
