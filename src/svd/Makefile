CC          = g++
OPTIMIZE    = -O2
OBJS        = svd.o svdpp.o

# For performance testing:
CFLAGS      = -Wall -DNDEBUG -ansi -pedantic -std=c++11 $(OPTIMIZE)

# For debugging:
# CFLAGS 		= -Wall -ggdb -O0 -DDEBUG_ZERO -std=c++11

all: clean svd svdpp

svd: svd.o
	$(CC) -o $@ $^

svdpp: svdpp.o
	$(CC) -o $@ $^

%.o: %.cpp
	$(CC) -c $(CFLAGS) -x c++ $< -o $@

clean:
	rm -f *.o svd svdpp

.PHONY: all clean
