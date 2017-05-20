CC          = g++
OPTIMIZE    = -O2
OBJS        = svd.o

# For performance testing:
CFLAGS      = -Wall -DNDEBUG -ansi -pedantic -std=c++11 $(OPTIMIZE)

# For debugging:
# CFLAGS 		= -Wall -ggdb -O0 -DDEBUG_ZERO

all: clean svd

svd: svd.o
	$(CC) -o $@ $^


%.o: %.cpp
	$(CC) -c $(CFLAGS) -x c++ $< -o $@

clean:
	rm -f *.o svd

.PHONY: all clean
