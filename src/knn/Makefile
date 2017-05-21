CC          = g++
OPTIMIZE    = -O2
#CFLAGS      = -Wall -g -ansi -pedantic -ggdb -std=c++11 $(OPTIMIZE)
CFLAGS      = -Wall -g -ansi -pedantic -ggdb -std=c++11
OBJS        = knn.o

all: clean knn_main

test: test.o
	$(CC) -o $@ $^

knn_main: $(OBJS) knn_main.o
	$(CC) -o $@ $^

%.o: %.cpp
	$(CC) -c $(CFLAGS) -x c++ $< -o $@

clean:
	rm -f *.o knn knn_main test

.PHONY: all clean
