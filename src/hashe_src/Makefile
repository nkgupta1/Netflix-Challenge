CC          = g++
OPTIMIZE    = -O2
CFLAGS      = -Wall -g -ansi -pedantic -ggdb -std=c++11 $(OPTIMIZE)
# OBJS        = player.o board.o

all: clean preprocess

preprocess: preprocess.o util.o
	$(CC) -o $@ $^
	
# $(PLAYERNAME): $(OBJS) wrapper.o
# 	$(CC) -o $@ $^

# testgame: testgame.o
# 	$(CC) -o $@ $^

# testminimax: $(OBJS) testminimax.o
# 	$(CC) -o $@ $^

%.o: %.cpp
	$(CC) -c $(CFLAGS) -x c++ $< -o $@
	
# java:
# 	make -C java/

# cleanjava:
# 	make -C java/ clean

clean:
	rm -f *.o util preprocess

.PHONY: all clean
