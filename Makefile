CXX = g++
FLAGS = -std=c++11 -O3 -g

.DEFAULT = all

all: bin bin/network

bin:
	mkdir bin

bin/network: src/network.h src/network.cpp
	$(CXX) $(FLAGS) src/network.cpp -o bin/network

clean:
	rm -rf bin/*
