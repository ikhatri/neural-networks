CXX = g++
FLAGS = -std=c++11 -O3 -g #-D'fopen_s(pFile,filename,mode)=((*(pFile))=fopen((filename),(mode)))==NULL'

.DEFAULT = all

all: bin bin/network

bin:
	mkdir bin

bin/network: src/MNISTParser.h src/network.h src/network.cpp
	$(CXX) $(FLAGS) src/network.cpp -o bin/network

clean:
	rm -rf bin/*
