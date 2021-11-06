CC = g++
MPICC = g++
#CFLAGS = -std=c++11 -fopenmp -O3 -Wall -march=native -mtune=native
CFLAGS = -std=c++11 -fopenmp -O3 -Wall -march=native -mtune=native -D USE_MKL -lblas

all: pWord2Vec
#pWord2Vec_mpi 

pWord2Vec: pWord2Vec.cpp
	$(CC) pWord2Vec.cpp -o pWord2Vec $(CFLAGS)
pWord2Vec_mpi: pWord2Vec_mpi.cpp
	$(MPICC) pWord2Vec_mpi.cpp -o pWord2Vec_mpi $(CFLAGS)
clean:
	rm -rf pWord2Vec pWord2Vec_mpi 
