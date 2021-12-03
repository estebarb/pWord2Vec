CC = g++
MPICC = g++
#CFLAGS = -std=c++11 -fopenmp -O3 -Wall -march=native -mtune=native
CFLAGS = -I/opt/arm/armpl_21.1_gcc-9.3/include -std=c++11 -fopenmp -O3 -Wall -march=native -mtune=native -D USE_MKL
LDFLAGS = -L/opt/arm/armpl_21.1_gcc-9.3/lib -fopenmp -larmpl_lp64 -lgfortran -lastring -lamath

all: pWord2Vec
#pWord2Vec_mpi 

pWord2Vec: pWord2Vec.cpp
	$(CC) -c pWord2Vec.cpp -o pWord2Vec.o $(CFLAGS)
	$(CC) pWord2Vec.o -o pWord2Vec $(LDFLAGS)
pWord2Vec_mpi: pWord2Vec_mpi.cpp
	$(MPICC) pWord2Vec_mpi.cpp -o pWord2Vec_mpi $(CFLAGS)
clean:
	rm -rf pWord2Vec pWord2Vec_mpi 
