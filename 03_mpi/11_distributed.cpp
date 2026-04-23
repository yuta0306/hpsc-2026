#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Body ibody[N/size], jbody[N/size], sbody[N/size];
  srand48(rank);
  for(int i=0; i<N/size; i++) {
    ibody[i].x = jbody[i].x = drand48();
    ibody[i].y = jbody[i].y = drand48();
    ibody[i].m = jbody[i].m = drand48();
    ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = 0;
  }
  int send_to = (rank - 1 + size) % size;
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);
  MPI_Win win;
  MPI_Win_create(jbody, (N/size) * sizeof(Body), sizeof(Body),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  for(int irank=0; irank<size; irank++) {
    for(int i=0; i<N/size; i++) sbody[i] = jbody[i];
    MPI_Win_fence(0, win);
    MPI_Put(sbody, N/size, MPI_BODY, send_to, 0, N/size, MPI_BODY, win);
    MPI_Win_fence(0, win);
    for(int i=0; i<N/size; i++) {
      for(int j=0; j<N/size; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }
  }
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        printf("%d %g %g\n",i+rank*N/size,ibody[i].fx,ibody[i].fy);
      }
    }
  }
  MPI_Win_free(&win);
  MPI_Finalize();
}

/*
  MPI_Send/MPI_Recv
    mpirun -n 8 a.out
    0 27.4217 -45.9659
    1 -80.5552 -25.5702
    2 74.1105 30.6659
    3 -123.102 -34.0765
    5 -11.4538 13.8815
    6 7.43006 4.5741
    7 -7.3739 -21.9327
    8 -168.199 -48.9042
    10 92.6142 -63.1341
    11 40.7691 18.7558
    12 0.465195 18.5385
    13 11.4524 6.1224
    15 0.633036 -17.2224
    16 123.646 53.6965
    17 -13.6764 -53.7064
    18 104.164 27.1904

  MPI_Put
    mpirun -n 8 a.out
    0 27.4217 -45.9659
    1 -80.5552 -25.5702
    2 74.1105 30.6659
    3 -123.102 -34.0765
    5 -11.4538 13.8815
    6 7.43006 4.5741
    7 -7.3739 -21.9327
    8 -168.199 -48.9042
    10 92.6142 -63.1341
    11 40.7691 18.7558
    12 0.465195 18.5385
    13 11.4524 6.1224
    15 0.633036 -17.2224
    16 123.646 53.6965
    17 -13.6764 -53.7064
    18 104.164 27.1904
*/