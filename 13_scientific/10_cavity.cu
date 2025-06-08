#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>


using namespace std;


__global__ void compute_b(int nx, int ny, double dx, double dy, double dt, double rho, float *u, float *v, float *b){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1){
		float du_dx = (u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2.0f * dx);
		float dv_dy = (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0f * dy);
		float du_dy = (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2.0f * dy);
		float dv_dx = (v[j * nx + (i + 1)] - v[j * nx + (i - 1)]) / (2.0f * dx);

		b[j * nx + i] = rho * (1.0f / dt * (du_dx + dv_dy) - (du_dx)*(du_dx) - 2.0f * ((du_dy) * (dv_dx)) - (dv_dy)*(dv_dy));
    
	}
}

__global__ void compute_pressure(int nx, int ny, double dx, double dy, float *p, float *pn, float *b){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1){
		float pni = pn[j * nx + (i + 1)] + pn[j * nx + (i - 1)];
		float pnj = pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i];
		p[j * nx + i] = (dy*dy * pni + dx*dx * pnj - b[j * nx + i] * dx*dx * dy*dy) / (2.0f * (dx*dx + dy*dy));    
	}
}

__global__ void boundary_pressure_lr(int nx, int ny, float *p){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < ny){
    p[idx * nx + 0] = p[idx * nx + 1];
    p[idx * nx + (nx - 1)] = p[idx * nx + (nx - 2)];
  }
}

__global__ void boundary_pressure_tb(int nx, int ny, float *p){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < nx){
		p[0 * nx + idx] = p[1 * nx + idx];
		p[(ny - 1) * nx + idx] = 0.0f;
  }
}

__global__ void compute_velocity(int nx, int ny, double dx, double dy, double dt, double rho, double nu,
 								 float *u, float *v, float *un, float *vn, float *p){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > 0 && i < nx - 1 && j > 0 && j < ny - 1){
    float und  = un[j * nx + i];
    float unip = un[j * nx + (i + 1)];
    float unim = un[j * nx + (i - 1)];
    float unjp = un[(j + 1) * nx + i];
    float unjm = un[(j - 1) * nx + i];
    
    float vnd  = vn[j * nx + i];      
    float vnip = vn[j * nx + (i + 1)];
    float vnim = vn[j * nx + (i - 1)];
    float vnjp = vn[(j + 1) * nx + i];
    float vnjm = vn[(j - 1) * nx + i];
    
    float pip  = p[j * nx + (i + 1)];
    float pim  = p[j * nx + (i - 1)];
    float pjp  = p[(j + 1) * nx + i];
    float pjm  = p[(j - 1) * nx + i];

		u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (und - unim)
              									   - vn[j * nx + i] * dt / dy * (und - unjm)
              									   - dt / (2.0f * rho * dx) * (pip - pim)
              									   + nu * dt / (dx*dx) * (unip - 2.0f * und + unim)
              									   + nu * dt / (dy*dy) * (unjp - 2.0f * und + unjm);
                                                          
		v[j * nx + i] = vn[j * nx + i] - un[j * nx + i] * dt / dx * (vnd - vnim)
              									   - vn[j * nx + i] * dt / dy * (vnd - vnjm)
              									   - dt / (2.0f * rho * dy) * (pjp - pjm)
              									   + nu * dt / (dx*dx) * (vnip - 2.0f * vnd + vnim)
              									   + nu * dt / (dy*dy) * (vnjp - 2.0f * vnd + vnjm);
	}
}

__global__ void boundary_velocity_lr(int nx, int ny, float *u, float *v){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < ny){
    	u[idx * nx + 0] = 0.0f;
     	u[idx * nx + nx - 1] = 0.0f;
    	v[idx * nx + 0] = 0.0f;
    	v[idx * nx + nx - 1] = 0.0f;
	}
}

__global__ void boundary_velocity_tb(int nx, int ny, float *u, float *v){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < nx){
	    u[0 * nx + idx] = 0.0f;
    	u[(ny - 1) * nx + idx] = 1.0f;
    	v[0 * nx + idx] = 0.0f;
    	v[(ny - 1) * nx + idx] = 0.0f;
	}
}

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  int size = nx * ny;

  float *u, *v, *p, *b;
  u = (float *)malloc(size * sizeof(float));
  v = (float *)malloc(size * sizeof(float));
  p = (float *)malloc(size * sizeof(float));
  b = (float *)malloc(size * sizeof(float));

  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j * nx + i] = 0.0f;
      v[j * nx + i] = 0.0f;
      p[j * nx + i] = 0.0f;
      b[j * nx + i] = 0.0f;
    }
  }

  float *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;
  cudaMalloc(&d_u, size * sizeof(float));
  cudaMalloc(&d_v, size * sizeof(float));
  cudaMalloc(&d_p, size * sizeof(float));
  cudaMalloc(&d_b, size * sizeof(float));
  cudaMalloc(&d_un, size * sizeof(float));
  cudaMalloc(&d_vn, size * sizeof(float));
  cudaMalloc(&d_pn, size * sizeof(float));

  cudaMemcpy(d_u, u, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, p, size * sizeof(float), cudaMemcpyHostToDevice);


  const int BS = 32;
  dim3 grid((nx + BS -1) / BS, (ny + BS -1) / BS);
  dim3 block(BS, BS);
  int lrgrid = (ny + BS - 1) / BS;
  int tbgrid = (nx + BS - 1) / BS;


  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {
  	
  	compute_b<<<grid, block>>>(nx, ny, dx, dy, dt, rho, d_u, d_v, d_b);
	  cudaDeviceSynchronize();
    
    for (int it=0; it<nit; it++) {
  	  
  	  cudaMemcpy(d_pn, d_p, size * sizeof(float), cudaMemcpyDeviceToDevice);
  
  	  compute_pressure<<<grid, block>>>(nx, ny, dx, dy, d_p, d_pn, d_b);
  	  cudaDeviceSynchronize();
  	  
	  
      boundary_pressure_lr<<<lrgrid, BS>>>(nx, ny, d_p);
      cudaDeviceSynchronize();
      boundary_pressure_tb<<<tbgrid, BS>>>(nx, ny, d_p);
      cudaDeviceSynchronize();
      
    }

    cudaMemcpy(d_un, d_u, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vn, d_v, size * sizeof(float), cudaMemcpyDeviceToDevice);

    compute_velocity<<<grid, block>>>(nx, ny, dx, dy, dt, rho, nu, d_u, d_v, d_un, d_vn, d_p);
    cudaDeviceSynchronize();

    boundary_velocity_lr<<<lrgrid, BS>>>(nx, ny, d_u, d_v);
    cudaDeviceSynchronize();
    boundary_velocity_tb<<<tbgrid, BS>>>(nx, ny, d_u, d_v);
    cudaDeviceSynchronize();


    if (n % 10 == 0) {
      cudaMemcpy(p, d_p, size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(v, d_v, size * sizeof(float), cudaMemcpyDeviceToHost);
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j * nx + i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j * nx + i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j * nx + i] << " ";
      pfile << "\n";
    }
  }
  
  free(u);
  free(v);
  free(p);
  free(b);
  
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_p);
  cudaFree(d_b);
  cudaFree(d_un);
  cudaFree(d_vn);
  cudaFree(d_pn);

  ufile.close();
  vfile.close();
  pfile.close();
  
  return 0;
}
