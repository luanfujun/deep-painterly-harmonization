extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math_constants.h>
#include <math_functions.h>
#include <stdint.h>
#include <unistd.h>
#include <omp.h>
#include <getopt.h>
#include "curand_kernel.h"

#define TB 256
#define EPS 0.1

#undef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#undef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))


__host__ __device__ int clamp(int x, int x_max, int x_min) {//assume x_max >= x_min
	if (x > x_max)
	{
		return x_max;
	}
	else if (x < x_min)
	{
		return x_min;
	}
	else
	{
		return x;
	}
}

__host__ __device__ int cuMax(int a, int b) {
	if (a > b) {
		return a;
	}
	else {
		return b;
	}
}
__host__ __device__ int cuMin(int a, int b) {
	if (a < b) {
		return a;
	}
	else {
		return b;
	}
}

__device__ float MycuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	 return curand_uniform(&state);

}

__device__ void InitcuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(i, 0, 0, &state);

}

THCState* getCutorchState(lua_State* L)
{
	lua_getglobal(L, "cutorch");
	lua_getfield(L, -1, "getState");
	lua_call(L, 0, 1);
	THCState *state = (THCState*) lua_touserdata(L, -1);
	lua_pop(L, 2);
	return state;
}

void checkCudaError(lua_State *L) {
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		luaL_error(L, cudaGetErrorString(status));
	}
}

THCudaTensor *new_tensor_like(THCState *state, THCudaTensor *x)
{
	THCudaTensor *y = THCudaTensor_new(state);
	THCudaTensor_resizeAs(state, y, x);
	return y;
}
 
__global__ void histogram_kernel(
	float *I, float *minI, float *maxI, float *mask, 
	int nbins, int c, int h, int w, float *hist
)
{
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w; 

	if (_id < c * size) {
		int id = _id % size, dc = _id / size;

		if (mask[id] < EPS)
			return ;

		float val  = I[_id];

		float _minI = minI[dc];
		float _maxI = maxI[dc];


		if (_minI == _maxI) {
			_minI -= 1;
			_maxI += 1;
		}

		if (_minI <= val && val <= _maxI) {
			int idx = MIN((val - _minI) / (_maxI - _minI) * nbins, nbins-1);
			int index = dc * nbins + idx;
			atomicAdd(&hist[index], 1.0f);
		}
		
	}

	return ;
}

int histogram(lua_State *L) {
	THCState *state     = getCutorchState(L);
	THCudaTensor *I     = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	int nbins           = luaL_checknumber(L, 2);
	THCudaTensor *minI  = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *maxI  = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *mask  = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");

	int c = THCudaTensor_size(state, I, 0);
	int h = THCudaTensor_size(state, I, 1);
	int w = THCudaTensor_size(state, I, 2);

	THCudaTensor *hist = THCudaTensor_new(state);
	THCudaTensor_resize2d(state, hist, c, nbins);
	THCudaTensor_zero(state, hist);

	histogram_kernel<<<(c*h*w-1)/TB+1, TB>>>(
		THCudaTensor_data(state, I),
		THCudaTensor_data(state, minI),
		THCudaTensor_data(state, maxI),
		THCudaTensor_data(state, mask),
		nbins, c, h, w,
		THCudaTensor_data(state, hist)
	);
	checkCudaError(L);	

	luaT_pushudata(L, hist, "torch.CudaTensor");
	return 1;
}

void histogram_cpu_kernel(
	float *I, float *minI, float *maxI, float *mask, 
	int nbins, int c, int h, int w, float *hist
)
{
	int size = h * w;
#pragma omp parallel for 
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int id = y * w + x;
			if (mask[id] < EPS)
				continue;

			for (int dc = 0; dc < c; dc++)
			{
				float val   = I[dc * size + id];
				float _minI = minI[dc];
				float _maxI = maxI[dc];
				if (_minI == _maxI) {
					_minI -= 1;
					_maxI += 1;
				}
				if (_minI <= val && val <= _maxI) {
					int idx = MIN((val - _minI) / (_maxI - _minI) * nbins, nbins-1);
					int index = dc * nbins + idx;
					// atomicAdd(&hist[index], 1.0f);
					#pragma omp atomic 
					hist[index]++;
				}
			}
		}

	return ;
}

int histogram_cpu(lua_State *L) {
	// THCState *state     = getCutorchState(L);
	THFloatTensor *I    = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
	int nbins           = luaL_checknumber(L, 2);
	THFloatTensor *minI  = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");
	THFloatTensor *maxI  = (THFloatTensor*)luaT_checkudata(L, 4, "torch.FloatTensor");
	THFloatTensor *mask  = (THFloatTensor*)luaT_checkudata(L, 5, "torch.FloatTensor");

	int c = THFloatTensor_size(I, 0);
	int h = THFloatTensor_size(I, 1);
	int w = THFloatTensor_size(I, 2);

	THFloatTensor *hist = THFloatTensor_new();
	THFloatTensor_resize2d(hist, c, nbins);
	THFloatTensor_zero(hist);

	histogram_cpu_kernel(
		THFloatTensor_data(I),
		THFloatTensor_data(minI),
		THFloatTensor_data(maxI),
		THFloatTensor_data(mask),
		nbins, c, h, w,
		THFloatTensor_data(hist)
	);

	luaT_pushudata(L, hist, "torch.FloatTensor");
	return 1;
}

__global__ void hist_remap2_kernel(
	float *I, int nI, float *mI, float *histJ, float *cumJ, 
	float *_minJ, float *_maxJ, int nbins, 
	float *_sortI, int *_idxI, float *R, int c, int h, int w
)
{
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;

	if (_id < c * size) {
		// _id = dc * size + id
		int id = _id % size, dc = _id / size;

		float minJ  = _minJ[dc];
		float maxJ  = _maxJ[dc];
		float stepJ = (maxJ - minJ) / nbins;

		int idxI = _idxI[_id] - 1;
		if (mI[idxI] < EPS)
			return ;
		int offset = h * w - nI;
		 
		int cdf = id - offset;

		int s = 0;
		int e = nbins - 1;
		int m = (s + e) / 2;
		int binIdx = -1;

		while (s <= e) {
			// special handling for range boundary
			float cdf_e = m == nbins - 1 ? 
						  cumJ[dc * nbins + m] + 0.5f : 
						  cumJ[dc * nbins + m];
			float cdf_s = m == 0         ? 
						  -0.5f : 
						  cumJ[dc * nbins + m - 1];

			if (cdf >= cdf_e) {
				s = m + 1;
				m = (s + e) / 2;
			} else if (cdf < cdf_s) {
				e = m - 1;
				m = (s + e) / 2;
			} else {
				binIdx = m;    break;
			}
		}

		float hist  = histJ[dc * nbins + binIdx];
		float cdf_e = cumJ[dc * nbins + binIdx];
		float cdf_s = cdf_e - hist;
		float ratio = MIN(MAX((cdf - cdf_s) / (hist + 1e-8), 0.0f), 1.0f);
		float activation = minJ + (static_cast<float>(binIdx) + ratio) * stepJ;
		R[dc * size + idxI] = activation;
	}

	return ;
}

int hist_remap2(lua_State *L) {
	THCState *state       = getCutorchState(L);
	THCudaTensor *I       = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	int nI                = luaL_checknumber(L, 2);
	THCudaTensor *mI      = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *histJ   = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *cumJ    = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *minJ    = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
	THCudaTensor *maxJ    = (THCudaTensor*)luaT_checkudata(L, 7, "torch.CudaTensor");
	int nbins             = luaL_checknumber(L, 8);
	THCudaTensor *sortI   = (THCudaTensor*)luaT_checkudata(L, 9, "torch.CudaTensor");
	THCudaIntTensor *idxI = (THCudaIntTensor*)luaT_checkudata(L, 10, "torch.CudaIntTensor");
	THCudaTensor *R       = (THCudaTensor*)luaT_checkudata(L, 11, "torch.CudaTensor");
	
	int c = THCudaTensor_size(state, I, 0);
	int h = THCudaTensor_size(state, I, 1);
	int w = THCudaTensor_size(state, I, 2);

	hist_remap2_kernel<<<(c*h*w-1)/TB+1, TB>>>(
		THCudaTensor_data(state, I),
		nI,
		THCudaTensor_data(state, mI),
		THCudaTensor_data(state, histJ),
		THCudaTensor_data(state, cumJ),
		THCudaTensor_data(state, minJ),
		THCudaTensor_data(state, maxJ),
		nbins, 
		THCudaTensor_data(state, sortI),
		THCudaIntTensor_data(state, idxI),
		THCudaTensor_data(state, R),
		c, h, w
	);
	checkCudaError(L);	

	return 0;
}


__global__ void patchmatch_conv_kernel(
	float *input, float *target, float *conv, 
	int patch, int c1, int h1, int w1, int h2, int w2, 
	int *mask = NULL
)
{	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;
	int N = size1 * size2;

	if (id < N) {
		conv[id] = -1; 
		// id = id1 * size2 + id2
		int id1 = id / size2, id2 = id % size2;
		if (mask && mask[id1] == 0)
			return ;

		int x1 = id1 % w1, y1 = id1 / w1;
		int x2 = id2 % w2, y2 = id2 / w2;
		int kernel_radius  = (patch - 1) / 2;
		double conv_result = 0;
		// double sigma       = 0.5;
		// double sum_weight  = 0;
		// int cnt            = 0;
		for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
			for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
				int xx1 = x1 + dx, yy1 = y1 + dy;
				int xx2 = x2 + dx, yy2 = y2 + dy;	
				if (0 <= xx1 && xx1 < w1 && 0 <= yy1 && yy1 < h1 &&
					0 <= xx2 && xx2 < w2 && 0 <= yy2 && yy2 < h2) 
				{	
					int _id1 = yy1 * w1 + xx1, _id2 = yy2 * w2 + xx2;
					// float weight = exp(-(dx*dx + dy*dy) / (2*sigma*sigma));
					for (int c = 0; c < c1; c++) {
						float term1 = input[c * size1 + _id1];  
						float term2 = target[c * size2 + _id2];  
						conv_result += term1 * term2;
						// conv_result += (term1 - term2) * (term1 - term2) * weight;
					}	
					// cnt++;
					// sum_weight += weight;
				}
			}	
		}

		// conv[id] = conv_result / cnt;
		// conv[id] = conv_result / sum_weight;
		conv[id] = conv_result;
	}

	return ;
}

__global__ void patchmatch_argmax_kernel(
	float *conv, int *correspondence, int patch,
	int c1, int h1, int w1, int h2, int w2
) 
{	
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;
	int kernel_radius = (patch - 1) / 2;
	if (id1 < size1) {
		float conv_max = -FLT_MAX;
		int y1 = id1 / w1, x1 = id1 % w1;

		for (int y2 = 0; y2 < h2; y2++) {
			for (int x2 = 0; x2 < w2; x2++) {
				int id2 = y2 * w2 + x2;
				int id = id1 * size2 + id2;
				float conv_result = conv[id];
				
				if (x2 < kernel_radius && !(x1 < kernel_radius))
					continue;
				if (x2 > w2 - 1 - kernel_radius && !(x1 > w1 - 1 - kernel_radius))
					continue;
				if (y2 < kernel_radius && !(y1 < kernel_radius))
					continue;
				if (y2 > h2 - 1 - kernel_radius && !(y1 > h1 - 1 - kernel_radius))
					continue;
				 
				if (conv_result > conv_max) {
					conv_max = conv_result;
					correspondence[id1 * 2 + 0] = x2;
					correspondence[id1 * 2 + 1] = y2;
				}
				// if (conv_result < conv_min) {
				// 	conv_min = conv_result;
				// 	correspondence[id1 * 2 + 0] = x2;
				// 	correspondence[id1 * 2 + 1] = y2;
				// }
			}
		}

	}

	return ;
}

int patchmatch(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *input  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int           patch  = luaL_checknumber(L, 3);

	int c1 = THCudaTensor_size(state, input, 0);
	int h1 = THCudaTensor_size(state, input, 1);
	int w1 = THCudaTensor_size(state, input, 2);

	int c2 = THCudaTensor_size(state, target, 0);
	int h2 = THCudaTensor_size(state, target, 1);
	int w2 = THCudaTensor_size(state, target, 2);

	THCudaTensor *conv = THCudaTensor_new(state);
	THCudaTensor_resize2d(state, conv, h1*w1, h2*w2);
	THCudaTensor_zero(state, conv);
	
	assert(c1 == c2);
	int N = h1*w1*h2*w2;
	patchmatch_conv_kernel<<<(N-1)/TB+1, TB>>>(
		THCudaTensor_data(state, input),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, conv),
		patch, 
		c1,
		h1, w1,
		h2, w2
	);
	checkCudaError(L);

	THCudaIntTensor *correspondence = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, correspondence, h1, w1, 2);
	THCudaIntTensor_zero(state, correspondence);

	patchmatch_argmax_kernel<<<(h1*w1-1)/TB+1, TB>>>(
		THCudaTensor_data(state, conv),
		THCudaIntTensor_data(state, correspondence),
		patch, 
		c1,
		h1, w1,
		h2, w2		
	);
	checkCudaError(L);	

	THCudaTensor_free(state, conv);

	luaT_pushudata(L, correspondence, "torch.CudaIntTensor");
	return 1;
}

int conv(lua_State *L) {
	THCState        *state  = getCutorchState(L);
	THCudaTensor    *input  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor    *target = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int              patch  = luaL_checknumber(L, 3);
	THCudaIntTensor *mask   = (THCudaIntTensor*)luaT_checkudata(L, 4, "torch.CudaIntTensor");

	int c1 = THCudaTensor_size(state, input, 0);
	int h1 = THCudaTensor_size(state, input, 1);
	int w1 = THCudaTensor_size(state, input, 2);

	int c2 = THCudaTensor_size(state, target, 0);
	int h2 = THCudaTensor_size(state, target, 1);
	int w2 = THCudaTensor_size(state, target, 2);

	THCudaTensor *conv = THCudaTensor_new(state);
	THCudaTensor_resize2d(state, conv, h1*w1, h2*w2);
	THCudaTensor_zero(state, conv);
	
	assert(c1 == c2);
	int N = h1*w1*h2*w2;
	patchmatch_conv_kernel<<<(N-1)/TB+1, TB>>>(
		THCudaTensor_data(state, input),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, conv),
		patch, 
		c1,
		h1, w1,
		h2, w2,
		THCudaIntTensor_data(state, mask)
	);
	checkCudaError(L);

	luaT_pushudata(L, conv, "torch.CudaTensor");
	return 1;
}

__global__ void avg_vote_kernel(
	float *A, float *B, int *corrAB, 
	int patch, int c, int h, int w
)
{
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;
	int radius = patch / 2;
	if (_id < c * size) {
		// _id = dc * size + id
		int id = _id % size, dc = _id / size;
		int x1 = id % w, y1 = id / w;
		double sum = 0;
		int    cnt = 0;
		for (int dx = -radius; dx <= radius; dx++) {
			for (int dy = -radius; dy <= radius; dy++) {
				int new_x1 = x1 + dx, new_y1 = y1 + dy;
				
				if (new_x1 >= 0 && new_x1 < w && new_y1 >= 0 && new_y1 < h) {
					int new_id1 = new_y1 * w + new_x1;
					int x2 = corrAB[new_id1 * 2 + 0];
					int y2 = corrAB[new_id1 * 2 + 1];
					int new_x2 = x2 - dx, new_y2 = y2 - dy;

					if (new_x2 >= 0 && new_x2 < w && new_y2 >= 0 && new_y2 < h) {
						int new_id2 = new_y2 * w + new_x2;
						sum += A[dc * size + new_id2];
						cnt++;
					}
				}
			}
		}
		if (cnt != 0)
			B[dc * size + id] = sum / cnt;

	}
	return ;
}

int avg_vote(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaIntTensor *corrAB  = (THCudaIntTensor*)luaT_checkudata(L, 1, "torch.CudaIntTensor");
	THCudaTensor    *A       = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int              patch   = luaL_checknumber(L, 3);
 	THCudaTensor    *B       = new_tensor_like(state, A);
 	THCudaTensor_zero(state, B);

	int c = THCudaTensor_size(state, A, 0);
	int h = THCudaTensor_size(state, A, 1);
	int w = THCudaTensor_size(state, A, 2);

	// int h1 = THCudaIntTensor_size(state, corrAB, 0);
	// int w1 = THCudaIntTensor_size(state, corrAB, 1);
	// int c1 = THCudaIntTensor_size(state, corrAB, 2);

	avg_vote_kernel<<<(c*h*w-1)/TB+1, TB>>>(
		THCudaTensor_data(state, A),
		THCudaTensor_data(state, B),
		THCudaIntTensor_data(state, corrAB),
		patch, c, h, w
	);
	checkCudaError(L);

	luaT_pushudata(L, B, "torch.CudaTensor");
	return 1;
}
 

__global__ void blend_kernel(
	float *A, float *BP, float *M, float *AP,
	float alpha, int c, int h, int w 
)
{
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;
	if (_id < c * size) {
		// _id = dc * size + id
		int id = _id % size, dc = _id / size;
		// int x = id % w, y = id / w;
		float weight = M[id] < 0.05f ? 0.f : alpha;
		AP[dc * size + id] = 
			A[dc * size + id] * weight + 
			BP[dc * size + id] * (1.f - weight);
	}
	return ;
}

int blend(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor    *A       = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor    *BP      = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor    *M       = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	float            alpha   = luaL_checknumber(L, 4);

	THCudaTensor    *AP      = new_tensor_like(state, A);
	THCudaTensor_zero(state, AP);

	int c = THCudaTensor_size(state, A, 0);
	int h = THCudaTensor_size(state, A, 1);
	int w = THCudaTensor_size(state, A, 2);

	blend_kernel<<<(c*h*w-1)/TB+1, TB>>>(
		THCudaTensor_data(state, A),
		THCudaTensor_data(state, BP),
		THCudaTensor_data(state, M),
		THCudaTensor_data(state, AP),
		alpha, c, h, w
	);
	checkCudaError(L);

	luaT_pushudata(L, AP, "torch.CudaTensor");
	return 1;
}

__global__ void patchmatch2_conv_kernel(
	float *A, float *B, float *AP, float *BP, float *conv, 
	int *prev_corrAB_upsampled, int patch, int s_rad, 
	int c, int h, int w
)
{	
	int h1 = h, h2 = h, w1 = w, w2 = w;
	int _id = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h * w, size2 = h * w; 
	int s_size = 2 * s_rad + 1;
	int s_n = s_size * s_size; 
	if (_id < size1 * s_n) {
		conv[_id] = -1;

		int id1 = _id / s_n, s_idx = _id % s_n;
		int y1 = id1 / w1, x1 = id1 % w1;
		int dy2 = s_idx / s_size - s_rad, dx2 = s_idx % s_size - s_rad;

		int x2 = prev_corrAB_upsampled[2 * id1 + 0];
		int y2 = prev_corrAB_upsampled[2 * id1 + 1];

		int new_y2 = y2 + dy2;
		int new_x2 = x2 + dx2;
		if (!(new_x2 >= 0 && new_x2 < w2 && new_y2 >= 0 && new_y2 < h2)) {
			return ;
		}

		// Improve by local searching
		int kernel_radius = (patch - 1) / 2;
		float conv_result = 0;
		int cnt = 0;
		for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
			for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
				int xx1 = x1 + dx, yy1 = y1 + dy;
				int xx2 = new_x2 + dx, yy2 = new_y2 + dy;	
				if (0 <= xx1 && xx1 < w1 && 0 <= yy1 && yy1 < h1 &&
					0 <= xx2 && xx2 < w2 && 0 <= yy2 && yy2 < h2) 
				{
					int _id1 = yy1 * w1 + xx1, _id2 = yy2 * w2 + xx2;
					for (int dc = 0; dc < c; dc++) {
						float term1 = A[dc * size1 + _id1];  
						float term2 = B[dc * size2 + _id2];  
						conv_result += term1 * term2;

						term1 = AP[dc * size1 + _id1];  
						term2 = BP[dc * size2 + _id2];  
						conv_result += term1 * term2;
					}	
					cnt++;

				}
			}
		}

		conv[_id] = conv_result / cnt;
	}
	return ;
}

__global__ void patchmatch2_argmax_kernel(
	float *conv, int *prev_corrAB_upsampled, int *corrAB, int s_rad, 
	int c, int h, int w
)
{
	int h1 = h, h2 = h, w1 = w, w2 = w;
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1;//, size2 = h2 * w2;
	int s_size = 2 * s_rad + 1;
	int s_n = s_size * s_size;

	if (id1 < size1) {
		float conv_max = -1;

		// int y1 = id1 / w1, x1 = id1 % w1;

		int x2 = prev_corrAB_upsampled[2 * id1 + 0];
		int y2 = prev_corrAB_upsampled[2 * id1 + 1];

		for (int dx2 = -s_rad; dx2 <= s_rad; dx2++) {
			for (int dy2 = -s_rad; dy2 <= s_rad; dy2++) {
				int new_y2 = y2 + dy2;
				int new_x2 = x2 + dx2;

				if (new_x2 >= 0 && new_x2 < w2 && new_y2 >= 0 && new_y2 < h2) {
					int s_idx = (dy2 + s_rad) * s_size + (dx2 + s_rad);
					int id = id1 * s_n + s_idx;
					float conv_result = conv[id];
					if (conv_result > conv_max) {
						conv_max = conv_result;
						corrAB[id1 * 2 + 0] = new_x2;
						corrAB[id1 * 2 + 1] = new_y2;
					}
				}
			}
		}
	}

	return ;
}


int patchmatch2(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *A  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B  = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *AP = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *BP = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	int           patch  = luaL_checknumber(L, 5);
	int           s_rad  = luaL_checknumber(L, 6);
	THCudaIntTensor *prev_corrAB_upsampled = (THCudaIntTensor*)luaT_checkudata(L, 7, "torch.CudaIntTensor");

	int c = THCudaTensor_size(state, A, 0);
	int h = THCudaTensor_size(state, A, 1);
	int w = THCudaTensor_size(state, A, 2);

	THCudaIntTensor *corrAB = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, corrAB, h, w, 2);
	THCudaIntTensor_zero(state, corrAB);

	THCudaTensor *conv = THCudaTensor_new(state);
	THCudaTensor_resize2d(state, conv, h*w, (2*s_rad+1)*(2*s_rad+1));
	THCudaTensor_zero(state, conv);

	int N = h*w*(2*s_rad+1)*(2*s_rad+1);
	patchmatch2_conv_kernel<<<(N-1)/TB+1, TB>>>(
		THCudaTensor_data(state, A),
		THCudaTensor_data(state, B),
		THCudaTensor_data(state, AP),
		THCudaTensor_data(state, BP),
		THCudaTensor_data(state, conv),
		THCudaIntTensor_data(state, prev_corrAB_upsampled),
		patch, s_rad,
		c, h, w
	);
	checkCudaError(L);

	patchmatch2_argmax_kernel<<<(h*w-1)/TB+1, TB>>>(
		THCudaTensor_data(state, conv),
		THCudaIntTensor_data(state, prev_corrAB_upsampled),
		THCudaIntTensor_data(state, corrAB),
		s_rad, c, h, w
	);
	checkCudaError(L);
 		
 	THCudaTensor_free(state, conv);

 	luaT_pushudata(L, corrAB, "torch.CudaIntTensor");
 	return 1;
}

__global__ void upsample_corr_kernel(
	int *curr_corr, int *next_corr, 
	int curr_h, int curr_w, int next_h, int next_w
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < next_h * next_w) {
		int next_x = id % next_w, next_y = id / next_w;

		float w_ratio = (float)next_w / (float)curr_w;
		float h_ratio = (float)next_h / (float)curr_h;

		int curr_x = (next_x + 0.5) / w_ratio;
		int curr_y = (next_y + 0.5) / h_ratio;

		curr_x = MAX(MIN(curr_x, curr_w-1), 0);
		curr_y = MAX(MIN(curr_y, curr_h-1), 0);

		int curr_id = curr_y * curr_w + curr_x;
		
		int curr_x2 = curr_corr[2 * curr_id + 0];
		int curr_y2 = curr_corr[2 * curr_id + 1];

		int next_x2 = next_x + (curr_x2 - curr_x) * w_ratio + 0.5;
		int next_y2 = next_y + (curr_y2 - curr_y) * h_ratio + 0.5;

		next_x2 = MAX(MIN(next_x2, next_w-1), 0);
		next_y2 = MAX(MIN(next_y2, next_h-1), 0);

		next_corr[2 * id + 0] = next_x2;
		next_corr[2 * id + 1] = next_y2;
	}

	return ;
}

int upsample_corr(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaIntTensor *curr_corrAB = (THCudaIntTensor*)luaT_checkudata(L, 1, "torch.CudaIntTensor");
	int              next_h      = luaL_checknumber(L, 2);
	int              next_w      = luaL_checknumber(L, 3);
	THCudaIntTensor *next_corrAB = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, next_corrAB, next_h, next_w, 2);
	THCudaIntTensor_zero(state, next_corrAB);

	int              curr_h      = THCudaIntTensor_size(state, curr_corrAB, 0);
	int              curr_w      = THCudaIntTensor_size(state, curr_corrAB, 1);

	upsample_corr_kernel<<<(next_h*next_w-1)/TB+1, TB>>>(
		THCudaIntTensor_data(state, curr_corrAB),
		THCudaIntTensor_data(state, next_corrAB),
		curr_h, curr_w, next_h, next_w
	);
	checkCudaError(L);

	luaT_pushudata(L, next_corrAB, "torch.CudaIntTensor");
	return 1;
}

__host__ __device__ float dist(float *A, float *B, float *AP, float *BP, 
	int x1, int y1, int x2, int y2, int c1, int h1, int w1, int h2, int w2, int patch_w) 
{
	int size1 = h1 * w1;
	int size2 = h2 * w2;
	float conv_result = 0;
	int cnt = 0;
	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {

			if (
				(y1 + dy) < h1 && (y1 + dy) >= 0 && (x1 + dx) < w1 && (x1 + dx) >= 0
				&&
				(y2 + dy) < h2 && (y2 + dy) >= 0 && (x2 + dx) < w2 && (x2 + dx) >= 0
				)
			{
				int _id1 = (y1 + dy) * w1 + (x1 + dx);
				int _id2 = (y2 + dy) * w2 + (x2 + dx);

				for (int c = 0; c < c1; c++) {
					float term1 = A[c * size1 + _id1];  
					float term2 = B[c * size2 + _id2];  
					conv_result += term1 * term2;

					term1 = AP[c * size1 + _id1];  
					term2 = BP[c * size2 + _id2];  
					conv_result += term1 * term2;
				}
				cnt++;
			}
		}
	}

	float d = conv_result / cnt;

	return d;
}

__device__ void improve_guess(float *A, float *B, float *AP, float *BP, 
	int x1, int y1, int x2, int y2, int c1, int h1, int w1, int h2, int w2, int patch_w, 
	int &xbest, int &ybest, float &dbest, float rr = 0.f)
{
	float d = dist(A, B, AP, BP, x1, y1, x2, y2, c1, h1, w1, h2, w2, patch_w);
	if (d > dbest + rr) { // note: normalized cross-correlation
		dbest = d;
		xbest = x2;
		ybest = y2;
	}	
}

#define ITERS 10
__global__ void PatchMatch_global_kernel(
	float *A, float *B, float *AP, float *BP, int *corr, int *prev_corr_upsampled,
	int patch, int rs_max, int c1, int h1, int w1, int h2, int w2
)
{
	int x1 = blockIdx.x*blockDim.x + threadIdx.x;
	int y1 = blockIdx.y*blockDim.y + threadIdx.y;
	int id1 = y1 * w1 + x1; 

	if (x1 < w1 && y1 < h1) {
		curandState state;
		InitcuRand(state);

		float dbest;
		int xbest, ybest;


		xbest = prev_corr_upsampled[2 * id1 + 0]; //static_cast<int>(MycuRand(state) * w2);
		ybest = prev_corr_upsampled[2 * id1 + 1]; //static_cast<int>(MycuRand(state) * h2);
		dbest = dist(A, B, AP, BP, x1, y1, xbest, ybest, c1, h1, w1, h2, w2, patch);
		corr[2 * id1 + 0] = xbest;
		corr[2 * id1 + 1] = ybest;

		for (int it = 0; it < ITERS; it++) 
		{	
			// Current best guess 
			xbest = corr[2 * id1 + 0];  
			ybest = corr[2 * id1 + 1];  
			dbest = dist(A, B, AP, BP, x1, y1, xbest, ybest, c1, h1, w1, h2, w2, patch);

			// Propagation
			for (int jump = 8; jump > 0; jump /= 2) {
				if (x1 - jump >= 0) { // left 
					int _id1 = y1 * w1 + x1 - jump;
					int xp = corr[2 * _id1 + 0] + jump;
					int yp = corr[2 * _id1 + 1];
					if (xp >= 0 && xp < w2 && yp >= 0 && yp < h2) {
						improve_guess(A, B, AP, BP, x1, y1, xp, yp, c1, h1, w1, h2, w2, patch, xbest, ybest, dbest);
						corr[2 * id1 + 0] = xbest;
						corr[2 * id1 + 1] = ybest;
					}
				}
				if (x1 + jump < w1) { // right 
					int _id1 = y1 * w1 + x1 + jump;
					int xp = corr[2 * _id1 + 0] - jump;
					int yp = corr[2 * _id1 + 1];
					if (xp >= 0 && xp < w2 && yp >= 0 && yp < h2) {
						improve_guess(A, B, AP, BP, x1, y1, xp, yp, c1, h1, w1, h2, w2, patch, xbest, ybest, dbest);
						corr[2 * id1 + 0] = xbest;
						corr[2 * id1 + 1] = ybest;
					}
				}
				if (y1 - jump >= 0) { // up 
					int _id1 = (y1 - jump) * w1 + x1;
					int xp = corr[2 * _id1 + 0];
					int yp = corr[2 * _id1 + 1] + jump;
					if (xp >= 0 && xp < w2 && yp >= 0 && yp < h2) {
						improve_guess(A, B, AP, BP, x1, y1, xp, yp, c1, h1, w1, h2, w2, patch, xbest, ybest, dbest);
						corr[2 * id1 + 0] = xbest;
						corr[2 * id1 + 1] = ybest;
					}
				}
				if (y1 + jump < h1) { // down 
					int _id1 = (y1 + jump) * w1 + x1;
					int xp = corr[2 * _id1 + 0];
					int yp = corr[2 * _id1 + 1] - jump;
					if (xp >= 0 && xp < w2 && yp >= 0 && yp < h2) {
						improve_guess(A, B, AP, BP, x1, y1, xp, yp, c1, h1, w1, h2, w2, patch, xbest, ybest, dbest);
						corr[2 * id1 + 0] = xbest;
						corr[2 * id1 + 1] = ybest;
					}
				}
			}

			// Random Search
			int rs = rs_max;
			if (rs > cuMax(h2, w2)) {
				rs = cuMax(h2, w2);
			}
			for (int mag = rs; mag >= 1; mag /= 2) {
				int xmin = cuMax(xbest - mag, 0),
				    xmax = cuMin(xbest + mag + 1, w2);
				int ymin = cuMax(ybest - mag, 0),
				    ymax = cuMin(ybest + mag + 1, h2);
				int xp   = xmin + (int)(MycuRand(state)*(xmax - xmin)) % (xmax - xmin);
				int yp   = ymin + (int)(MycuRand(state)*(ymax - ymin)) % (ymax - ymin);
				improve_guess(A, B, AP, BP, x1, y1, xp, yp, c1, h1, w1, h2, w2, patch, xbest, ybest, dbest, FLT_MIN);
				corr[2 * id1 + 0] = xbest;
				corr[2 * id1 + 1] = ybest;
			}

			__syncthreads();

		}
	}

	return ;
}

int PatchMatch(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *A  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *B  = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *AP = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *BP = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	int           patch  = luaL_checknumber(L, 5);
	int           s_rad  = luaL_checknumber(L, 6);
	THCudaIntTensor *prev_corr_upsampled = (THCudaIntTensor*)luaT_checkudata(L, 7, "torch.CudaIntTensor");

	int c1 = THCudaTensor_size(state, A, 0);
	int h1 = THCudaTensor_size(state, A, 1);
	int w1 = THCudaTensor_size(state, A, 2);

	int c2 = THCudaTensor_size(state, BP, 0);
	int h2 = THCudaTensor_size(state, BP, 1);
	int w2 = THCudaTensor_size(state, BP, 2);

	THCudaIntTensor *corr = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, corr, h1, w1, 2);
	THCudaIntTensor_zero(state, corr);

	// Generalized PatchMatch
	assert(c1 == c2);
	dim3 blocksPerGrid(w1 / 20 + 1, h1 / 20 + 1, 1);
	dim3 threadsPerBlock(20, 20, 1);

	PatchMatch_global_kernel<<<blocksPerGrid, threadsPerBlock>>>(
		THCudaTensor_data(state, A),
		THCudaTensor_data(state, B),
		THCudaTensor_data(state, AP),
		THCudaTensor_data(state, BP),
		THCudaIntTensor_data(state, corr),
		THCudaIntTensor_data(state, prev_corr_upsampled),
		patch, s_rad, 
		c1,
		h1, w1,
		h2, w2
	);
	checkCudaError(L);

	luaT_pushudata(L, corr, "torch.CudaIntTensor");
	return 1;
}

__global__ void Ring_kernel(
	float *A, float *BP, int *corrAB, float *M, 
	int ring, int c, int h, int w
)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;
	int ringSize  = 2*ring + 1;
	int ringPatch = ringSize * ringSize;
	if (id1 < size) {
		int y1 = id1 / w, x1 = id1 % w;
		int y2 = corrAB[2 * id1 + 1], x2 = corrAB[2 * id1 + 0];
		// int id2 = y2 * w + x2;

		for (int dx = -ring; dx <= ring; dx++) 
			for (int dy = -ring; dy <= ring; dy++)
			{
				int pIdx = (dy + ring) * ringSize + (dx + ring);
				int _x2 = x2 + dx, _y2 = y2 + dy;
				if (_x2 >= 0 && _x2 < w && _y2 >= 0 && _y2 < h)
				{
					for (int dc = 0; dc < c; dc++) {
						// M[(dc * size + y1 * w + x1) * ringPatch + pIdx] = 
						M[(dc * size + y1 * w) * ringPatch + pIdx * w + x1] = 
							BP[dc * size + _y2 * w + _x2];
					}
				}
			}

	}

	return ;
}

int Ring(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *A         = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *BP        = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaIntTensor *corrAB = (THCudaIntTensor*)luaT_checkudata(L, 3, "torch.CudaIntTensor");
	int ring                = luaL_checknumber(L, 4);

	int c = THCudaTensor_size(state, A, 0);
	int h = THCudaTensor_size(state, A, 1);
	int w = THCudaTensor_size(state, A, 2);
 
	THCudaTensor *M = THCudaTensor_new(state);
	THCudaTensor_resize3d(state, M, c, h, w*(2*ring+1)*(2*ring+1));
 	THCudaTensor_zero(state, M); 

	Ring_kernel<<<(c*h*w-1)/TB+1, TB>>>(
		THCudaTensor_data(state, A),
		THCudaTensor_data(state, BP),
		THCudaIntTensor_data(state, corrAB),
		THCudaTensor_data(state, M),
		ring, c, h, w
	);
	checkCudaError(L);

	luaT_pushudata(L, M, "torch.CudaTensor");
	return 1;
}

void Ring_cpu_kernel(
	float *A, float *BP, int *corrAB, float *M, 
	int ring, int c, int h, int w
)
{
	int ringSize  = 2*ring + 1;
	int ringPatch = ringSize * ringSize;
	int size      = h * w;

#pragma omp parallel for 
	for (int y1 = 0; y1 < h; y1++)
		for (int x1 = 0; x1 < w; x1++)
		{
			int id = y1 * w + x1;
			int x2 = corrAB[2*id + 0];
			int y2 = corrAB[2*id + 1];
			for (int dx = -ring; dx <= ring; dx++) 
				for (int dy = -ring; dy <= ring; dy++)
				{
					int pIdx = (dy + ring) * ringSize + (dx + ring);
					int _x2 = x2 + dx, _y2 = y2 + dy;
					if (_x2 >= 0 && _x2 < w && _y2 >= 0 && _y2 < h)
					{
						for (int dc = 0; dc < c; dc++) {
							M[(dc * size + y1 * w) * ringPatch + pIdx * w + x1] = 
								BP[dc * size + _y2 * w + _x2];
						}
					}
				}	

		}

	return ;
}

int Ring_cpu(lua_State *L) {
	THFloatTensor *A    = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
	THFloatTensor *BP   = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
	THIntTensor *corrAB = (THIntTensor*)luaT_checkudata(L, 3, "torch.IntTensor");
	int ring            = luaL_checknumber(L, 4);

	int c = THFloatTensor_size(A, 0);
	int h = THFloatTensor_size(A, 1);
	int w = THFloatTensor_size(A, 2);
 
	THFloatTensor *M = THFloatTensor_new();
	THFloatTensor_resize3d(M, c, h, w*(2*ring+1)*(2*ring+1));
 	THFloatTensor_zero(M); 

	Ring_cpu_kernel(
		THFloatTensor_data(A),
		THFloatTensor_data(BP),
		THIntTensor_data(corrAB),
		THFloatTensor_data(M),
		ring, c, h, w
	);

	luaT_pushudata(L, M, "torch.FloatTensor");
	return 1;
}

__global__ void Ring2_kernel(
	float *A, float *BP, int *corrAB, int *mask, int *m, 
	int ring, int c, int h, int w
)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;
	if (id1 < size) {
		// int y1 = id1 / w, x1 = id1 % w;
		if (mask[id1] != 0) {

			int y2 = corrAB[2 * id1 + 1], x2 = corrAB[2 * id1 + 0];
			for (int dx = -ring; dx <= ring; dx++) 
				for (int dy = -ring; dy <= ring; dy++)
				{
					int _x2 = x2 + dx, _y2 = y2 + dy;
					if (_x2 >= 0 && _x2 < w && _y2 >= 0 && _y2 < h)
					{
						m[_y2 * w + _x2] = 1;
					}
				}
		}
	}

	return ;
}

int Ring2(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *A         = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *BP        = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaIntTensor *corrAB = (THCudaIntTensor*)luaT_checkudata(L, 3, "torch.CudaIntTensor");
	int ring                = luaL_checknumber(L, 4);
	THCudaIntTensor *mask   = (THCudaIntTensor*)luaT_checkudata(L, 5, "torch.CudaIntTensor");

	int c = THCudaTensor_size(state, A, 0);
	int h = THCudaTensor_size(state, A, 1);
	int w = THCudaTensor_size(state, A, 2);
 
	THCudaIntTensor *m = THCudaIntTensor_new(state);
	THCudaIntTensor_resize2d(state, m, h, w);
	THCudaIntTensor_zero(state, m);

	Ring2_kernel<<<(h*w-1)/TB+1, TB>>>(
		THCudaTensor_data(state, A),
		THCudaTensor_data(state, BP),
		THCudaIntTensor_data(state, corrAB),
		THCudaIntTensor_data(state, mask),
		THCudaIntTensor_data(state, m),
		ring, c, h, w
	);
	checkCudaError(L);

	luaT_pushudata(L, m, "torch.CudaIntTensor");
	return 1;
}


__global__ void patchmatch_r_conv_kernel(
	float *input, float *target, float *conv, 
	int patch, int stride,  
	int c1, int h1, int w1, int h2, int w2
)
{	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;
	int N = size1 * size2;
	// id = id1 * size2 + id2

	if (id < N) {
		int id1 = id / size2, id2 = id % size2;

		int x1 = id1 % w1, y1 = id1 / w1;
		int x2 = id2 % w2, y2 = id2 / w2;

		int kernel_radius = (patch - 1) / 2;

		double conv_result = 0, norm_1 = 0, norm_2 = 0;
		for (int dy = -kernel_radius; dy <= kernel_radius; dy+=stride) {
			for (int dx = -kernel_radius; dx <= kernel_radius; dx+=stride) {
				int xx1 = x1 + dx, yy1 = y1 + dy;
				int xx2 = x2 + dx, yy2 = y2 + dy;	
				if (0 <= xx1 && xx1 < w1 && 0 <= yy1 && yy1 < h1 &&
					0 <= xx2 && xx2 < w2 && 0 <= yy2 && yy2 < h2) 
				{	
					int _id1 = yy1 * w1 + xx1, _id2 = yy2 * w2 + xx2;
					for (int c = 0; c < c1; c++) {
						float term1 = input[c * size1 + _id1];  
						float term2 = target[c * size2 + _id2];  
						conv_result += term1 * term2;
						norm_1      += term1 * term1;
						norm_2      += term2 * term2;
					}	

				}
			}	
		}

		norm_1 = sqrt(norm_1);
		norm_2 = sqrt(norm_2);

		conv[id] = conv_result / (norm_1 * norm_2 + 1e-9);		
	}

	return ;
}

__global__ void patchmatch_r_argmax_kernel(
	float *conv, float *target, float *match, int *correspondence,
	int c1, int h1, int w1, int h2, int w2
) 
{	
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = h1 * w1, size2 = h2 * w2;
	
	if (id1 < size1) {
		//int x1 = id1 % w1, y1 = id1 / w1;
		double conv_max = -1e20;

		for (int y2 = 0; y2 < h2; y2++) {
			for (int x2 = 0; x2 < w2; x2++) {
				int id2 = y2 * w2 + x2;

				int id = id1 * size2 + id2;
				float conv_result = conv[id];

				if (conv_result > conv_max) {
					conv_max = conv_result;
					correspondence[id1 * 2 + 0] = x2;
					correspondence[id1 * 2 + 1] = y2;
					for (int c = 0; c < c1; c++) {
						match[c * size1 + id1] = target[c * size2 + id2];
					}
				}
			}
		}

	}

	return ;
}

int patchmatch_r(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *input  = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int           patch  = luaL_checknumber(L, 3);
	int 		  stride = luaL_checknumber(L, 4);

	int c1 = THCudaTensor_size(state, input, 0);
	int h1 = THCudaTensor_size(state, input, 1);
	int w1 = THCudaTensor_size(state, input, 2);

	int c2 = THCudaTensor_size(state, target, 0);
	int h2 = THCudaTensor_size(state, target, 1);
	int w2 = THCudaTensor_size(state, target, 2);

	THCudaTensor *conv = THCudaTensor_new(state);
	THCudaTensor_resize2d(state, conv, h1*w1, h2*w2);
	THCudaTensor_zero(state, conv);
	
	assert(c1 == c2);
	int N = h1*w1*h2*w2;
	patchmatch_r_conv_kernel<<<(N-1)/TB+1, TB>>>(
		THCudaTensor_data(state, input),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, conv),
		patch, stride, 
		c1,
		h1, w1,
		h2, w2
	);
	checkCudaError(L);


	THCudaTensor *match = new_tensor_like(state, input);
	THCudaTensor_zero(state, match);

	THCudaIntTensor *correspondence = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, correspondence, h1, w1, 2);
	THCudaIntTensor_zero(state, correspondence);

	
	patchmatch_r_argmax_kernel<<<(h1*w1-1)/TB+1, TB>>>(
		THCudaTensor_data(state, conv),
		THCudaTensor_data(state, target),
		THCudaTensor_data(state, match),
		THCudaIntTensor_data(state, correspondence),
		c1,
		h1, w1,
		h2, w2		
	);
	checkCudaError(L);	

	THCudaTensor_free(state, conv);

	luaT_pushudata(L, match, "torch.CudaTensor");
	luaT_pushudata(L, correspondence, "torch.CudaIntTensor");
	return 2;
}

__global__ void refineNNF_kernel(
	float *N_A, float *N_BP,
	int *init_corr, float *guide, 
	int *tmask, int *corr, 
	int patch, int c, int h, int w
)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	int size = h * w;
	int r = (patch - 1) / 2;
	if (id1 < size) {
		int x1 = id1 % w, y1 = id1 / w;
		int x2 = init_corr[2*id1 + 0];
		int y2 = init_corr[2*id1 + 1];

		corr[2*id1 + 0] = x2;
		corr[2*id1 + 1] = y2;

		if (tmask[id1] < EPS)
			return ;
		
		float best_d = FLT_MAX;
		int best_x2 = x2, best_y2 = y2;

		for (int dx = -r; dx <= r; dx++)
		for (int dy = -r; dy <= r; dy++)
		{
			int new_x1 = x1 + dx;
			int new_y1 = y1 + dy;
			int new_id1 = new_y1 * w + new_x1;
			if (new_x1 >= 0 && new_x1 < w && new_y1 >= 0 && new_y1 < h) {
				int new_x2 = init_corr[2*new_id1 + 0] - dx;
				int new_y2 = init_corr[2*new_id1 + 1] - dy;
				int new_id2 = new_y2 * w + new_x2;
				if (new_x2 >= r && new_x2 < w - r - 1 && new_y2 >= r && new_y2 < h - r - 1) {
					
					float dist = 0; 
					int cnt = 0;

					for (int _dx = -r; _dx <= r; _dx++)
					for (int _dy = -r; _dy <= r; _dy++)
					{
						int _new_x1 = x1 + _dx;
						int _new_y1 = y1 + _dy;
						int _new_id1 = _new_y1 * w + _new_x1;
						if (_new_x1 >= 0 && _new_x1 < w && _new_y1 >= 0 && _new_y1 < h) {
							int _new_x2 = init_corr[2*_new_id1 + 0] - _dx;
							int _new_y2 = init_corr[2*_new_id1 + 1] - _dy;
							int _new_id2 = _new_y2 * w + _new_x2;
							if (_new_x2 >= 0 && _new_x2 < w && _new_y2 >= 0 && _new_y2 < h) {
								float d = 0;
								for (int dc = 0; dc < 3; dc++) {
									float diff = guide[dc * size + new_id2] - guide[dc * size + _new_id2];
									d += diff * diff;
								}
								d = sqrt(d);
								dist += d;
								cnt++;
							}
						}
					}

					dist = dist / cnt;

					if (dist < best_d) {
						best_d = dist;
						best_x2 = new_x2;
						best_y2 = new_y2;
					}					


				}
			}
		}

		corr[2*id1 + 0] = best_x2;
		corr[2*id1 + 1] = best_y2;

	}
	return ;
}

int refineNNF(lua_State *L) {
	THCState        *state     = getCutorchState(L);
	THCudaTensor    *N_A       = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor    *N_BP      = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaIntTensor *init_corr = (THCudaIntTensor*)luaT_checkudata(L, 3, "torch.CudaIntTensor");
	THCudaTensor    *guide     = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaIntTensor *tmask     = (THCudaIntTensor*)luaT_checkudata(L, 5, "torch.CudaIntTensor");
	int              patch     = luaL_checknumber(L, 6);
	int              niter     = luaL_checknumber(L, 7);

	int c = THCudaTensor_size(state, N_BP, 0);
	int h = THCudaTensor_size(state, N_BP, 1);
	int w = THCudaTensor_size(state, N_BP, 2);
 
	THCudaIntTensor *corr = THCudaIntTensor_new(state);
	THCudaIntTensor_resize3d(state, corr, h, w, 2);
	THCudaIntTensor_zero(state, corr);

	for (int iter = 0; iter < niter; iter++) {
		printf("  iter=%d\n", iter);
		refineNNF_kernel<<<(h*w-1)/TB+1, TB>>>(
			THCudaTensor_data   (state, N_A),
			THCudaTensor_data   (state, N_BP),
			THCudaIntTensor_data(state, init_corr),
			THCudaTensor_data   (state, guide),
			THCudaIntTensor_data(state, tmask),
			THCudaIntTensor_data(state, corr),
			patch, c, h, w
		);
		checkCudaError(L);
		cudaMemcpy(
			THCudaIntTensor_data(state, init_corr),
			THCudaIntTensor_data(state, corr),
			sizeof(int) * h * w * 2,
			cudaMemcpyDeviceToDevice
		);
	}

	
		 
	luaT_pushudata(L, corr, "torch.CudaIntTensor");
	return 1;
}

static const struct luaL_Reg funcs[] = {
	{"histogram"    , histogram},     // compute histogram
	{"histogram_cpu", histogram_cpu}, // compute histogram on cpu
	{"hist_remap2"  , hist_remap2},   // histogram remapping
	{"patchmatch"   , patchmatch},    // brute force
	{"patchmatch_r" , patchmatch_r},  // raw
	{"conv"         , conv},          // brute force conv
	{"avg_vote"     , avg_vote},      // avg reconstruction features
	{"blend"        , blend},         // blend feature maps
	{"patchmatch2"  , patchmatch2},   // patch match locally
	{"upsample_corr", upsample_corr}, // upsample correspondence
	{"PatchMatch"   , PatchMatch},    // PatchMatch algorithm
	{"Ring"         , Ring},          // Spatial neural patch for more texture
	{"Ring_cpu"     , Ring_cpu},      // ... cpu
	{"Ring2"        , Ring2},         // draw on BP instead of A (no many-to-one since used once)
	{"refineNNF"    , refineNNF},     // NNF spatial consistency 
	{NULL, NULL}
};

extern "C" int luaopen_libcuda_utils(lua_State *L) {
	luaL_openlib(L, "cuda_utils", funcs, 0);
	return 1;
}