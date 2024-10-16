#include "Func.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <chrono>         //header to calculate time
#include <iostream>
/////////////////////////////////////////////////////////////////////////
// 1. 함수는 Colab 환경에서 동작해야 합니다.
// 2. 자유롭게 구현하셔도 되지만 모든 함수에서 GPU를 활용해야 합니다.
// 3. CPU_Func.cu에 있는 Image_Check함수에서 True가 Return되어야 하며, CPU코드에 비해 속도가 빨라야 합니다.
/////////////////////////////////////////////////////////////////////////

__global__ void g_scale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) {
	
	unsigned int Xvalue = (blockIdx.x * blockDim.x) + threadIdx.x * 3;
	if (Xvalue >= 54 && len>Xvalue && (Xvalue %3==0)) {
		int tmp = (buf[Xvalue] * 0.114 + buf[Xvalue + 1] * 0.587 + buf[Xvalue + 2] * 0.299);
		gray[Xvalue] = tmp;
		gray[Xvalue + 1] = tmp;
		gray[Xvalue + 2] = tmp;
	}
}

void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len) {

	dim3 dimGrid2(49, 49, 1);
	dim3 dimBlock2(32, 32, 1);

	uint8_t * GPU_buf = NULL;
	uint8_t * GPU_gray = NULL;

	cudaMalloc((void**)&GPU_buf, (len+2) * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_gray, (len+2) * sizeof(uint8_t));

	cudaMemcpy(GPU_buf, buf, (len+2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_gray, gray, (len+2) * sizeof(uint8_t), cudaMemcpyHostToDevice);

	g_scale <<< dimGrid2, dimBlock2 >>> (GPU_buf, GPU_gray, start_add, len);
	cudaMemcpy(gray, GPU_gray, (len+2) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	
	cudaError_t err;
	err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
	}

	cudaFree(GPU_gray);
	cudaFree(GPU_buf);

}

__constant__ float filter_con[25];

__device__ float GPU_conv2d_5x5(float* filter, uint8_t* pixel, int x, int y, int width) {
	float v = 0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			v += pixel[(y + i) * width + x + j] * filter[i * 5 + j];
		}
	}
	return v;
}


__global__ void mulKernel(int width, int height, uint8_t* gray, uint8_t* gaussian, uint8_t * tmp) {
	float sigma = 1.0;

	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			filter_con[(i + 2) * 5 + j + 2]
				= (1 / (2 * 3.14 * sigma * sigma)) * exp(-(i * i + j * j) / (2 * sigma * sigma));
		}
	}

	int Xvalue = (blockIdx.x * blockDim.x) + threadIdx.x;
	int Xvalue2 = (blockIdx.x * blockDim.x) + threadIdx.x + 2;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int j_2 = (blockIdx.y * blockDim.y) + threadIdx.y + 2;

	if (Xvalue < height && j_2 < width+2) {
		tmp[Xvalue2 * (width + 4) + j_2] = gray[((Xvalue2 - 2) * width + (j_2 - 2)) * 3];
	}
	__syncthreads();
	if (Xvalue < height && j<width) {
		uint8_t v = GPU_conv2d_5x5(filter_con, tmp, j, Xvalue, width + 4);
		gaussian[(Xvalue * width + j) * 3] = v;
		gaussian[(Xvalue * width + j) * 3 + 1] = v;
		gaussian[(Xvalue * width + j) * 3 + 2] = v;
	}
}


void GPU_Noise_Reduction(int width, int height, uint8_t *gray, uint8_t *gaussian) {

	printf("before gaussian[54] %d\n", gray[54]);
	printf("%d, %d\n", width, height);
	
	uint8_t* GPU_gray = NULL;
	cudaMalloc((void**)&GPU_gray, 3* (width) * (height) * sizeof(uint8_t));
	cudaMemcpy(GPU_gray, gray, 3* (width) * (height) * sizeof(uint8_t), cudaMemcpyHostToDevice);

	//uint8_t* tmp = (uint8_t*)malloc((width + 4) * (height + 4));
	//memset(tmp, (uint8_t)0, (width + 4) * (height + 4));
	uint8_t* GPU_tmp = NULL;
	cudaMalloc((void**)&GPU_tmp, (width + 4) * (height + 4) * sizeof(uint8_t));
	cudaMemset(GPU_tmp, (uint8_t)0, (width + 4) * (height + 4) * sizeof(uint8_t));

	uint8_t* GPU_gaussian = NULL;
	cudaMalloc((void**)&GPU_gaussian, 3* (width + 4) * (height + 4) * sizeof(uint8_t));

	dim3 dimGrid2(49, 49, 1);
	dim3 dimBlock2(32, 32, 1);

	mulKernel << < dimGrid2, dimBlock2 >> > (width, height, GPU_gray, GPU_gaussian, GPU_tmp);	//GPU TMP CAREFUL
	cudaMemcpy(gaussian, GPU_gaussian, 3* (width) * (height) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	printf("after gaussian[54] %d\n", gaussian[54]);
	//GaussianBlur
	cudaError_t err;
	err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
	}

	cudaFree(GPU_gray);
	cudaFree(GPU_gaussian);
	cudaFree(GPU_tmp);

}

__device__ void GPU_conv2d_3x3(uint8_t* pixel, int x, int y, int width, int& gx, int& gy) {
	//int gx = 0;
	//int gy = 0;
	int filter_x[9] = { -1,0,1
						,-2,0,2
						,-1,0,1 };
	int filter_y[9] = { 1,2,1
						,0,0,0
						,-1,-2,-1 };

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			gy += (int)pixel[(y + i) * width + x + j] * filter_y[i * 3 + j];
			gx += (int)pixel[(y + i) * width + x + j] * filter_x[i * 3 + j];
		}
	}
}

__global__ void GPU_Inten(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t* angle, uint8_t * tmp){
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i_2 = (blockIdx.x * blockDim.x) + threadIdx.x + 1;

	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int j_2 = (blockIdx.y * blockDim.y) + threadIdx.y+1;


	if (i < height && j_2<width+1) {
		tmp[i_2 * (width + 2) + j_2] = gaussian[((i_2 - 1) * width + (j_2 - 1)) * 3];
	}
	__syncthreads();
	
	if (i < height && j< width) {
		int gx = 0;
		int gy = 0;
		GPU_conv2d_3x3(tmp, j, i, width + 2, gx, gy);
		double add = gx * gx + gy * gy;
		int n = (float)sqrt((double)add);

		uint8_t  v = 0;
		if (n > 255) {
			v = 255;
		}
		else {
			v = n;
		}
		//v = 255;
		/*sobel[(i * width + j) * 3] = v;
		sobel[(i * width + j) * 3 + 1] = v;
		sobel[(i * width + j) * 3 + 2] = v;
		*/
		sobel[(i * width + j) * 3] = v;
		sobel[(i * width + j) * 3 + 1] = v;
		sobel[(i * width + j) * 3 + 2] = v;

		float t_angle = 0;
		if (gy != 0 || gx != 0)
			t_angle = (float)atan2((double)gy, (double)gx) * 180.0 / 3.14;
		//t_angle = (double)atan2(gy, gx) * 180.0 / 3.14;
		if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
			angle[i * width + j] = 0;
		else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
			angle[i * width + j] = 45;
		else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
			angle[i * width + j] = 90;
		else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
			angle[i * width + j] = 135;	
	}
	
}

void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t*angle){
	uint8_t* GPU_tmp = NULL;
	uint8_t* GPU_gaussian = NULL;
	uint8_t* GPU_sobel = NULL;
	uint8_t* GPU_angle = NULL;

	cudaMalloc((void**)&GPU_tmp, (width + 2) * (height + 2) * sizeof(uint8_t));
	cudaMemset(GPU_tmp, (uint8_t)0, (width + 2) * (height + 2) * sizeof(uint8_t));

	cudaMalloc((void**)&GPU_gaussian, (width + 2) * (height + 2) * 3 *sizeof(uint8_t));
	cudaMalloc((void**)&GPU_sobel, (width + 2) * (height + 2) * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_angle, (width + 2) * (height + 2) * sizeof(uint8_t));

	dim3 dimGrid2(2347, 1, 1);
	dim3 dimBlock2(1024, 1, 1);

	cudaMemcpy(GPU_sobel, sobel, (width + 2) * (height + 2) * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_gaussian, gaussian, (width + 2) * (height + 2) * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

	GPU_Inten << < dimGrid2, dimBlock2 >> > (width, height, GPU_gaussian, GPU_sobel, GPU_angle, GPU_tmp);	//GPU TMP CAREFUL

	cudaMemcpy(sobel, GPU_sobel, (width) * (height) * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(angle, GPU_angle, (width) * (height) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	//zero padding
	cudaError_t err;
	err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
	}

	cudaFree(GPU_tmp);
	cudaFree(GPU_gaussian);
	cudaFree(GPU_sobel);
	cudaFree(GPU_angle);

}
__global__ void GPU_max2(int width, int height, uint8_t* angle, uint8_t* sobel, uint8_t* suppression_pixel, uint8_t* chk) {
	uint8_t p1 = 0;
	uint8_t p2 = 0;
	int i = (blockIdx.x * blockDim.x) + threadIdx.x + 1;

	int j = (blockIdx.y * blockDim.y) + threadIdx.y + 1;

	if (i<(height-1) && j < width - 1) {
		if (angle[i * width + j] == 0) {
			p1 = sobel[((i + 1) * width + j) * 3];
			p2 = sobel[((i - 1) * width + j) * 3];
		}
		else if (angle[i * width + j] == 45) {
			p1 = sobel[((i + 1) * width + j - 1) * 3];
			p2 = sobel[((i - 1) * width + j + 1) * 3];
		}
		else if (angle[i * width + j] == 90) {
			p1 = sobel[((i)*width + j + 1) * 3];
			p2 = sobel[((i)*width + j - 1) * 3];
		}
		else {
			p1 = sobel[((i + 1) * width + j + 1) * 3];
			p2 = sobel[((i - 1) * width + j - 1) * 3];
		}

		uint8_t v = sobel[(i * width + j) * 3];

		if (chk[0] > v)
			chk[0] = v;
		if (chk[1] < v)
			chk[1] = v;

		if ((v >= p1) && (v >= p2)) {
			suppression_pixel[(i * width + j) * 3] = v;
			suppression_pixel[(i * width + j) * 3 + 1] = v;
			suppression_pixel[(i * width + j) * 3 + 2] = v;
		}
		else {
			suppression_pixel[(i * width + j) * 3] = 0;
			suppression_pixel[(i * width + j) * 3 + 1] = 0;
			suppression_pixel[(i * width + j) * 3 + 2] = 0;
		}
	}
	/*
	if (i < (height - 1)) {
		for (int j = 1; j < width - 1; j++) {

			if (angle[i * width + j] == 0) {
				p1 = sobel[((i + 1) * width + j) * 3];
				p2 = sobel[((i - 1) * width + j) * 3];
			}
			else if (angle[i * width + j] == 45) {
				p1 = sobel[((i + 1) * width + j - 1) * 3];
				p2 = sobel[((i - 1) * width + j + 1) * 3];
			}
			else if (angle[i * width + j] == 90) {
				p1 = sobel[((i)*width + j + 1) * 3];
				p2 = sobel[((i)*width + j - 1) * 3];
			}
			else {
				p1 = sobel[((i + 1) * width + j + 1) * 3];
				p2 = sobel[((i - 1) * width + j - 1) * 3];
			}

			uint8_t v = sobel[(i * width + j) * 3];

			if (chk[0] > v)
				chk[0] = v;
			if (chk[1] < v)
				chk[1] = v;

			if ((v >= p1) && (v >= p2)) {
				suppression_pixel[(i * width + j) * 3] = v;
				suppression_pixel[(i * width + j) * 3 + 1] = v;
				suppression_pixel[(i * width + j) * 3 + 2] = v;
			}
			else {
				suppression_pixel[(i * width + j) * 3] = 0;
				suppression_pixel[(i * width + j) * 3 + 1] = 0;
				suppression_pixel[(i * width + j) * 3 + 2] = 0;
			}

		}
	}
	*/
	
}
void GPU_Non_maximum_Suppression(int width, int height, uint8_t* angle, uint8_t* sobel, uint8_t* suppression_pixel, uint8_t& min, uint8_t& max) {
	dim3 dimGrid2(49, 49, 1);
	dim3 dimBlock2(32, 32, 1);

	uint8_t* GPU_pix = NULL;
	uint8_t* GPU_sobel = NULL;
	uint8_t* GPU_angle = NULL;

	uint8_t* GPU_max = NULL;
	cudaMalloc((void**)&GPU_max, 2 * sizeof(uint8_t));

	cudaMalloc((void**)&GPU_sobel, (width) * (height) * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_angle, (width) * (height) * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_pix, (width) * (height) * 3 * sizeof(uint8_t));

	uint8_t m_check[2];
	
	m_check[0] = min;
	m_check[1] = max;
	cudaMemcpy(GPU_max, m_check, 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	
	//cudaMemset(GPU_max, (unit8_t)min, sizeof(uint8_t));
	//cudaMemset(GPU_max + 1, (unit8_t)max, sizeof(uint8_t));

	cudaMemcpy(GPU_sobel, sobel, (width) * (height) * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_angle, angle, (width) * (height) * sizeof(uint8_t), cudaMemcpyHostToDevice);

	GPU_max2 << < dimGrid2, dimBlock2 >> > (width, height, GPU_angle, GPU_sobel, GPU_pix, GPU_max);	//GPU TMP CAREFUL

	cudaMemcpy(suppression_pixel, GPU_pix, (width) * (height) * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_check, GPU_max, 2*sizeof(uint8_t), cudaMemcpyDeviceToHost);

	min = m_check[0];
	max = m_check[1];

	cudaFree(GPU_pix);
	cudaFree(GPU_max);
	cudaFree(GPU_sobel);
	cudaFree(GPU_angle);

	cudaError_t err;
	err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
	}
}

__device__ void GPU_Hysteresis_check(int width, int height, int x, int y, uint8_t* hysteresis) {
	for (int i = y - 1; i < y + 2; i++) {
		for (int j = x - 1; j < x + 2; j++) {
			if ((i < height && j < width) && (i >= 0 && j >= 0)) {
				if (hysteresis[(i * width + j) * 3] == 255) {
					hysteresis[(y * width + x) * 3] = 255;
					hysteresis[(y * width + x) * 3 + 1] = 255;
					hysteresis[(y * width + x) * 3 + 2] = 255;
					return;
				}
			}
		}
	}
}

__global__ void GPU_Hyster(int width, int height, uint8_t* suppression_pixel, uint8_t* hysteresis, uint8_t* chk) {
	uint8_t diff = chk[1] - chk[0];
	uint8_t low_t = chk[0] + diff * 0.01;
	uint8_t high_t = chk[0] + diff * 0.2;

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < height && j < width) {
		uint8_t v = suppression_pixel[(i * width + j) * 3];
		if (v < low_t) {
			hysteresis[(i * width + j) * 3] = 0;
			hysteresis[(i * width + j) * 3 + 1] = 0;
			hysteresis[(i * width + j) * 3 + 2] = 0;
		}
		else if (v < high_t) {
			hysteresis[(i * width + j) * 3] = 123;
			hysteresis[(i * width + j) * 3 + 1] = 123;
			hysteresis[(i * width + j) * 3 + 2] = 123;
		}
		else {
			hysteresis[(i * width + j) * 3] = 255;
			hysteresis[(i * width + j) * 3 + 1] = 255;
			hysteresis[(i * width + j) * 3 + 2] = 255;
		}
		__syncthreads();

		if (hysteresis[(i * width + j) * 3] == 255) {
			GPU_Hysteresis_check(width, height, j, i, hysteresis);
		}
		__syncthreads();

		if (hysteresis[(i * width + j) * 3] != 255) {
			hysteresis[(i * width + j) * 3] = 0;
			hysteresis[(i * width + j) * 3 + 1] = 0;
			hysteresis[(i * width + j) * 3 + 2] = 0;
		}
	}
	/*
	if (i < height) {
		for (int j = 0; j < width; j++) {
			uint8_t v = suppression_pixel[(i * width + j) * 3];
			if (v < low_t) {
				hysteresis[(i * width + j) * 3] = 0;
				hysteresis[(i * width + j) * 3 + 1] = 0;
				hysteresis[(i * width + j) * 3 + 2] = 0;
			}
			else if (v < high_t) {
				hysteresis[(i * width + j) * 3] = 123;
				hysteresis[(i * width + j) * 3 + 1] = 123;
				hysteresis[(i * width + j) * 3 + 2] = 123;
			}
			else {
				hysteresis[(i * width + j) * 3] = 255;
				hysteresis[(i * width + j) * 3 + 1] = 255;
				hysteresis[(i * width + j) * 3 + 2] = 255;
			}
		}
		__syncthreads();

		for (int j = 0; j < width; j++) {
			if (hysteresis[(i * width + j) * 3] == 255) {
				GPU_Hysteresis_check(width, height, j, i, hysteresis);
			}
		}
		__syncthreads();

		for (int j = 0; j < width; j++) {
			if (hysteresis[(i * width + j) * 3] != 255) {
				hysteresis[(i * width + j) * 3] = 0;
				hysteresis[(i * width + j) * 3 + 1] = 0;
				hysteresis[(i * width + j) * 3 + 2] = 0;
			}
		}
	}
	*/
	
}
void GPU_Hysteresis_Thresholding(int width, int height, uint8_t *suppression_pixel, uint8_t *hysteresis, uint8_t min, uint8_t max) {
	uint8_t* GPU_pix = NULL;
	uint8_t* GPU_hy = NULL;

	cudaMalloc((void**)&GPU_pix, (width) * (height) * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&GPU_hy, (width) * (height) * 3 * sizeof(uint8_t));

	uint8_t m_check[2];
	m_check[0] = min;
	m_check[1] = max;
	printf("w : %d h : %d\n", width, height);

	uint8_t* GPU_max = NULL;
	cudaMalloc((void**)&GPU_max, 2 * sizeof(uint8_t));
	printf("min : %d max : %d\n", m_check[0], m_check[1]);
	cudaMemcpy(GPU_max, m_check, 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);

	//cudaMemset(GPU_hy, (uint8_t)0, (width) * (height) * 3 * sizeof(uint8_t));
	cudaMemcpy(GPU_pix, suppression_pixel, (width) * (height) * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

	dim3 dimGrid2(49, 49, 1);
	dim3 dimBlock2(32, 32, 1);
	GPU_Hyster << < dimGrid2, dimBlock2 >> > (width, height, GPU_pix, GPU_hy, GPU_max);

	cudaMemcpy(hysteresis, GPU_hy, (width) * (height) * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_check, GPU_max, 2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	min = m_check[0];
	max = m_check[1];
	printf("min : %d max : %d\n", m_check[0], m_check[1]);

	cudaError_t err;
	err = cudaGetLastError(); // `cudaGetLastError` will return the error from above.
	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
	}

	cudaFree(GPU_max);
	cudaFree(GPU_pix);
	cudaFree(GPU_hy);

}
