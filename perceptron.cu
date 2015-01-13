#include "perceptron.h"


/**
 *=======================================================================================================
 *                                  Элементарный перцептрон
 *=======================================================================================================
 */


void SaveElementaryPerceptron(const ElementaryPerceptronData& pd, string FileName) {
    IO io;
    io.openOF(FileName.c_str());
    io.writet(pd.amountOfS, sizeof(pd.amountOfS) * 8);
    io.writet(pd.amountOfA, sizeof(pd.amountOfA) * 8);
    io.writet(pd.amountOfR, sizeof(pd.amountOfR) * 8);
    for (size_t i = 0; i < pd.amountOfA; i++)
        for (size_t j = 0; j < pd.amountOfS; j++)
            if (pd.ASEdges[i][j] == 1)
                io.writebit(1);
            else
                io.writebit(0);
    for (size_t i = 0; i < pd.amountOfR; i++)
        for (size_t j = 0; j < pd.amountOfA; j++)
            io.writet(pd.RAEdges[i][j], sizeof(pd.RAEdges[i][j]) * 8);
    io.closeOF();
}

void LoadElementaryPerceptron(ElementaryPerceptronData& pd, string FileName) {
    IO io;
    io.openIF(FileName.c_str());
    io.readt(pd.amountOfS, sizeof(pd.amountOfS) * 8);
    io.readt(pd.amountOfA, sizeof(pd.amountOfA) * 8);
    io.readt(pd.amountOfR, sizeof(pd.amountOfR) * 8);

    pd.ASEdges = new short*[pd.amountOfA];
    for (size_t i = 0; i < pd.amountOfA; i++) {
        pd.ASEdges[i] = new short[pd.amountOfS];
        for (size_t j = 0; j < pd.amountOfS; j++) {
            uchar bit;
            io.readbit(bit);
            if (bit)
                pd.ASEdges[i][j] = 1;
            else
                pd.ASEdges[i][j] = -1;
        }
    }
    pd.RAEdges = new short*[pd.amountOfR];
    for (size_t i = 0; i < pd.amountOfR; i++) {
        pd.RAEdges[i] = new short[pd.amountOfA];
        for (size_t j = 0; j < pd.amountOfA; j++)
            io.readt(pd.RAEdges[i][j], sizeof(pd.RAEdges[i][j]) * 8);
    }
    io.closeIF();
}

ElementaryPerceptronData::~ElementaryPerceptronData() {
    if (ASEdges) {
        for (size_t i = 0; i < amountOfA; i++)
            if (ASEdges[i])
                delete ASEdges[i];
        delete ASEdges;
    }

    if (RAEdges) {
        for (size_t i = 0; i < amountOfR; i++)
            if (RAEdges[i])
                delete RAEdges[i];
        delete RAEdges;
    }
}

cudaElementaryPerceptron::cudaElementaryPerceptron() { }

cudaElementaryPerceptron::cudaElementaryPerceptron(size_t amountOfS, size_t amountOfA, size_t amountOfR) {
	initialize(amountOfS, amountOfA, amountOfR);
}

__global__ void generateASLayerKernel(short** dev_ASEdges, size_t amountOfS, int seed) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int index = y + amountOfS * x;

	int randomNumber = seed + index;
	randomNumber = (randomNumber * randomNumber + randomNumber) % 1000000007;
	randomNumber = (randomNumber + 1000000007) % 1000000007;
	randomNumber = ((randomNumber * 214013L + 2531011L) >> 16) & 0x7fff;

	dev_ASEdges[x][y] = (randomNumber & 1) ? 1 : -1;
}

void cudaElementaryPerceptron::initialize(size_t amountOfS, size_t amountOfA, size_t amountOfR){
	this->amountOfS = amountOfS;
	this->amountOfA = amountOfA;
    this->amountOfR = amountOfR;

    cudaMalloc((void**)&dev_AOutput, amountOfA * sizeof(bool));
    ROutput = new int[amountOfR];
    input = new bool[amountOfS];

    short** ptr;

    cudaMalloc((void**)&dev_ASEdges, amountOfA * sizeof(short*));
    ptr = new short*[amountOfA];
    for (size_t i = 0; i < amountOfA; i++)
    	cudaMalloc((void**)&ptr[i], amountOfS * sizeof(short));
    cudaMemcpy(dev_ASEdges, ptr, amountOfA * sizeof(short*), cudaMemcpyHostToDevice);
    delete ptr;

    generateASLayerKernel<<<dim3(amountOfA, amountOfS), 1>>>(dev_ASEdges, amountOfS, clock());

    cudaMalloc((void**)&dev_RAEdges, amountOfR * sizeof(short*));
    ptr = new short*[amountOfR];
    for (size_t i = 0; i < amountOfR; i++) {
        cudaMalloc((void**)&ptr[i], amountOfA * sizeof(short));
        cudaMemset(ptr[i], 0, amountOfA * sizeof(short));
    }
    cudaMemcpy(dev_RAEdges, ptr, amountOfR * sizeof(short*), cudaMemcpyHostToDevice);
    delete ptr;
}

cudaElementaryPerceptron::~cudaElementaryPerceptron() {
	cudaFree(dev_AOutput);
	delete ROutput;
	delete input;

	short** ptr;

	if (dev_ASEdges) {
		ptr = new short*[amountOfA];
		cudaMemcpy(ptr, dev_ASEdges, amountOfA * sizeof(short*), cudaMemcpyDeviceToHost);
		for (size_t i = 0; i < amountOfA; i++)
			if (ptr[i])
				cudaFree(ptr[i]);
		delete ptr;
	    cudaFree(dev_ASEdges);
	}

	if (dev_RAEdges) {
		ptr = new short*[amountOfR];
		cudaMemcpy(ptr, dev_RAEdges, amountOfR * sizeof(short*), cudaMemcpyDeviceToHost);
		for (size_t i = 0; i < amountOfR; i++)
			if (ptr[i])
				cudaFree(ptr[i]);
		delete ptr;
		cudaFree(dev_RAEdges);
	}
}

void cudaElementaryPerceptron::restoreElementaryPerceptron(const ElementaryPerceptronData& pd) {
	this->~cudaElementaryPerceptron();

    amountOfR = pd.amountOfR;
    amountOfA = pd.amountOfA;
    amountOfS = pd.amountOfS;

    cudaMalloc((void**)&dev_AOutput, amountOfA * sizeof(bool));
    ROutput = new int[amountOfR];
    input = new bool[amountOfS];

    short** ptr;

    cudaMalloc((void**)&dev_ASEdges, amountOfA * sizeof(short*));
    ptr = new short*[amountOfA];
    for (size_t i = 0; i < amountOfA; i++) {
        cudaMalloc((void**)&ptr[i], amountOfS * sizeof(short));
        cudaMemcpy(ptr[i], pd.ASEdges[i], sizeof(short) * amountOfS, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dev_ASEdges, ptr, amountOfA * sizeof(short*), cudaMemcpyHostToDevice);
    delete ptr;

    cudaMalloc((void**)&dev_RAEdges, amountOfR * sizeof(short*));
    ptr = new short*[amountOfR];
    for (size_t i = 0; i < amountOfR; i++) {
        cudaMalloc((void**)&ptr[i], amountOfA * sizeof(short));
        cudaMemcpy(ptr[i], pd.RAEdges[i], sizeof(short) * amountOfA, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dev_RAEdges, ptr, amountOfR * sizeof(short*), cudaMemcpyHostToDevice);
    delete ptr;
}

ElementaryPerceptronData cudaElementaryPerceptron::getElementaryPerceptronData() {
    ElementaryPerceptronData pd;
    pd.amountOfR = amountOfR;
    pd.amountOfA = amountOfA;
    pd.amountOfS = amountOfS;

    short** ptr;

    ptr = new short*[amountOfA];
    cudaMemcpy(ptr, dev_ASEdges, amountOfA * sizeof(short*), cudaMemcpyDeviceToHost);
    pd.ASEdges = new short*[amountOfA];
    for (size_t i = 0; i < amountOfA; i++) {
        pd.ASEdges[i] = new short[amountOfS];
        cudaMemcpy(pd.ASEdges[i], ptr[i], sizeof(short) * amountOfS, cudaMemcpyDeviceToHost);
    }
    delete ptr;

    ptr = new short*[amountOfR];
    cudaMemcpy(ptr, dev_RAEdges, amountOfR * sizeof(short*), cudaMemcpyDeviceToHost);
    pd.RAEdges = new short*[amountOfR];
    for (size_t i = 0; i < amountOfR; i++) {
        pd.RAEdges[i] = new short[amountOfA];
        cudaMemcpy(pd.RAEdges[i], ptr[i], sizeof(short) * amountOfA, cudaMemcpyDeviceToHost);
    }
    delete ptr;

    return pd;
}

void cudaElementaryPerceptron::setInput(bool* in) {
    memcpy(input, in, amountOfS * sizeof(bool));
}

void cudaElementaryPerceptron::setInput(size_t index, bool value) {
    if (index >= amountOfS)
        cerr << "Выход за границу массива в функции cudaElementaryPerceptron::setInput()\n";
    input[index] = value;
}

void cudaElementaryPerceptron::setAOutput(bool* out) {
    cudaMemcpy(dev_AOutput, out, amountOfA * sizeof(bool), cudaMemcpyHostToDevice);
}

void cudaElementaryPerceptron::setAOutput(size_t index, bool value) {
    if (index >= amountOfA)
        cerr << "Выход за границу массива в функции cudaElementaryPerceptron::setAOutput()\n";
    cudaMemcpy(dev_AOutput + index, &value, sizeof(value), cudaMemcpyHostToDevice);
}

void __global__ calculateAOutputKernel(bool* dev_input, bool* dev_AOutput, size_t amountOfS, size_t amountOfA, short** dev_ASEdges) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < amountOfA) {
		int sum = 0;
		for (int i = 0; i < amountOfS; i++)
			if (dev_input[i])
				sum += dev_ASEdges[index][i];
		dev_AOutput[index] = sum > 0;
	}
}



void cudaElementaryPerceptron::calculateAOutput() {
	bool* dev_input;
	cudaMalloc((void**)&dev_input, amountOfS * sizeof(bool));
	cudaMemcpy(dev_input, input, amountOfS * sizeof(bool), cudaMemcpyHostToDevice);

	int amountOfBlocks = (amountOfA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    calculateAOutputKernel<<<amountOfBlocks, THREADS_PER_BLOCK>>>(dev_input, dev_AOutput, amountOfS, amountOfA, dev_ASEdges);
}

__global__ void calculateROutputKernel(int RIndex, bool* dev_AOutput, int* dev_sum, size_t dev_amountOfA, short** dev_RAEdges) {
	__shared__ int tmp[THREADS_PER_BLOCK];

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dev_amountOfA && dev_AOutput[index])
		tmp[threadIdx.x] = dev_RAEdges[RIndex][index];
	else
		tmp[threadIdx.x] = 0;


	__syncthreads();

	int i = THREADS_PER_BLOCK >> 1;
	while (i) {
		if (threadIdx.x < i)
			tmp[threadIdx.x] += tmp[threadIdx.x + i];
		i >>= 1;
		__syncthreads();
	}
	if (threadIdx.x == 0)
		dev_sum[blockIdx.x] = tmp[0];
}

void cudaElementaryPerceptron::calculateROutput() {
	int amountOfBlocks = (amountOfA + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int* sum = new int[amountOfBlocks];
	int* dev_sum;
	cudaMalloc((void**)&dev_sum, amountOfBlocks * sizeof(int));

    for (size_t i = 0; i < amountOfR; ++i) {
        calculateROutputKernel<<<amountOfBlocks, THREADS_PER_BLOCK>>>(i, dev_AOutput, dev_sum, amountOfA, dev_RAEdges);

        cudaMemcpy(sum, dev_sum, amountOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        ROutput[i] = 0;
        for (size_t j = 0; j < amountOfBlocks; j++)
        	ROutput[i] += sum[j];
    }

    delete sum;
    cudaFree(dev_sum);
}

void cudaElementaryPerceptron::calculateOutput() {
	/*float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);*/
    calculateAOutput();
    /*cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);*/

    /*cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%.6f	", elapsedTime);

    cudaEventRecord(start, 0);*/
    calculateROutput();
    /*cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("%.6f	", elapsedTime);*/

}

__global__ void correctKernel(int RIndex, int add, bool* dev_AOutput, short** dev_RAEdges) {
	int index = blockIdx.x;
	if (dev_AOutput[index])
		dev_RAEdges[RIndex][index] += add;
}

void cudaElementaryPerceptron::correct(size_t index, int add) {
    if (index >= amountOfR)
        cerr << "Выход за границы массива в функции cudaElementaryPerceptron::correct()\n";

    correctKernel<<<amountOfA, 1>>>(index, add, dev_AOutput, dev_RAEdges);
}

void cudaElementaryPerceptron::teach(int* desierdOutput) {
	//#pragma omp parallel for                 /// ?????
    for (size_t i = 0; i < amountOfR; i++) {
        bool ans1 = ROutput[i] > 0;
        bool ans2 = desierdOutput[i] > 0;
        if (ans1 != ans2)
            if (ROutput[i] > 0)
                correct(i, -1);
            else
                correct(i, 1);
    }
}

bool* cudaElementaryPerceptron::getAOutput() {
    bool* output = new bool[amountOfA];
    cudaMemcpy(output, dev_AOutput, sizeof(bool) * amountOfA, cudaMemcpyDeviceToHost);
    return output;
}

bool cudaElementaryPerceptron::getAOutput(size_t index) {
    if (index >= amountOfA)
        cerr << "Выход за границу массива в функции cudaElementaryPerceptron::getAOutput()\n";
    bool value;
    cudaMemcpy(&value, dev_AOutput + index, sizeof(value), cudaMemcpyDeviceToHost);

    return value;
}

int* cudaElementaryPerceptron::getROutput() {
    int* output = new int[amountOfR];
    memcpy(output, ROutput, sizeof(int) * amountOfR);
    return output;
}

int cudaElementaryPerceptron::getROutput(size_t index) {
    if (index >= amountOfR)
        cerr << "Выход за границу массива в функции cudaElementaryPerceptron::getROutput()\n";
    return ROutput[index];
}

size_t cudaElementaryPerceptron::getAmountOfR() {
    return amountOfR;
}

size_t cudaElementaryPerceptron::getAmountOfA() {
    return amountOfA;
}

size_t cudaElementaryPerceptron::getAmountOfS() {
    return amountOfS;
}

