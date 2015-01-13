/**
Библиотека для работы с перцептронами
*/

#ifndef PERCEPTRON_H_INCLUDED
#define PERCEPTRON_H_INCLUDED

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <string.h>

#include <cuda.h>

#include "IO/IO.h"

using std::string;
using std::cerr;
using std::endl;
using std::pair;
using std::max;
using std::min;
using std::swap;
using std::cout;
using std::ofstream;
using std::ifstream;

const int THREADS_PER_BLOCK = 1024;

/**
======================================================================================================
                                Элементарный перцептрон.
 https://ru.wikipedia.org/wiki/Перцептрон.
 Перцептрон представляет собой сеть нейронов.
 Нейроны разделены на 3 типа: S - сенсорные, A - ассоциативные, R - результатирущие.
 R нейроны - выходные значения перцепртона. S нейроны - входные значения перцептрона.
 Каждый А нейрон получеет сигнал от некоторых S нейронов и передает сигнал R нейронам.
 Веса S-A ребер - элементы {-1, 1}, выбираются случайно и не меняются.
 Пороговые значения A нейронов равны 0.
 Выход А-нейронов - 0 или 1. Активировался нейрон или нет.
 Выход R-нейронов - целое число. > 0 если вход относится к 1ому классу, < 0 если ко второму.
 Обучение по методу коррекции ошибки.

 Замечания:
 1) Для гарантированного решения задачи xor двух битов потребовалось 20 A-нейронов
 2) Из-за того, что пороговые значения равны 0, то на нулевом входе будет нулевой выход.
     Чтобы этого избежать можно добавить один фиктивный вход, который всегда будет активным.
 3) Есть возможность ускорить обучение за счет запоминания выходных значений A нейронов
     для каждого входа.
 4) R нейрон возвращает не {+1, -1} а произвольное целое число, причем если это число больше 0,
     то скорее всего вход принадлежит объекту 1ого класса, иначе объекту 2ого класса.
     чем больше модуль этого числа, тем больше уверенность перцептрона в выданном ответе.
======================================================================================================
*/

/**
 * Структура для хранения информации о элементарном перцептроне
 */
struct ElementaryPerceptronData {
	size_t amountOfS;                    /** Количество S нейронов */
	size_t amountOfA;                    /** Количество A нейронов */
	size_t amountOfR;                    /** Количество R нейронов */

	short** ASEdges;                     /** Веса A-S ребер */
	short** RAEdges;                     /** Веса R-A ребер */

	~ElementaryPerceptronData();
};

void SaveElementaryPerceptron(const ElementaryPerceptronData& pd, string FileName);

void LoadElementaryPerceptron(ElementaryPerceptronData& pd, string FileName);



class cudaElementaryPerceptron {
public:

	cudaElementaryPerceptron();

	cudaElementaryPerceptron(size_t amountOfS, size_t amountOfA, size_t amountOfR);

	~cudaElementaryPerceptron();

	void initialize(size_t amountOfS, size_t amountOfA, size_t amountOfR);

	void restoreElementaryPerceptron(const ElementaryPerceptronData& pd);

	ElementaryPerceptronData getElementaryPerceptronData();

	void setInput(bool in[]);

	void setInput(size_t index, bool value);

	void setAOutput(bool out[]);

	void setAOutput(size_t index, bool value);

	void calculateAOutput();

	void calculateROutput();

	void calculateOutput();

	void correct(size_t index, int add);

	void teach(int desierdOutput[]);

	bool* getAOutput();

	bool getAOutput(size_t index);

	int* getROutput();

	int getROutput(size_t index);

	size_t getAmountOfR();

	size_t getAmountOfA();

	size_t getAmountOfS();

private:
	size_t  amountOfS;                    	  /** Количество S нейронов */
	size_t  amountOfA;                    	  /** Количество A нейронов */
	size_t  amountOfR;                    	  /** Количество R нейронов */

	bool*   input;                        	  /** Входные значения для S нейронов. 0 или 1 */
	int*    ROutput;                          /** Выход R нейронов */

	short** dev_ASEdges;                      /** Веса A-S ребер */
	bool*   dev_AOutput;                      /** Выход A нейронов */
	short** dev_RAEdges;                      /** Веса R-A ребер */
};

#endif // PERCEPTRON_H_INCLUDED
