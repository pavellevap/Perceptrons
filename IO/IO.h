#ifndef IOCLASS
#define IOCLASS

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

using namespace std;

class IO {
public:
	void openIF(string fn);
	void openOF(string fn);
	//void setInputStream(ifstream fin);
	//void setOutputStream(ofstream fout);
	void closeIF();
	void closeOF();
	void writebit(uchar b);
	void write(uchar* x, int s);
	template <class T>
	void writet(T x, int s){
        for (int i = s; i > 0; i--){
            T q = 1;
            writebit((x & (q << (i - 1))) != 0);
        }
	}
	bool readbit(uchar& b);
	bool read(uchar* x, int s);
	template <class T>
	bool readt(T& x, int s){
        bool flag = true;
        x = 0;
        for (int i = s; i > 0; i--){
            uchar b;
            flag = flag && readbit(b);
            T q = b;
            x |= q << (i - 1);
        }
        return flag;
	}
	void readFile(vector<uchar>& s);
	~IO();

private:
	ifstream in;
	ofstream out;
	uchar oc, ic;
	uint ot, it;
};

#endif   /// IOCLASS
