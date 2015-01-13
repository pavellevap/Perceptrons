#include "IO.h"

IO::~IO(){
	closeIF();
	closeOF();
}
void IO::openIF(string fn){
    ic = 0;
    it = 0;
	in.open(fn.c_str(),ios::binary|ios::in);
}
void IO::openOF(string fn){
    oc = 0;
    ot = (1 << 7);
	out.open(fn.c_str(),ios::binary|ios::out);
}

void IO::closeIF(){
	in.close();
}
void IO::closeOF(){
    if (ot != (1 << 7))
        out.write((char*)&oc, 1);
	out.close();
}
void IO::writebit(uchar b){
    oc |= ot * b;
    ot >>= 1;
    if (ot == 0) {
        out.write((char*)&oc, 1);
        ot = (1 << 7);
        oc = 0;
    }
}
void IO::write(uchar* x, int s){
    for (int a = 0; s > 0; a++)
        for (int b = 7; b >= 0 && s > 0; b--, s--)
            writebit((x[a] & (1 << b)) != 0);
}

bool IO::readbit(uchar& b){
    bool flag = true;
    if (it == 0){
        it = (1 << 7);
        flag = flag && in.read((char*)&ic, 1);
    }
    b = (ic & it) != 0;
    it >>= 1;
    return flag;
}

bool IO::read(uchar* x, int s){
    bool flag = true;
    for (int a = 0; s > 0; a++){
        x[a] = 0;
        for (int b = 7; b >= 0 && s > 0; b--, s--){
            uchar q;
            flag = flag && readbit(q);
            x[a] |= (q << b);
        }
    }
    return flag;
}
void IO::readFile(vector<uchar>& s){
	uchar ch;
	while (read((uchar*)&ch, 8)) s.push_back(ch);
}
