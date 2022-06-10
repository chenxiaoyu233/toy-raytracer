#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
using namespace std;

typedef float Ftype;

struct Vec {
    Ftype w[3];
    
    Vec(Ftype x = 0, Ftype y = 0, Ftype z = 0);
    Ftype& operator[] (int i);
    Ftype norm(int p = 2);
};
typedef Vec Color;

Vec::Vec(Ftype x, Ftype y, Ftype z) { w[0] = x, w[1] = y, w[2] = z; }
Ftype& Vec::operator[] (int i) { return w[i]; }
Ftype Vec::norm(int p) { return pow(pow(w[0], p) + pow(w[1], p) + pow(w[2], p), 1/p); }

Vec operator + (Vec &a, Vec &b)  { return Vec(a[0] + b[0], a[1] + b[1], a[2] + b[2]); }
Vec operator - (Vec &a, Vec &b)  { return Vec(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
Vec operator * (Vec &a, Ftype b) { return Vec(a[0] * b, a[1] * b, a[2] * b); }
Vec operator * (Ftype a, Vec &b) { return Vec(a * b[0], a * b[1], a * b[2]); }
Vec operator / (Vec &a, Ftype b) { return Vec(a[0] / b, a[1] / b, a[2] / b); }
Vec operator / (Ftype a, Vec &b) { return Vec(a / b[0], a / b[1], a / b[2]); }
    

Ftype dot(Vec &a, Vec &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Vec cross(Vec &a, Vec &b) {
    return Vec(a[1] * b[2] - a[2] * b[1],
               a[0] * b[2] - a[2] * b[0],
               a[0] * b[1] - a[1] * b[0]); }

Color** createImg(int r, int c) {
    Color **p = new Color* [r];
    for (int i = 0; i < r; i++) p[i] = new Color[c];
    return p;
}
void deleteImg(Color **p, int r) {
    for (int i = 0; i < r; i++) delete[] p[i];
    delete[] p;
}
void toPPM(Color **img, int r, int c, string fname = "out.ppm") {
    FILE *fpt = fopen(fname.c_str(), "w");
    fprintf(fpt, "P3\n");
    fprintf(fpt, "%d %d\n255\n", c, r);
    for(int i = 0; i < r; i++)
        for(int j = 0; j < c; j++){
            for(int k = 0; k < 3; k++) 
                fprintf(fpt, "%d ", int(img[i][j][k]));
            fprintf(fpt, "\n");
        }
    fclose(fpt);
}
//Ftype& Vec::operator [] (int i) { return w[i];}
//Vec Vec::operator + (const Vec &a, const Vec &b) { }
//Vec Vec::operator - (const Vec &a, const Vec &b) { }
//Vec Vec::operator * (const Vec &a, const Vec &b) { }
//Vec Vec::operator / (const Vec &a, Ftype b) { }


int main() {
    int c = 200, r = 100;
    Color **a = createImg(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            a[i][j][0] = 255.99 * float(i) / float(r);
            a[i][j][1] = 255.99 * float(j) / float(c);
            a[i][j][2] = 255.99 * 0.2;
        }
    toPPM(a, r, c);
    deleteImg(a, r);
    return 0;
}

