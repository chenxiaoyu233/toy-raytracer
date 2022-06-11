#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
#include <vector>
using namespace std;

typedef float Ftype;

struct Vec {
    Ftype w[3];
    
    Vec(Ftype x = 0, Ftype y = 0, Ftype z = 0) { w[0] = x, w[1] = y, w[2] = z; }
    Ftype& operator[] (int i) { return w[i]; }
    const Ftype operator[] (int i) const { return w[i]; }
    Ftype norm () { return sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); }
    Ftype x () { return w[0]; }
    Ftype y () { return w[1]; }
    Ftype z () { return w[2]; }
};
typedef Vec Color;

Vec operator + (const Vec &a, const Vec &b) { return Vec(a[0] + b[0], a[1] + b[1], a[2] + b[2]); }
Vec operator - (const Vec &a, const Vec &b) { return Vec(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
Vec operator * (const Vec &a, Ftype b)      { return Vec(a[0] * b, a[1] * b, a[2] * b); }
Vec operator * (Ftype a, const Vec &b)      { return Vec(a * b[0], a * b[1], a * b[2]); }
Vec operator / (const Vec &a, Ftype b)      { return Vec(a[0] / b, a[1] / b, a[2] / b); }
Vec operator / (Ftype a, const Vec &b)      { return Vec(a / b[0], a / b[1], a / b[2]); }

Ftype dot (const Vec &a, const Vec &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Vec cross (const Vec &a, const Vec &b) {
    return Vec(a[1] * b[2] - a[2] * b[1],
               a[0] * b[2] - a[2] * b[0],
               a[0] * b[1] - a[1] * b[0]); }

struct Ray { Vec o, p; };

inline Ray rayFromCam (int i, int j, int r, int c) {
    Vec o(0, 0, 0);
    Vec p(-2 + float(j) / float(c) * 4.0 , 1 - float(i) / float(r) * 2.0  ,-1);
    return Ray {o, p};
}

struct Obj {
    virtual Ftype hit (const Ray& ray) = 0;
    virtual Vec n (const Vec& p) = 0;
};
typedef vector<Obj*> ObjList;

struct Sphere: Obj {
    Vec o; Ftype r;

    Sphere(Vec _o, Ftype _r):o(_o), r(_r) {}
    
    Ftype hit (const Ray& ray) {
        Ftype a = dot(ray.p, ray.p);
        Ftype b = 2.0 * dot(ray.p, ray.o - o);
        Ftype c = dot(ray.o - o, ray.o - o) - r * r;
        Ftype d = b * b - 4 * a * c;
        if (d < 0) return -1;
        Ftype t1 = (-b + sqrt(d)) / (2 * a);
        Ftype t2 = (-b - sqrt(d)) / (2 * a);
        return t2 > 0 ? t2 : t1;
    }

    Vec n (const Vec& p) { return (p - o) / (p - o).norm(); }
};

Color rayTrace (Ray ray, ObjList& objs) {
    ray.p = ray.p / ray.p.norm();
    Ftype t = -1.0; Obj* cur = NULL;
    for (auto &obj: objs) {
        Ftype tmp = obj -> hit(ray);
        if (tmp > 0 && (tmp < t || cur == NULL))
            t = tmp, cur = obj; 
    }
    if (cur != NULL) {
        Vec n = cur -> n(ray.o + ray.p * t);
        n = n / n.norm();
        return 0.5 * (n + Vec(1, 1, 1));
    }
    t = 0.5 * (ray.p.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

Color** createImg (int r, int c) {
    Color **p = new Color* [r];
    for (int i = 0; i < r; i++) p[i] = new Color[c];
    return p;
}

void deleteImg (Color **p, int r) {
    for (int i = 0; i < r; i++) delete[] p[i];
    delete[] p;
}

void toPPM (Color **img, int r, int c, string fname = "out.ppm") {
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

int main () {
    int c = 200, r = 100;
    Color **img = createImg(r, c);
    ObjList lst {
        new Sphere {Vec(0, 0, -1), 0.5},
        new Sphere {Vec(0, -100.5, -1), 100}
    };
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            img[i][j] = rayTrace(rayFromCam(i, j, r, c), lst);
            img[i][j] = img[i][j] * 255.99;
        }
    toPPM(img, r, c);
    deleteImg(img, r);
    return 0;
}

