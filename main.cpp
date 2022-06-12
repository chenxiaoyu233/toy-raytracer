#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <functional>
using namespace std;

typedef float Ftype;

struct Vec {
    Ftype w[3];
    
    Vec(Ftype x = 0, Ftype y = 0, Ftype z = 0) { w[0] = x, w[1] = y, w[2] = z; }
    Ftype& operator[] (int i) { return w[i]; }
    const Ftype operator[] (int i) const { return w[i]; }
    Vec& operator += (const Vec &b) { w[0] += b[0]; w[1] += b[1]; w[2] += b[2]; return *this; }
    Vec& operator -= (const Vec &b) { w[0] -= b[0]; w[1] -= b[1]; w[2] -= b[2]; return *this; }
    Vec& operator *= (Ftype b)      { w[0] *= b; w[1] *= b; w[2] *= b; return *this; }
    Vec& operator /= (Ftype b)      { w[0] /= b; w[1] /= b; w[2] /= b; return *this; }
    Vec& operator *= (const Vec &b) { w[0] *= b[0]; w[1] *= b[1]; w[2] *= b[2]; return *this; }
    Vec& operator /= (const Vec &b) { w[0] /= b[0]; w[1] /= b[1]; w[2] /= b[2]; return *this; }
    Ftype norm () { return sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); }
    Ftype x () { return w[0]; }
    Ftype y () { return w[1]; }
    Ftype z () { return w[2]; }
};

Vec operator + (const Vec &a, const Vec &b) { return Vec(a[0] + b[0], a[1] + b[1], a[2] + b[2]); }
Vec operator - (const Vec &a, const Vec &b) { return Vec(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
Vec operator * (const Vec &a, const Vec &b) { return Vec(a[0] * b[0], a[1] * b[1], a[2] * b[2]); }
Vec operator / (const Vec &a, const Vec &b) { return Vec(a[0] / b[0], a[1] / b[1], a[2] / b[2]); }
Vec operator * (const Vec &a, Ftype b)      { return Vec(a[0] * b, a[1] * b, a[2] * b); }
Vec operator * (Ftype a, const Vec &b)      { return Vec(a * b[0], a * b[1], a * b[2]); }
Vec operator / (const Vec &a, Ftype b)      { return Vec(a[0] / b, a[1] / b, a[2] / b); }
Vec operator / (Ftype a, const Vec &b)      { return Vec(a / b[0], a / b[1], a / b[2]); }

Ftype dot (const Vec &a, const Vec &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

Vec cross (const Vec &a, const Vec &b) {
    return Vec(a[1] * b[2] - a[2] * b[1],
               a[2] * b[0] - a[0] * b[2],
               a[0] * b[1] - a[1] * b[0]); }

typedef Vec Color;

struct Ray { Vec o, p; };

struct Camera {
    Vec o, x, y, z;
    Ftype hh, hw;
    Camera(Vec up, Vec from, Vec at, Ftype angle, Ftype aspect) {
        o = from;
        z = (from - at) / (from - at).norm();
        x = cross(up, z), x /= x.norm();
        y = cross(z, x), y /= y.norm();
        hh = tan(angle * M_PI / 360);
        hw = aspect * hh;
    }
    Ray rayAt(Ftype i, Ftype j, Ftype r, Ftype c) {
        return Ray {o, (-hw + j / c * 2 * hw) * x + (hh - i / r * 2 * hh) * y - z};
    }
};

Vec randUnitBall() {
    Vec p;
    do p = 2 * Vec(drand48(), drand48(), drand48()) - Vec(1, 1, 1);
    while (p.norm() > 1);
    return p;
}

Vec reflect (const Vec& p, const Vec& n) { return p - 2 * dot(p, n) * n; }

struct Material {
    virtual ~Material() { }
    virtual function<Color(const Color&)> scatter (const Vec& n, const Ray& in, Ray& out) = 0;
};

struct Diffuse: Material {
    Color albedo;
    Diffuse(Color _albedo):albedo(_albedo) { }
    function<Color(const Color&)> scatter (const Vec& n, const Ray& in, Ray& out) {
        out.o = in.o;
        out.p = n + randUnitBall();
        return [=] (const Color& incol) -> Color { return albedo * incol; };
    }
};

struct Metal: Material {
    Color albedo;
    Ftype fuzz;
    Metal(Color _albedo, Ftype _fuzz = 0): albedo(_albedo), fuzz(_fuzz) { }
    function<Color(const Color&)> scatter (const Vec& n, const Ray& in, Ray &out) {
        out.o = in.o;
        out.p = reflect(in.p, n) + fuzz * randUnitBall();
        return [=] (const Color& incol) -> Color { return (dot(out.p, n) > 0) * albedo * incol; };
    }
};

Ftype reflectRatio(Ftype r, Ftype cos) { // Schlick's approximation
    Ftype R0 = (1 - r) / (1 + r);
    R0 = R0 * R0;
    return R0 + (1 - R0) * pow(1 - cos, 5) ;
}

struct Glass: Material {
    Color albedo;
    Ftype ratio; // sin θt / sin θi == ni / nt
    Glass(Ftype _ratio, Color _albedo = Vec(0.95, 0.95, 0.95)): albedo(_albedo), ratio(_ratio) { }
    function<Color(const Color&)> scatter (const Vec& n, const Ray& in, Ray &out) {
        Ftype r = dot(in.p, n) > 0 ? 1.0 / ratio : ratio;
        Vec pcos = dot(in.p, n) * n, psin = in.p - pcos;
        Ftype cos = pcos.norm(), sin = psin.norm();
        Ftype nsin = sin * r, ncos = sqrt(1 - nsin * nsin);
        
        out.o = in.o;
        if (nsin >= 1 || drand48() < reflectRatio(r, cos)) out.p = reflect(in.p, n);
        else out.p = pcos * ncos / cos + psin * nsin / sin;
        return [=] (const Color& incol) -> Color { return  albedo * incol; };
    }
};

struct Obj {
    Material *mat;
    Obj(Material *_mat):mat(_mat){ }
    virtual ~Obj() {}
    virtual Ftype hit (const Ray& ray) = 0;
    virtual Vec n (const Vec& p) = 0;
};
typedef vector<Obj*> ObjList;

struct Sphere: Obj {
    Vec o; Ftype r;

    Sphere(Vec _o, Ftype _r, Material* mat): Obj(mat), o(_o), r(_r) {}
    
    Ftype hit (const Ray& ray) {
        Ftype a = dot(ray.p, ray.p);
        Ftype b = 2.0 * dot(ray.p, ray.o - o);
        Ftype c = dot(ray.o - o, ray.o - o) - r * r;
        Ftype d = b * b - 4 * a * c;
        if (d < 0) return -1;
        Ftype t1 = (-b + sqrt(d)) / (2 * a);
        Ftype t2 = (-b - sqrt(d)) / (2 * a);
        return t2 > 0.001 ? t2 : t1;
    }

    Vec n (const Vec& p) { return (p - o) / (p - o).norm(); }
};

struct Triangle: Obj {
    Vec a, b, c, nor;

    Triangle(Vec _a, Vec _b, Vec _c, Vec _nor, Material *mat): Obj(mat), nor(_nor), a(_a), b(_b), c(_c) {
        nor /= nor.norm();
        if (dot(cross(b - a, c - a), nor) < 0) swap(b, c);
    }

    Ftype hit (const Ray& ray) {
        Ftype t = dot(a - ray.o, nor) / dot(ray.p, nor), eps = -0.0001;
        Vec at = ray.o + ray.p * t;
        if ( dot(cross(b - a, at - a), nor) >= eps &&
             dot(cross(c - b, at - b), nor) >= eps &&
             dot(cross(a - c, at - c), nor) >= eps    ) return t;
        else return -1;
    }

    Vec n (const Vec& p) { return nor; }
};


Color rayTrace (Ray ray, ObjList& objs, int depth = 50) {
    ray.p = ray.p / ray.p.norm();
    Ftype t = -1.0; Obj* cur = NULL;
    for (auto &obj: objs) {
        Ftype tmp = obj -> hit(ray);
        if (tmp > 0.001 && (tmp < t || cur == NULL))
            t = tmp, cur = obj; 
    }
    if (cur != NULL) {
        Vec at = ray.o + ray.p * t;
        Vec n = cur -> n(at);
        Ray out, in {at, ray.p};
        auto col = cur -> mat -> scatter(n, in, out);
        if (depth > 0) return col(rayTrace(out, objs, depth - 1));
        else return Color(0, 0, 0);
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

struct ProgressBar {
    int n, w;
    ProgressBar(int _w): w(_w) { n = 0; }
    ~ProgressBar() { printf("\n"); }
    void show() {
        printf("\r Rendering: [");
        for (int i = 1; i <= w; i++)
            if (i <= n) printf("=");
            else printf(" ");
        printf("] %3d%%", int(float(n) / float(w) * 100 + 0.9));
        fflush(stdout);
    }
    void update(float nn) {
        if (w * nn <= n) return;
        n = w * nn + 0.9; show();
    }
};

void readBinarySTL(string filename, ObjList& objs) {
    FILE *f = fopen(filename.c_str(), "rb");

    // get size of the buffer
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    char *buffer = new char[size];
    fread(buffer, 1, size, f);

    unsigned int n = *(unsigned int*)(buffer + 80);
    for (int i = 0; i < n; i++) {
        auto flt = [=] (int idx) -> float { return *(float*)(buffer + 84 + i * 50 + idx * 4); };
        objs.push_back(
            new Triangle {
                Vec(flt(3), flt(4), flt(5)),
                Vec(flt(6), flt(7), flt(8)),
                Vec(flt(9), flt(10), flt(11)),
                Vec(flt(0), flt(1), flt(2)),
                new Diffuse(Color(0.3, 0.2, 0.8))
            }
        );
    }
    
    delete[] buffer;
    fclose(f);

    fprintf(stderr, "construct over: %lu triangles\n", objs.size());

}

int main () {
    ProgressBar progress(70);
    int c = 1600, r = 900;
    Color **img = createImg(r, c);
    Camera cam {
        Vec (0, 0, 1),
        Vec (-1, -30, 25),
        Vec (-1.726, 19.175, 18.6763),
        90, 16.0 / 9.0
    };
    /*
    ObjList objs {
        new Sphere {Vec(0, 0, -1), 0.5, new Diffuse(Color(0.8, 0.3, 0.3))},
        new Sphere {Vec(1, 0, -1), 0.5, new Metal(Color(0.8, 0.6, 0.2), 0.1)},
        new Sphere {Vec(-1, 0, -1), 0.5, new Glass(2.0/3.0)},
        new Sphere {Vec(-1, 0, -1), 0.45, new Glass(1.5)},
        new Sphere {Vec(0, -100.5, -1), 100, new Diffuse(Color(0.8, 0.8, 0))},
        new Triangle {Vec(-2, 0, -2), Vec(2, 0, -2), Vec(0, 2, -2), Vec(0, 0, 1), new Metal(Color(0.4, 0.2, 0.8), 0.01)}
        }; */
    ObjList objs;
    readBinarySTL("little-witch.stl", objs);
    objs.push_back(new Sphere {Vec(0, 0, -10000.5), 10000, new Diffuse(Color(0.8, 0.8, 0))});
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            img[i][j] = Color(0, 0, 0);
            for (int s = 0; s < 4; s++)
                img[i][j] += rayTrace(cam.rayAt(Ftype(i) + drand48(), Ftype(j) + drand48(), r, c), objs, 10);
            img[i][j] /= 4.0;
            img[i][j] = Color(sqrt(img[i][j][0]), sqrt(img[i][j][1]), sqrt(img[i][j][2]));
            img[i][j] = img[i][j] * 255.99;
            progress.update(float(i * c + j) / float(r * c));
        }
    }
    toPPM(img, r, c);
    deleteImg(img, r);
    for (auto obj: objs) {
        delete obj -> mat;
        delete obj;
    }
    return 0;
}

