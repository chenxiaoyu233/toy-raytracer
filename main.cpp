#include <iostream>
#include <cstdio>
#include <string>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <functional>
#include <algorithm>
#include <map>
#include <random>

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
    Ftype x () const { return w[0]; }
    Ftype y () const { return w[1]; }
    Ftype z () const { return w[2]; }
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
    void setup(Vec up, Vec from, Vec at, Ftype angle, Ftype aspect) {
        o = from;
        z = (from - at) / (from - at).norm();
        x = cross(up, z), x /= x.norm();
        y = cross(z, x), y /= y.norm();
        hh = tan(angle * M_PI / 360);
        hw = aspect * hh;
    }
    Camera() { }
    Camera(Vec up, Vec from, Vec at, Ftype angle, Ftype aspect) {
        setup(up, from, at, angle, aspect);
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

struct Obj;

struct HitRecord {
    Ftype t;
    Obj* obj;
};

struct BBox {
    Vec lb, ub;

    static BBox merge (const BBox &a, const BBox &b) {
        return BBox{
            Vec(std::min(a.lb.x(), b.lb.x()), std::min(a.lb.y(), b.lb.y()), std::min(a.lb.z(), b.lb.z())), 
            Vec(std::max(a.ub.x(), b.ub.x()), std::max(a.ub.y(), b.ub.y()), std::max(a.ub.z(), b.ub.z()))
        };
    }
};

struct Material;

typedef std::pair<Ftype, Ftype> UV;

struct Obj {
    Material *mat;
    Obj(Material *_mat):mat(_mat){ }
    virtual ~Obj() {}
    virtual HitRecord hit (const Ray& ray) = 0;
    virtual Vec n (const Vec& p) const = 0;
    virtual BBox calcBBox () const = 0;
    virtual UV uv (const Vec &p) const = 0;
};
typedef std::vector<Obj*> ObjList;

struct Texture {
    virtual ~Texture() { }
    virtual Color color(const Obj* obj, Vec p) = 0;
};

struct Material {
    virtual ~Material() { }
    virtual std::function<Color(const Color&)> scatter (const Obj* obj, const Ray& in, Ray& out) = 0;
};

struct BVHnode: Obj {
    BBox box;
    Obj *lson, *rson;

    BVHnode():Obj(NULL){ }

    Vec n (const Vec& p) const { return Vec(0, 0, 0); }

    UV uv (const Vec& p) const { return UV(0.0, 0.0); }

    bool hitBox(const Ray& ray) {
        auto interval = [=] (int axis) -> std::pair<Ftype, Ftype> {
            Ftype inv = 1.0f / ray.p[axis];
            Ftype t0 = (box.lb.w[axis] - ray.o.w[axis]) * inv;
            Ftype t1 = (box.ub.w[axis] - ray.o.w[axis]) * inv;
            if (inv < 0.0f) std::swap(t0, t1);
            return std::make_pair(t0, t1);
        };
        std::pair<Ftype, Ftype> pr = interval(0);
        for (int i = 1; i <= 2; ++i) {
            std::pair<Ftype, Ftype> tmp = interval(i);
            pr.first = tmp.first > pr.first ? tmp.first : pr.first;
            pr.second = tmp.second < pr.second ? tmp.second : pr.second;
            if (pr.second <= pr.first) return false;
        }
        return true;
    }

    HitRecord hit (const Ray& ray) { 
        HitRecord rt{-1, NULL};
        if (!hitBox(ray)) return rt;
        auto update = [&] (HitRecord tmp) -> void {
            if (tmp.obj != NULL && tmp.t > 0.005 
                    && (tmp.t < rt.t || rt.obj == NULL)) rt = tmp;
        };
        if (lson != NULL) update(lson -> hit(ray));
        if (rson != NULL) update(rson -> hit(ray));
        return rt;
    }

    BBox calcBBox() const { return box; }
};

struct BVH {
    Obj* root;
    ObjList nodePool;

    HitRecord hit (const Ray& ray) { return root -> hit(ray); }

    Obj* build(ObjList& objs, bool isRoot = true) {
        if (isRoot && objs.size() == 0) {
            std::cerr << "building empty BVH" << std::endl;
            return root = NULL;
        }
        if (objs.size() == 0) return NULL;
        if (!isRoot && objs.size() == 1) return objs[0];

        int axis = rand() % 3;
        auto cmp = [=] (const Obj* a, const Obj* b) -> bool {
            return (a -> calcBBox()).lb.w[axis] < (b -> calcBBox()).lb.w[axis];
        };
        sort(objs.begin(), objs.end(), cmp);

        BVHnode* rt = new BVHnode;
        if (isRoot) root = rt;
        nodePool.push_back(rt);
        rt -> box = objs[0] -> calcBBox();
        for (auto obj: objs) 
            rt -> box = BBox::merge(rt -> box, obj -> calcBBox());

        ObjList half; 
        size_t m = objs.size() >> 1;

        half.clear();
        for (size_t i = 0; i < m; ++i) half.push_back(objs[i]);
        rt -> lson = build(half, false);

        half.clear();
        for (size_t i = m; i < objs.size(); ++i) half.push_back(objs[i]);
        rt -> rson = build(half, false);

        return rt;
    }

    void clear() {
        for (auto pt: nodePool) delete pt;
        nodePool.clear();
    }

     BVH (ObjList &objs) { build(objs); }
    ~BVH () { clear(); }
};

struct PerlinNoise {
    const int pt_cnt = 256;
    std::vector<Vec> randv;
    std::vector<int> perm[3];
    PerlinNoise() {
        randv.resize(pt_cnt);
        for (int i = 0; i < pt_cnt; ++i) randv[i] = randUnitBall();
        for (int i = 0; i < 3; ++i) {
            perm[i].resize(pt_cnt);
            for (int j = 0; j < pt_cnt; ++j) perm[i][j] = j;
            auto rng = std::default_random_engine {};
            std::shuffle(perm[i].begin(), perm[i].end(), rng);
        }
    }
    Ftype noise(const Vec& p) {
        Ftype u = p.x() - floor(p.x());
        Ftype v = p.y() - floor(p.y());
        Ftype w = p.z() - floor(p.z());
        int x = int(floor(p.x()));
        int y = int(floor(p.y()));
        int z = int(floor(p.z()));

        // rand vec on the grid
        auto gvec = [=] (int i, int j, int k) -> Vec {
            return randv[
                perm[0][(x + i) & 255] ^ 
                perm[1][(y + j) & 255] ^ 
                perm[2][(z + k) & 255]
            ];
        };

       // smooth
       u = u * u * (3 - 2 * u);
       v = v * v * (3 - 2 * v);
       w = w * w * (3 - 2 * w);

       Ftype ret = 0;
       for (int i = 0; i <= 1; ++i)
           for (int j = 0; j <= 1; ++j)
               for (int k = 0; k <= 1; ++k) {
                   ret += (i * u + (1 - i) * (1 - u))
                        * (j * v + (1 - j) * (1 - v))
                        * (k * w + (1 - k) * (1 - w))
                        * dot( gvec(i, j, k), Vec(u - i, v - j, w - k) );
               }
       return ret;
    }
};

struct NoiseTexture: Texture {
    PerlinNoise perlin;
    Ftype scale;
    NoiseTexture(Ftype _scale = 40): scale(_scale) { }
    Color color(const Obj *obj, Vec p) { 
        return Color(1, 1, 1) * 0.5 * (1.0 + perlin.noise(scale * p));
    }
};

struct SolidColor: Texture {
    Color col;
    SolidColor(Color _col): col(_col) { }
    Color color(const Obj *obj, Vec p) { return col; }
};

struct CheckerTexture: Texture {
    Texture *odd, *even;
    CheckerTexture(Texture* _odd, Texture* _even): odd(_odd), even(_even) { }
    Color color(const Obj * obj, Vec p) {
        Ftype sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        return sines < 0 ? odd -> color(obj, p) : even -> color(obj, p);
    }
};

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
struct ImageTexture: Texture {
    unsigned char *data;
    int width, height;
    ImageTexture(const char* filename) {
        int cnt;
        data = stbi_load(filename, &width, &height, &cnt, 3);
        if (data == NULL) {
            std::cerr << "fail to load: " << filename << std::endl;
        }
    }
    ~ImageTexture() {
        stbi_image_free(data);
    }
    Color color(const Obj *obj, Vec p) {
        if (data == NULL) return Color(1, 0, 1);
        UV uv = obj -> uv(p);
        Ftype u = uv.first, v = 1 - uv.second;
        int r = int(v * height), c = int(u * width);
        r = r < height ? r : height - 1;
        c = c < width  ? c : width  - 1;
        unsigned char *pt = data + r * width * 3 + c * 3;
        return Color(pt[0], pt[1], pt[2]) / 255.0;
    }
};

struct Diffuse: Material {
    Texture* albedo;
    Diffuse(Texture* _albedo):albedo(_albedo) { }
    std::function<Color(const Color&)> scatter (const Obj* obj, const Ray& in, Ray& out) {
        Vec n = obj -> n(in.o);
        out.o = in.o;
        out.p = n + randUnitBall();
        return [=] (const Color& incol) -> Color { return albedo -> color(obj, in.o) * incol; };
    }
};

struct Metal: Material {
    Texture* albedo;
    Ftype fuzz;
    Metal(Texture* _albedo, Ftype _fuzz = 0): albedo(_albedo), fuzz(_fuzz) { }
    std::function<Color(const Color&)> scatter (const Obj* obj, const Ray& in, Ray &out) {
        Vec n = obj -> n(in.o);
        out.o = in.o;
        out.p = reflect(in.p, n) + fuzz * randUnitBall();
        return [=] (const Color& incol) -> Color { 
            return (dot(out.p, n) > 0) * albedo -> color(obj, in.o) * incol; 
        };
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
    std::function<Color(const Color&)> scatter (const Obj* obj, const Ray& in, Ray &out) {
        Vec n = obj -> n(in.o);
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


struct Sphere: Obj {
    Vec o; Ftype r;

    Sphere(Vec _o, Ftype _r, Material* mat): Obj(mat), o(_o), r(_r) {}
    
    HitRecord hit (const Ray& ray) {
        Ftype a = dot(ray.p, ray.p);
        Ftype b = 2.0 * dot(ray.p, ray.o - o);
        Ftype c = dot(ray.o - o, ray.o - o) - r * r;
        Ftype d = b * b - 4 * a * c;
        if (d < 0) return HitRecord{-1, this};
        Ftype t1 = (-b + sqrt(d)) / (2 * a);
        Ftype t2 = (-b - sqrt(d)) / (2 * a);
        return t2 > 0.001 ? HitRecord{t2, this} : HitRecord{t1, this};
    }

    Vec n (const Vec& p) const { return (p - o) / (p - o).norm(); }

    UV uv (const Vec& p) const {
        Vec n = this -> n(p);
        Ftype theta = acos(-n.y());
        Ftype phi = atan2(-n.z(), n.x()) + M_PI;
        return UV(phi / 2.0 / M_PI, theta / M_PI);
    }

    BBox calcBBox() const {
        return BBox {o - r * Vec(1, 1, 1), o + r * Vec(1, 1, 1)};
    }
};

struct Triangle: Obj {
    Vec a, b, c, nor;

    Triangle(Vec _a, Vec _b, Vec _c, Vec _nor, Material *mat): Obj(mat), nor(_nor), a(_a), b(_b), c(_c) {
        nor /= nor.norm();
        if (dot(cross(b - a, c - a), nor) < 0) std::swap(b, c);
    }

    HitRecord hit (const Ray& ray) {
        Ftype t = dot(a - ray.o, nor) / dot(ray.p, nor), eps = -0.0001;
        Vec at = ray.o + ray.p * t;
        if ( dot(cross(b - a, at - a), nor) >= eps &&
             dot(cross(c - b, at - b), nor) >= eps &&
             dot(cross(a - c, at - c), nor) >= eps    ) return HitRecord{t, this};
        else return HitRecord{-1, this};
    }

    Vec n (const Vec& p) const { return nor; }

    UV uv (const Vec& p) const {
        return UV(0, 0); // TODO
    }

    BBox calcBBox() const {
        return BBox::merge(BBox{a, a}, BBox::merge(BBox{b, b}, BBox{c, c}));
    }
};

// broute force ray tracing
Color rayTrace (Ray ray, ObjList& objs, int depth = 50) {
    ray.p = ray.p / ray.p.norm();
    Ftype t = -1.0; Obj* cur = NULL;
    for (auto &obj: objs) {
        Ftype tmp = (obj -> hit(ray)).t;
        if (tmp > 0.001 && (tmp < t || cur == NULL))
            t = tmp, cur = obj; 
    }
    if (cur != NULL) {
        Vec at = ray.o + ray.p * t;
        Ray out, in {at, ray.p};
        auto col = cur -> mat -> scatter(cur, in, out);
        if (depth > 0) return col(rayTrace(out, objs, depth - 1));
        else return Color(0, 0, 0);
    }
    t = 0.5 * (ray.p.y() + 1.0);
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
}

// ray tracing using Bounding Volumes Hierarchies
Color rayTrace (Ray ray, BVH& bvh, int depth = 50) {
    ray.p = ray.p / ray.p.norm();
    HitRecord rec = bvh.hit(ray);
    if (rec.obj == NULL){
        Ftype t = 0.5 * (ray.p.y() + 1.0);
        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0);
    }
    Vec at = ray.o + ray.p * rec.t;
    Ray out, in {at, ray.p};
    auto col = rec.obj -> mat -> scatter(rec.obj, in, out);
    if (depth > 0) return col(rayTrace(out, bvh, depth - 1));
    else return Color(0, 0, 0);
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

void toPPM (Color **img, int r, int c, std::string fname = "out.ppm") {
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
        printf("\rRendering: [");
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

void readBinarySTL(std::string filename, ObjList& objs, Material* mat) {
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
                mat
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
    std::map<std::string, Texture*> textures;
    std::map<std::string, Material*> materials;

    // Textures
    textures["pink"] = new SolidColor(Color(0.8, 0.3, 0.3));
    textures["brown"] = new SolidColor(Color(0.8, 0.6, 0.2));
    textures["green"] = new SolidColor(Color(0.8, 0.8, 0.0));
    textures["purple"] = new SolidColor(Color(0.3, 0.2, 0.8));
    textures["checker"] = new CheckerTexture(textures["green"], textures["brown"]);
    textures["noise"] = new NoiseTexture();

    // Materials
    materials["diffuse-pink"] = new Diffuse(textures["pink"]);
    materials["diffuse-green"] = new Diffuse(textures["green"]);
    materials["diffuse-purple"] = new Diffuse(textures["purple"]);
    materials["diffuse-checker"] = new Diffuse(textures["checker"]);
    materials["metal"] = new Metal(textures["purple"]);
    materials["glass-outer"] = new Glass(2.0/3.0);
    materials["glass-inner"] = new Glass(1.5);
    materials["diffuse-noise"] = new Diffuse(textures["noise"]);

    // Scens
    ObjList objs;
    Camera cam;
    //
    switch(4) {
        // scene: tree-balls
        case 1:
            cam = Camera {
                Vec (0, 1, 0),
                Vec (-1.25, 0.8, 0.6),
                Vec (0, 0, -1),
                90, 16.0 / 9.0
            };
            objs.push_back(new Sphere {Vec(0, 0, -1), 0.5, materials["diffuse-pink"]});
            objs.push_back(new Sphere {Vec(1, 0, -1), 0.5, materials["metal"]});
            objs.push_back(new Sphere {Vec(-1, 0, -1), 0.5, materials["glass-outer"]});
            objs.push_back(new Sphere {Vec(-1, 0, -1), 0.45, materials["glass-inner"]});
            objs.push_back(new Sphere {Vec(0, -100.5, -1), 100, materials["diffuse-checker"]});
            break;

        // scene: little-witch
        case 2:
            readBinarySTL("little-witch.stl", objs, materials["metal"]);
            objs.push_back(new Sphere {Vec(0, 0, -10000.5), 10000, materials["diffuse-green"]});
            cam = Camera {
                Vec (0, 0, 1),
                Vec (-1, -30, 25),
                Vec (-1.726, 19.175, 18.6763),
                90, 16.0 / 9.0
            };
            break;

        // scene: two perlin shperes
        case 3:
            cam = Camera {
                Vec (0, 1, 0),
                Vec (-1.25, 0.8, 0.6),
                Vec (0, 0, -1),
                90, 16.0 / 9.0
            };
            objs.push_back(new Sphere {Vec(0, 0, -1), 0.5, materials["diffuse-noise"]});
            objs.push_back(new Sphere {Vec(1, 0, -1), 0.5, materials["metal"]});
            objs.push_back(new Sphere {Vec(-1, 0, -1), 0.5, materials["glass-outer"]});
            objs.push_back(new Sphere {Vec(-1, 0, -1), 0.45, materials["glass-inner"]});
            objs.push_back(new Sphere {Vec(0, -100.5, -1), 100, materials["diffuse-checker"]});
            break;
        case 4:
            textures["earth"] = new ImageTexture("earthmap.jpg");
            materials["diffuse-earth"] = new Diffuse(textures["earth"]);
            objs.push_back(new Sphere {Vec(0, 0, 0), 2, materials["diffuse-earth"]});
            cam = Camera {
                Vec (0, 1, 0),
                Vec (13, 2, 3),
                Vec (0, 0, 0),
                20, 16.0 / 9.0
            };
            break;
    }

    BVH bvh(objs);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            img[i][j] = Color(0, 0, 0);
            for (int s = 0; s < 10; s++)
                img[i][j] += rayTrace(cam.rayAt(Ftype(i) + drand48(), Ftype(j) + drand48(), r, c), bvh, 20);
            img[i][j] /= 10.0;
            img[i][j] = Color(sqrt(img[i][j][0]), sqrt(img[i][j][1]), sqrt(img[i][j][2]));
            img[i][j] = img[i][j] * 255.99;
            progress.update(float(i * c + j) / float(r * c));
        }
    }
    toPPM(img, r, c);
    deleteImg(img, r);
    for (auto obj: objs) delete obj;
    for (auto tp: materials) delete tp.second;
    for (auto tp: textures) delete tp.second;
    return 0;
}

