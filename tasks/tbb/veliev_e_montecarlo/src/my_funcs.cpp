#include "tbb/veliev_e_montecarlo/include/my_funcs.hpp"

#include <cmath>

double veliev_func_tbb::Flin(double x, double y) { return x + y; }
double veliev_func_tbb::FsinxPsiny(double x, double y) { return sin(x) + sin(y); }
double veliev_func_tbb::FcosxPcosy(double x, double y) { return cos(x) + cos(y); }
double veliev_func_tbb::Fxy(double x, double y) { return x * y; }
double veliev_func_tbb::Fxyy(double x, double y) { return x * y * y; }
