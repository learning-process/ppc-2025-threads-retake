#include <cmath>

#include "stl/veliev_e_montecarlo/include/my_funcs.hpp"

double veliev_func_stl::Flin(double x, double y) { return x + y; }
double veliev_func_stl::FsinxPsiny(double x, double y) { return sin(x) + sin(y); }
double veliev_func_stl::FcosxPcosy(double x, double y) { return cos(x) + cos(y); }
double veliev_func_stl::Fxy(double x, double y) { return x * y; }
double veliev_func_stl::Fxyy(double x, double y) { return x * y * y; }
