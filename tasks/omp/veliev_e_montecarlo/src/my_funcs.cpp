#include <cmath>

#include "omp/veliev_e_montecarlo/include/my_funcs.hpp"

double veliev_func_omp::Flin(double x, double y) { return x + y; }
double veliev_func_omp::FsinxPsiny(double x, double y) { return sin(x) + sin(y); }
double veliev_func_omp::FcosxPcosy(double x, double y) { return cos(x) + cos(y); }
double veliev_func_omp::Fxy(double x, double y) { return x * y; }
double veliev_func_omp::Fxyy(double x, double y) { return x * y * y; }
