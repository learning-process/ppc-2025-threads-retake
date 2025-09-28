#include <cmath>

#include "all/veliev_e_montecarlo/include/my_funcs.hpp"

double veliev_func_all::Flin(double x, double y) { return x + y; }
double veliev_func_all::FsinxPsiny(double x, double y) { return sin(x) + sin(y); }
double veliev_func_all::FcosxPcosy(double x, double y) { return cos(x) + cos(y); }
double veliev_func_all::Fxy(double x, double y) { return x * y; }
double veliev_func_all::Fxyy(double x, double y) { return x * y * y; }
