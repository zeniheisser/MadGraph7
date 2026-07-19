#pragma once

#include <pybind11/pybind11.h>

// Binds madtrex::TupperWare/WareHouse (the MatrixElementApi <-> REX::tea
// reweighting bridge) into m. Declared separately from madspace.cpp, mirroring
// how Rex/src/python/rex.cpp calls into tearex.cpp's bind_tearex, so the
// (fairly large) TupperWare/WareHouse binding surface doesn't grow the main
// madspace.cpp module file further.
void bind_madtrex(pybind11::module_ m);
