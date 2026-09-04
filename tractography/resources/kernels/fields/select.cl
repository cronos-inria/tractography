#ifndef __FIELD_SELECT__
#define __FIELD_SELECT__

#if defined(FIELD_TETRAHEDRAL_MESH)
#include "fields/mesh.cl"
#elif defined(FIELD_IMAGE)
#include "fields/image.cl"
#endif

#endif
