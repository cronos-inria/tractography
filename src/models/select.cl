#ifndef __MODEL_SELECT__
#define __MODEL_SELECT__

#if defined(MODEL_SYMMETRIC_REAL_SPHERICAL_HARMONICS)
#include "models/symmetric-real-spherical-harmonics.cl"
#elif defined(MODEL_DTI)
#include "models/diffusion-tensor-imaging.cl"
#endif

#endif