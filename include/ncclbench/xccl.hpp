#pragma once

#if defined(USE_RCCL)
#include <rccl/rccl.h>
#else // USE_NCCL by default
#include <nccl.h>
#endif