#include <string>

#include "ncclbench/ncclbench.hpp"

auto main() -> int
{
  auto const exported = exported_class {};

  return std::string("ncclbench") == exported.name() ? 0 : 1;
}
