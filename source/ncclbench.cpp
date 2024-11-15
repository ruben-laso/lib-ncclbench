#include <string>

#include "ncclbench/ncclbench.hpp"

exported_class::exported_class()
    : m_name {"ncclbench"}
{
}

auto exported_class::name() const -> char const*
{
  return m_name.c_str();
}
