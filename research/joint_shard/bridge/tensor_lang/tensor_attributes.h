#ifndef RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_ATTRIBUTES_H_
#define RESEARCH_JOINT_SHARD_BRIDGE_TENSOR_LANG_TENSOR_ATTRIBUTES_H_

#include <functional>
#include <type_traits>

#include "llvm/Support/raw_ostream.h"
#include "research/joint_shard/bridge/tensor_lang/tensorlang.h"

namespace joint_shard {
// These algorithms recurse through schema records; adding an attribute field
// automatically includes it in node identity and diagnostic/binding text.
template <class T>
void printAttributeValue(llvm::raw_ostream& out, const T& value) {
  if constexpr (requires { AttributeSchema<T>::members; }) {
    out << '{';
    visitAttributeFields(value, [&](auto field, const auto& member) {
      out << field.name << '=';
      printAttributeValue(out, member);
      out << ';';
    });
    out << '}';
  } else if constexpr (std::is_same_v<T, Axes>) {
    out << '[';
    for (auto axis : value) out << axis << ',';
    out << ']';
  } else if constexpr (std::is_enum_v<T>) {
    out << static_cast<std::underlying_type_t<T>>(value);
  } else if constexpr (requires { value.getAsOpaquePointer(); }) {
    if (value)
      out << value;
    else
      out << "none";
  } else {
    out << value;
  }
}
template <class T>
std::string formatAttributeValue(const T& value) {
  std::string text;
  llvm::raw_string_ostream out(text);
  printAttributeValue(out, value);
  return text;
}
template <class T>
void hashAttributeValue(std::size_t& hash, const T& value) {
  if constexpr (requires { AttributeSchema<T>::members; }) {
    visitAttributeFields(value, [&](auto, const auto& member) {
      hashAttributeValue(hash, member);
    });
  } else if constexpr (std::is_same_v<T, Axes>) {
    eggc::hash_combine(hash, value.size());
    for (auto axis : value) hashAttributeValue(hash, axis);
  } else if constexpr (requires { value.getAsOpaquePointer(); }) {
    eggc::hash_combine(hash, std::hash<const void*>{}(
                                 value ? value.getAsOpaquePointer() : nullptr));
  } else {
    eggc::hash_combine(hash, std::hash<T>{}(value));
  }
}
}  // namespace joint_shard
#endif
