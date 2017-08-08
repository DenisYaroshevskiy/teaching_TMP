#define CATCH_CONFIG_MAIN
#include "catch.h"

#include "variant.h"

namespace static_table_tests {

using namespace hogwarts;

// Compiler will show what X & Y are unpacked.
template <typename X, typename Y>
constexpr void is_same_test() {
  static_assert(std::is_same<X, Y>::value, "");
}

struct transfrom_array_test_action {
  template <size_t idx>
  constexpr long operator()(index_constant<idx>, int x) {
    return x + idx;
  }
};

struct set_zero {
  template <size_t... idxs>
  constexpr int operator()(std::index_sequence<idxs...>, int) const {
    return 0;
  }
};

struct full_test {
  template <size_t idx1, size_t idx2, size_t idx3>
  constexpr size_t operator()(std::index_sequence<idx1, idx2, idx3>,
                              size_t x) const {
    return x + idx1 * 100 + idx2 * 10 + idx3;
  }
};

constexpr bool check_full_test(const static_table_t<size_t, 3, 4, 5>& table) {
  for (size_t i = 0; i < table.size(); ++i) {
    for (size_t j = 0; j < table[i].size(); ++j) {
      for (size_t k = 0; k < table[i][j].size(); ++k) {
        if (table[i][j][k] != i * 100 + j * 10 + k)
          throw i + j + k;
      }
    }
  }
  return true;
}

}  // namespace static_table_tests

TEST_CASE("static_table", "[variant]") {
  using namespace static_table_tests;
  {
    is_same_test<static_table_t<int, 3>, cpp17_array<int, 3>>();
    is_same_test<static_table_t<int, 3, 4>,
                 cpp17_array<cpp17_array<int, 4>, 3>>();
  }
  {
    constexpr cpp17_array<int, 3> before{{1, 1, 1}};

    constexpr cpp17_array<long, 3> after =
        cpp_array_transfrom(before, transfrom_array_test_action{});

    constexpr cpp17_array<long, 3> expected_after = {{1, 2, 3}};
    static_assert(after == expected_after, "");
  }
  {
    constexpr static_table_t<int, 3> t1{};
    is_same_test<decltype(detect_static_table(t1)), static_table<int, 3>>();

    constexpr static_table_t<int, 3, 4> t2{};
    is_same_test<decltype(detect_static_table(t2)), static_table<int, 3, 4>>();

    constexpr static_table_t<int, 3, 4, 5> t3{};
    is_same_test<decltype(detect_static_table(t3)),
                 static_table<int, 3, 4, 5>>();
  }
  {
    constexpr int x = set_zero{}(std::index_sequence<1, 2, 4>{}, 3);
    static_assert(x == 0, "");
  }
  {
    constexpr static_table_t<int, 3> step1{{1, 1, 1}};
    constexpr auto step2 = transfrom_static_table(step1, set_zero{});

    constexpr cpp17_array<int, 3> expected = {{0, 0, 0}};
    static_assert(step2 == expected, "");
  }
  {
    constexpr std::index_sequence<1, 2, 3> x{};
    constexpr std::index_sequence<4, 5> y{};
    is_same_test<decltype(concut(x, y)), std::index_sequence<1, 2, 3, 4, 5>>();
    is_same_test<decltype(concut(x, index_constant<4>{})),
                 std::index_sequence<1, 2, 3, 4>>();
    is_same_test<decltype(concut(index_constant<3>{}, y)),
                 std::index_sequence<3, 4, 5>>();
  }
  {
    constexpr static_table_t<int, 3, 4> step1{{
        {{1, 1, 1, 1}},  //
        {{1, 1, 1, 1}},  //
        {{1, 1, 1, 1}},  //
    }};
    constexpr auto step2 = transfrom_static_table(step1, set_zero{});

    constexpr static_table_t<int, 3, 4> expected = {{
        {{0, 0, 0, 0}},  //
        {{0, 0, 0, 0}},  //
        {{0, 0, 0, 0}},  //
    }};

    static_assert(step2 == expected, "");
  }
  {
    constexpr static_table_t<int, 3, 4, 5> step1{};
    constexpr auto step2 = transfrom_static_table(step1, set_zero{});
    constexpr auto step3 = transfrom_static_table(step2, full_test{});
    static_assert(check_full_test(step3), "");
  }
}

TEST_CASE("variant default constructor", "[variant]") {
  hogwarts::variant<int, char> v;
}
