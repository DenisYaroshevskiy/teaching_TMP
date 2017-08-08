
#include <cstddef>
#include <type_traits>
#include <tuple>
#include <utility>

// algorithms ----------------------------------------------------------------

template <typename I1, typename I2>
constexpr bool equal(I1 f1, I1 l1, I2 f2) {
  while (f1 != l1) {
    if (!(*f1++ == *f2++))
      return false;
  }
  return true;
}

// cpp17_array ---------------------------------------------------------------

template <typename T, size_t N>
struct cpp17_array {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;
  // todo reverse

  value_type body_[N];

  constexpr reference operator[](size_type pos) { return body_[pos]; }

  constexpr const_reference operator[](size_type pos) const {
    return body_[pos];
  }

  constexpr size_type size() const noexcept { return N; }

  constexpr iterator begin() { return body_; }
  constexpr const_iterator begin() const { return body_; }
  constexpr const_iterator cbegin() const { return body_; }

  constexpr iterator end() { return begin() + size(); }
  constexpr const_iterator end() const { return begin() + size(); }
  constexpr const_iterator cend() const { return begin() + size(); }

  friend constexpr bool operator==(const cpp17_array& x, const cpp17_array& y) {
    return equal(x.begin(), x.end(), y.begin());
  }

  friend constexpr bool operator!=(const cpp17_array& x, const cpp17_array& y) {
    return !(x == y);
  }
};

// compile time indexing ------------------------------------------------------

template <size_t idx>
using index_constant = std::index_sequence<idx>;

template <size_t... idxs1, size_t... idxs2>
constexpr std::index_sequence<idxs1..., idxs2...> concut(
    std::index_sequence<idxs1...>,
    std::index_sequence<idxs2...>) {
  return {};
}

// array transform ------------------------------------------------------------

template <typename T, size_t N, typename F, size_t... idxs>
constexpr auto cpp_array_transfrom_impl(cpp17_array<T, N> in,
                                        F f,
                                        std::index_sequence<idxs...>) {
  static_assert(N == sizeof...(idxs), "we generate indexes for each element");

  // TODO: more return type processing.
  using r_type = decltype(f(index_constant<0>{}, std::declval<T>()));

  return cpp17_array<r_type, N>{{f(index_constant<idxs>{}, in[idxs])...}};
}

template <typename T, size_t N, typename F>
constexpr auto cpp_array_transfrom(cpp17_array<T, N> in, F f) {
  return cpp_array_transfrom_impl(in, f, std::make_index_sequence<N>{});
}

// static tables --------------------------------------------------------------

template <typename T, size_t... dims>
struct static_table;

template <typename T, size_t... dims>
using static_table_t = typename static_table<T, dims...>::type;

template <typename T, size_t dim>
struct static_table<T, dim> {
  using value_type = T;
  using type = cpp17_array<T, dim>;
  using dims_sequence = std::index_sequence<dim>;

  template <typename F, size_t ...previous_idxs>
  struct apply_impl {
    F f;

    template <size_t idx>
    constexpr auto operator()(index_constant<idx> i, T x) {
      return f(concut(std::index_sequence<previous_idxs...>{}, i), x);
    }
  };

  template <typename F, size_t... previous_idxs>
  static constexpr auto apply(type arr,
                              F f,
                              std::index_sequence<previous_idxs...>) {
    return cpp_array_transfrom(arr, apply_impl<F, previous_idxs...>{f});
  }
};

template <typename T, size_t dim, size_t... dims>
struct static_table<T, dim, dims...> {
  using value_type = T;
  using smaller_table = static_table<T, dims...>;
  using smaller_table_t = typename smaller_table::type;
  using type = cpp17_array<smaller_table_t, dim>;
  using dims_sequence = std::index_sequence<dim, dims...>;

  template <typename F, size_t ...previous_idxs>
  struct apply_impl {
    F f;

    template <size_t idx>
    constexpr auto operator()(index_constant<idx> i,
                              smaller_table_t smaller) const {
      return smaller_table::apply(
          smaller, f, concut(std::index_sequence<previous_idxs...>{}, i));
    }
  };

  template <typename F, size_t... previous_idxs>
  static constexpr auto apply(type arr,
                              F f,
                              std::index_sequence<previous_idxs...>) {
    return cpp_array_transfrom(arr, apply_impl<F, previous_idxs...>{f});
  }
};

template <typename T, size_t... idxs>
constexpr static_table<T, idxs...> make_static_table_from_index_sequence(
    std::index_sequence<idxs...>) {
  return {};
}

constexpr std::false_type is_static_table_impl(...) {
  return {};
}
template <typename T, size_t dim>
constexpr std::true_type is_static_table_impl(const cpp17_array<T, dim>&) {
  return {};
}

template <typename T>
constexpr bool is_static_table_v =
    decltype(is_static_table_impl(std::declval<T>()))::value;

template <typename T,
          size_t dim,
          typename = std::enable_if_t<!is_static_table_v<T>>>
constexpr static_table<T, dim> detect_static_table(const cpp17_array<T, dim>&) {
  return {};
}

template <typename T,
          size_t dim,
          typename = std::enable_if_t<is_static_table_v<T>>>
constexpr auto detect_static_table(const cpp17_array<T, dim>& table) {
  using smaller_table = decltype(detect_static_table(table[0]));
  using value_type = typename smaller_table::value_type;
  using tail_dims = typename smaller_table::dims_sequence;
  return make_static_table_from_index_sequence<value_type>(
      concut(index_constant<dim>{}, tail_dims{}));
}

template <typename T, size_t dim, typename F>
constexpr auto transfrom_static_table(cpp17_array<T, dim> table, F f) {
  return decltype(detect_static_table(table))::apply(table, f,
                                                     std::index_sequence<>{});
}

// tests ---------------------------------------------------------------------

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

int main() {
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
