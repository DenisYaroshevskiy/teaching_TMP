
#include <cstddef>
#include <type_traits>
#include <utility>

namespace hogwarts {

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

namespace detail {

template <typename... Ts>
struct variant_impl {
  // TODO: we can select variant indx_t better.
  using indx_t = short;

  // type meta functions -----------------------------------------------------

  // private -----------------------------------------------------------------

  template <indx_t idx_, typename T>
  struct type_info {
    using type = T;
    static constexpr indx_t idx = idx_;
  };

  template <size_t... idxs>
  static constexpr auto build_indexed_type_infos_impl(
      std::index_sequence<idxs...>) {
    struct result_type : type_info<static_cast<indx_t>(idxs), Ts>... {};
    return result_type{};
  }

  using all_type_info = decltype(variant_impl::build_indexed_type_infos_impl(
      std::index_sequence_for<Ts...>{}));

  template <typename T, indx_t idx>
  static constexpr auto idx_4_type_impl(const type_info<idx, T>& info) {
    return info;
  }

  template <indx_t idx, typename T>
  static constexpr auto type_4_idx_impl(const type_info<idx, T>& info) {
    return info;
  }

  // public: -----------------------------------------------------------------

  template <typename T>
  static constexpr indx_t idx_4_type_v =
      decltype(indx_4_type_impl(all_type_info{}))::idx;

  template <indx_t idx>
  using type_4_idx_t =
      typename decltype(type_4_idx_impl<idx>(all_type_info{}))::type;

  // memory -----------------------------------------------------------------
  using memory_t = std::aligned_union_t<0u, Ts...>;

  template <typename T>
  static T& cast(memory_t& mem) noexcept {
    return reinterpret_cast<T&>(mem);
  }

  template <typename T>
  static const T& cast(const memory_t& mem) noexcept {
    return reinterpret_cast<const T&>(mem);
  }

  template <typename T, typename... Args>
  static void construct(memory_t& mem, Args&&... args) noexcept(T{
      std::forward<Args>(args)...}) {
    new (&cast<T>(mem)) T{std::forward<Args>(args)...};
  }
};

}  // namespace detail

template <typename... Ts>
class variant {
  using meta = detail::variant_impl<Ts...>;
  using indx_t = typename meta::indx_t;

  template <indx_t idx>
  using type_4_idx_t = typename meta::template type_4_idx_t<idx>;

  template <typename T>
  static constexpr indx_t idx_4_type_v = meta::template idx_4_type_v<T>;

  using memory_t = typename meta::memory_t;

  memory_t mem_;
  indx_t idx_;

 public:
  variant() noexcept(type_4_idx_t<0>{}) {
    meta::template construct<type_4_idx_t<0>>(mem_);
    idx_ = 0;
  }

  template <typename T>
  variant(T&& rhs) noexcept(T(std::forward<T>(rhs))) {
    meta::template construct<T>(mem_, std::forward<T>(rhs));
    idx_ = idx_4_type_v<T>;
  }
};

}  // namespace hogwarts
