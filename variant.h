#include <cassert>
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

  template <typename F, size_t... previous_idxs>
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

  template <typename StaticTable, typename I>
  static constexpr decltype(auto) index_into_impl(StaticTable&& table,
                                                  I f_i,
                                                  I l_i) {
    assert(l_i - f_i == dims_sequence::size());
    return std::forward<StaticTable>(table)[*f_i];
  }
};

template <typename T, size_t dim, size_t... dims>
struct static_table<T, dim, dims...> {
  using value_type = T;
  using smaller_table = static_table<T, dims...>;
  using smaller_table_t = typename smaller_table::type;
  using type = cpp17_array<smaller_table_t, dim>;
  using dims_sequence = std::index_sequence<dim, dims...>;

  template <typename F, size_t... previous_idxs>
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

  template <typename StaticTable, typename I>
  static constexpr decltype(auto) index_into_impl(StaticTable&& table,
                                                  I f_i,
                                                  I l_i) {
    assert(l_i - f_i == dims_sequence::size());
    size_t i = *f_i++;
    return smaller_table::index_into_impl(std::forward<StaticTable>(table)[i],
                                          f_i, l_i);
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

template <typename T>
constexpr void detect_static_table_impl(const T&) = delete;

template <typename T,
          size_t dim,
          typename = std::enable_if_t<!is_static_table_v<T>>>
constexpr static_table<T, dim> detect_static_table_impl(
    const cpp17_array<T, dim>&) {
  return {};
}

template <typename T,
          size_t dim,
          typename = std::enable_if_t<is_static_table_v<T>>>
constexpr auto detect_static_table_impl(const cpp17_array<T, dim>& table) {
  using smaller_table = decltype(detect_static_table_impl(table[0]));
  using value_type = typename smaller_table::value_type;
  using tail_dims = typename smaller_table::dims_sequence;
  return make_static_table_from_index_sequence<value_type>(
      concut(index_constant<dim>{}, tail_dims{}));
}

template <typename StaticTable>
using detect_static_table_t =
    decltype(detect_static_table_impl(std::declval<StaticTable>()));

template <typename StaticTable, typename F>
constexpr auto transfrom_static_table(StaticTable table, F f) {
  return detect_static_table_t<StaticTable>::apply(table, f,
                                                   std::index_sequence<>{});
}

template <typename StaticTable>
using array_of_idxs_t =
    cpp17_array<size_t,
                detect_static_table_t<StaticTable>::dims_sequence::size()>;

template <typename StaticTable>
constexpr decltype(auto) index_into_static_table(
    StaticTable&& table,
    const array_of_idxs_t<StaticTable>& idxs) {
  return detect_static_table_t<StaticTable>::index_into_impl(
      std::forward<StaticTable>(table), idxs.begin(), idxs.end());
}

// variant_impl --------------------------------------------------------------

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
      decltype(idx_4_type_impl<T>(all_type_info{}))::idx;

  template <indx_t idx>
  using type_4_idx_t =
      typename decltype(type_4_idx_impl<idx>(all_type_info{}))::type;

  static constexpr indx_t type_count_v = sizeof...(Ts);

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
  static void construct(memory_t& mem, Args&&... args) {
    new (&cast<T>(mem)) T{std::forward<Args>(args)...};
  }

  // later friends.
  struct dependent;
};

}  // namespace detail

template <typename... Ts>
class variant {
  using meta = detail::variant_impl<Ts...>;
  friend meta;

  using indx_t = typename meta::indx_t;

  template <indx_t idx>
  using type_4_idx_t = typename meta::template type_4_idx_t<idx>;

  template <typename T>
  static constexpr indx_t idx_4_type_v = meta::template idx_4_type_v<T>;

  using memory_t = typename meta::memory_t;

  memory_t mem_;
  indx_t idx_;

 public:
  variant() {
    meta::template construct<type_4_idx_t<0>>(mem_);
    idx_ = 0;
  }

  template <typename T>
  variant(T&& rhs) {
    meta::template construct<T>(mem_, std::forward<T>(rhs));
    idx_ = idx_4_type_v<T>;
  }
};

namespace detail {

template <typename... Ts>
struct variant_impl<Ts...>::dependent {
  template <typename T, typename V>
  static decltype(auto) get(V&& v) {
    assert(v.idx_ == idx_4_type_v<T>);
    return cast<T>(std::forward<V>(v).mem_);
  }

  template <size_t idx_, typename V>
  static decltype(auto) get(V&& v) {
    constexpr auto idx = static_cast<indx_t>(idx_);
    assert(v.idx_ == idx);
    return cast<type_4_idx_t<idx>>(std::forward<V>(v).mem_);
  }

  static indx_t get_active_idx(const variant<Ts...>& v) { return v.idx_; }
};

// declarations are enough for decltype
std::false_type is_variant_impl(...);

template <typename... Ts>
std::true_type is_variant_impl(const variant<Ts...>&);

template <typename... Ts>
variant_impl<Ts...> variant_meta_impl(const variant<Ts...>&);

template <typename V>
using variant_meta_t = decltype(variant_meta_impl(std::declval<V>()));

template <typename V>
using varinat_meta_dependent_t = typename variant_meta_t<V>::dependent;

}  // namespace detail

template <typename T>
constexpr bool is_variant_v =
    decltype(detail::is_variant_impl(std::declval<T>()))::value;

template <typename T, typename V, typename = std::enable_if_t<is_variant_v<V>>>
decltype(auto) get(V&& v) {
  return detail::varinat_meta_dependent_t<V>::template get<T>(
      std::forward<V>(v));
}

template <std::size_t idx,
          typename V,
          typename = std::enable_if_t<is_variant_v<V>>>
decltype(auto) get(V&& v) {
  return detail::varinat_meta_dependent_t<V>::template get<idx>(
      std::forward<V>(v));
}

namespace detail {

template <typename F, typename... Vs>
using visit_return_type_t =
    decltype(std::declval<F>()(get<0>(std::declval<Vs>())...));

template <typename F, typename... Vs>
using vtable_entry_t = visit_return_type_t<F, Vs...>(*)(F, Vs...);

template <typename V>
constexpr std::size_t variant_type_count_v =
    static_cast<std::size_t>(variant_meta_t<V>::type_count_v);

template <typename F, typename... Vs>
struct build_vtable_transform {
  template <size_t... idxs>
  constexpr vtable_entry_t<F, Vs...> operator()(
      std::index_sequence<idxs...>,
      vtable_entry_t<F, Vs...>) const {
    return [](F f, Vs... vs) -> visit_return_type_t<F, Vs...> {
      return std::forward<F>(f)(get<idxs>(std::forward<Vs>(vs))...);
    };
  }
};

template <typename F, typename... Vs>
using vtable_t =
    static_table_t<vtable_entry_t<F, Vs...>, variant_type_count_v<Vs>...>;

// F and Vs are already with correct l/rvalue references.
template <typename F, typename... Vs>
constexpr auto build_vtable() {
  return transfrom_static_table(vtable_t<F, Vs...>{},
                                build_vtable_transform<F, Vs...>{});
}

template <typename F, typename... Vs>
decltype(auto) visit_with_return_value(F&& f, Vs&&... vs) {
  static const auto vtable = build_vtable<decltype(std::forward<F>(f)),
                                          decltype(std::forward<Vs>(vs))...>();
  auto found_entry = index_into_static_table(
      vtable, {{static_cast<size_t>(
                  varinat_meta_dependent_t<Vs>::get_active_idx(vs))...}});

  return found_entry(std::forward<F>(f), std::forward<Vs>(vs)...);
}

}  // namespace detail

template <typename F, typename... Vs>
decltype(auto) visit(F&& f, Vs&&... vs) {
  using return_t =
      detail::visit_return_type_t<decltype(std::forward<F>(f)),
                                  decltype(std::forward<Vs>(vs))...>;
  if (!std::is_same<return_t, void>::value)
    return detail::visit_with_return_value(std::forward<F>(f),
                                           std::forward<Vs>(vs)...);

  detail::visit_with_return_value(
      [&](auto&&... unwrapped_vs) {
        std::forward<F>(f)(
            std::forward<decltype(unwrapped_vs)>(unwrapped_vs)...);
        return 0;
      },
      std::forward<Vs>(vs)...);
}

}  // namespace hogwarts
