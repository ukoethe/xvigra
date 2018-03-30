/************************************************************************/
/*                                                                      */
/*     Copyright 2017-2018 by Ullrich Koethe                            */
/*                                                                      */
/*    This file is part of the XVIGRA image analysis library.           */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#pragma once

#ifndef XVIGRA_ARRAY_ND_HPP
#define XVIGRA_ARRAY_ND_HPP

#include <utility>
#include <numeric>
#include <algorithm>
#include <xtensor/xtensor.hpp>
#include <xtensor/xcontainer.hpp>
#include <xtensor/xsemantic.hpp>
#include <xtensor/xeval.hpp>

#include "global.hpp"
#include "concepts.hpp"
#include "error.hpp"
#include "math.hpp"
#include "tiny_vector.hpp"
#include "slice.hpp"

// Bounds checking Macro used if VIGRA_CHECK_BOUNDS is defined.
#ifdef XVIGRA_CHECK_BOUNDS
#define XVIGRA_ASSERT_INSIDE(diff) \
  vigra_precondition(this->is_inside(diff), "Index out of bounds")
#else
#define XVIGRA_ASSERT_INSIDE(diff)
#endif

/************************************************/
/* interoperability with xtensor's xexpressions */
/************************************************/

namespace xt
{
    namespace detail
    {
        template <class V, xvigra::index_t L, class R>
        struct index_type_impl<xvigra::tiny_vector<V, L, R>>
        {
            using type = xvigra::tiny_vector<V, L, R>;
        };

        template <class T, xvigra::index_t N, class A>
        struct is_xexpression_impl<xvigra::array_nd<T, N, A>> : std::true_type
        {
        };
    }

    template <class T, xvigra::index_t N>
    struct xcontainer_inner_types<xvigra::view_nd<T, N>>
    {
        using temporary_type = xvigra::array_nd<T, N>;
    };

    template <class T, xvigra::index_t N>
    struct xiterable_inner_types<xvigra::view_nd<T, N>>
    {
        using inner_shape_type = xvigra::shape_t<N>;
        using stepper = xindexed_stepper<xvigra::view_nd<T, N>, std::is_const<T>::value>;
        using const_stepper = xindexed_stepper<xvigra::view_nd<T, N>, true>;
    };
}

namespace xvigra
{
    namespace tags
    {
            // Tags to assign semantic meaning to axes.
            // (arranged in sorting order)
        enum axis_tag  { axis_missing = -1,
                         axis_unknown = 0,
                         axis_c,  // channel axis
                         axis_n,  // node map for a graph
                         axis_x,  // spatial x-axis
                         axis_y,  // spatial y-axis
                         axis_z,  // spatial z-axis
                         axis_t,  // time axis
                         axis_fx, // Fourier transform of x-axis
                         axis_fy, // Fourier transform of y-axis
                         axis_fz, // Fourier transform of z-axis
                         axis_ft, // Fourier transform of t-axis
                         axis_e,  // edge map for a graph
                         axis_end // marker for the end of the list
                       };


        //     // Support for tags::axis keyword argument to select
        //     // the axis an algorithm is supposed to operator on
        // struct axis_selection_proxy
        // {
        //     int value;
        // };

        // struct axis_selection_tag
        // {
        //     axis_selection_proxy operator=(int i) const
        //     {
        //         return {i};
        //     }

        //     axis_selection_proxy operator()(int i) const
        //     {
        //         return {i};
        //     }
        // };

        // namespace
        // {
        //     axis_selection_tag axis;
        // }

            // Support for tags::byte_strides keyword argument
            // to pass strides in units of bytes rather than `sizeof(T)`.
        // template <int N>
        // struct byte_strides_proxy
        // {
        //     tiny_vector<index_t, N> value;
        // };

        // struct byte_strides_tag
        // {
        //     template <index_t N, class R>
        //     byte_strides_proxy<N> operator=(tiny_vector<index_t, N, R> const & s) const
        //     {
        //         return {s};
        //     }

        //     template <index_t N, class R>
        //     byte_strides_proxy<N> operator()(tiny_vector<index_t, N, R> const & s) const
        //     {
        //         return {s};
        //     }
        // };

        // namespace
        // {
        //     byte_strides_tag byte_strides;
        // }

    } // namespace tags

    template <index_t N=runtime_size>
    using axis_tags = tiny_vector<tags::axis_tag, N>;

    inline auto
    default_axistags(index_t N, bool with_channels = false, tags::memory_order order = c_order)
    {
        static const tags::axis_tag std[] = { tags::axis_t,
                                              tags::axis_z,
                                              tags::axis_y,
                                              tags::axis_x,
                                              tags::axis_c };
        int count = with_channels ? 5 : 4;
        vigra_precondition(0 <= N && N <= count,
            "default_axistags(): only defined for up to five dimensions.");
        axis_tags<> res(std + count - N, std + count);
        return order == c_order
                  ? res
                  : reversed(res);
    }

    namespace detail
    {
        template <index_t N>
        inline auto
        permutation_to_order(shape_t<N> const & stride, tags::memory_order order)
        {
            auto res = shape_t<N>::range((index_t)stride.size());
            if(order == f_order)
            {
                std::sort(res.begin(), res.end(),
                         [stride](index_t l, index_t r)
                         {
                            if(stride[l] == 0 || stride[r] == 0)
                            {
                                return stride[r] < stride[l];
                            }
                            return stride[l] < stride[r];
                         });
            }
            else
            {
                std::sort(res.begin(), res.end(),
                         [stride](index_t l, index_t r)
                         {
                            if(stride[l] == 0 || stride[r] == 0)
                            {
                                return stride[l] < stride[r];
                            }
                            return stride[r] < stride[l];
                         });
            }
            return res;
        }

        struct overlapping_memory_checker
        {
            char * begin, * end;

            template <class T>
            overlapping_memory_checker(T * b, T * e)
            : begin((char*)(b <= e ? b : e))
            , end((char*)(b <= e ? e : b))
            {}

                // note: last points to the last element _in_ the range
            template <class T>
            bool check(T * first, T * last) const
            {
                if(first <= last)
                {
                    return (char*)first < end && begin < (char*)(last+1);
                }
                else
                {
                    return (char*)(last-1) < end && begin < (char*)first;
                }
            }

            template <class T>
            bool check(T * first) const
            {
                return check(first, first);
            }

            template<std::size_t I = 0, class... T>
            std::enable_if_t<(I == sizeof...(T)), bool>
            check_tuple(std::tuple<T...> const &) const
            {
                return false;
            }

            template<std::size_t I = 0, class... T>
            std::enable_if_t<(I < sizeof...(T)), bool>
            check_tuple(std::tuple<T...> const & t) const
            {
                return (*this)(std::get<I>(t)) || check_tuple<I+1>(t);
            }

            template <class F, class R, class... CT>
            bool operator()(xt::xfunction_base<F, R, CT ...> const & f) const
            {
                return check_tuple(f.arguments());
            }

            template <class T, index_t N>
            bool operator()(view_nd<T, N> const & v) const
            {
                return check(&v(), &v[v.shape()-1]);
            }

            template <class T>
            bool operator()(xt::xcontainer<T> const & v) const
            {
                return check(&v(), &*v.crbegin());
            }

            template <class T>
            bool operator()(xt::xscalar<T> const & v) const
            {
                return check(&v());
            }

            template <class T, class... S>
            bool
            operator()(xt::xview<T, S...> const & v) const
            {
                if(has_raw_data_api<xt::xview<T, S...>>::value)
                {
                    return check(&v(), &*v.crbegin());
                }
                else
                {
                    return true; // always create a temporary for views over xexpressions
                }
            }

            template <class T, class S, class U>
            bool
            operator()(xt::xstrided_view<T, S, U> const & v) const
            {
                if(has_raw_data_api<xt::xstrided_view<T, S, U>>::value)
                {
                    return check(&v(), &*v.crbegin());
                }
                else
                {
                    return true; // always create a temporary for views over xexpressions
                }
            }

            // FIXME: improve overlapping_memory_checker for views over xexpressions
        };
    }

    /***********/
    /* view_nd */
    /***********/

    template <class T, index_t N>
    class view_nd
    : public xt::xiterable<view_nd<T, N>>
    , public xt::xview_semantic<view_nd<T, N>>
    , public tags::view_nd_tag
    {

      public:

        enum flags_t { contiguous_memory_flag = 1,
                       owns_memory_flag = 2
                     };

            /** the array's internal dimensionality.
                This ensures that view_nd can also be used for
                scalars (that is, when <tt>N == 0</tt>). Calculated as:<br>
                \code
                internal_dimension = (N==0) ? 1 : N
                \endcode
             */
        static constexpr index_t internal_dimension = (N==0) ? 1 : N;
        static constexpr index_t ndim = N;

        using value_type             = T;
        using const_value_type       = typename std::add_const<T>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;
        using shape_type             = shape_t<internal_dimension>;
        using strides_type           = shape_type;
        using axistags_type          = tiny_vector<tags::axis_tag, internal_dimension>;

        using self_type = view_nd<T, N>;
        using view_type = self_type;
        using semantic_base = xt::xview_semantic<self_type>;

        using inner_shape_type = shape_type;
        using inner_strides_type = inner_shape_type;

        using iterable_base = xt::xiterable<self_type>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

      protected:

        unsigned is_contiguous_impl() const
        {
            return (size() == 0 || &operator[](shape_ - 1) == &operator[](size()-1))
                         ? contiguous_memory_flag
                         : 0;
        }

        void swap_impl(view_nd & rhs)
        {
            if(this != &rhs)
            {
                shape_.swap(rhs.shape_);
                strides_.swap(rhs.strides_);
                axistags_.swap(rhs.axistags_);
                std::swap(data_, rhs.data_);
                std::swap(flags_, rhs.flags_);
            }
        }

            // ensure that singleton axes have zero stride
        void zero_singleton_strides()
        {
            for (decltype(dimension()) k = 0; k < dimension(); ++k)
            {
                if (shape_[k] == 1)
                {
                    strides_[k] = 0;
                }
            }
        }

        template <class S>
        bool can_broadcast_rhs(S const & s) const
        {
            index_t diff = shape_.size() - s.size();
            if(diff < 0)
            {
                return false;
            }
            for(index_t k=s.size()-1; k>=0; --k)
            {
                if(s[k] != 1 && (index_t)s[k] != shape_[diff+k])
                {
                    return false;
                }
            }
            return true;
        }

        template <class E, class P=pointer,
                  class Q=std::remove_const_t<decltype(std::declval<E>().raw_data())>>
        auto
        create_view_impl(E const & e)
        -> std::enable_if_t<has_raw_data_api<E>::value && std::is_convertible<Q, P>::value>
        {
            shape_     = e.shape();
            strides_   = e.strides();
            axistags_  = axistags_type();
            data_      = const_cast<pointer>(e.raw_data() + e.raw_data_offset());
            flags_     = is_contiguous_impl() & ~owns_memory_flag;
        }

        template <class E, class P=pointer,
                  class Q=std::remove_const_t<decltype(std::declval<E>().raw_data())>>
        auto
        create_view_impl(E const & e)
        -> std::enable_if_t<has_raw_data_api<E>::value && !std::is_convertible<Q, P>::value>
        {
            vigra_fail("view_nd::operator=(): raw data pointers are incompatible.");
        }

        template <class E>
        auto
        create_view_impl(E const & e)
        -> std::enable_if_t<!has_raw_data_api<E>::value>
        {
            vigra_fail("view_nd::operator=(): cannot assign an expression to an empty view.");
        }

        template <class U>
        void assign_impl(U const & rhs)
        {
            detail::overlapping_memory_checker m(&(*this)(), &(*this)[shape()-1]+1);
            if(m(rhs))
            {
                // memory overlaps => we need a temporary
                semantic_base::assign(xt::xarray<value_type>(rhs));
            }
            else
            {
                semantic_base::assign(rhs);
            }
        }

        shape_type shape_;
        shape_type strides_;
        axistags_type axistags_;
        pointer data_;
        unsigned flags_;

      public:

            /** default constructor: create an invalid view,
             * i.e. has_data() returns false and size() is zero.
             */
        view_nd()
        : shape_()
        , strides_()
        , axistags_()
        , data_(0)
        , flags_(0)
        {}

        view_nd(view_nd const & other)
        : shape_(other.shape_)
        , strides_(other.strides_)
        , axistags_(other.axistags_)
        , data_(const_cast<pointer>(other.data_))
        , flags_(other.flags_ & ~owns_memory_flag)
        {}

        view_nd(view_nd && other)
        : view_nd(static_cast<view_nd const &>(other))
        {}

        template <index_t M>
        view_nd(view_nd<std::remove_const_t<T>, M> const & other)
        : shape_(other.shape())
        , strides_(other.strides())
        , axistags_(other.axistags())
        , data_(const_cast<pointer>(other.raw_data()))
        , flags_(other.flags() & ~owns_memory_flag)
        {
            static_assert(N == runtime_size || M == runtime_size || M == N,
                "view_nd<N>(view_nd<M>): dimension mismatch.");
        }

             /** construct from shape and pointer
             */
        view_nd(shape_type const & shape,
                const_pointer ptr,
                tags::memory_order order = c_order)
        : view_nd(shape, shape_to_strides(shape, order), ptr)
        {}

            /** construct from shape, axistags, and pointer
             */
        view_nd(shape_type const & shape,
                axistags_type   const & axistags,
                const_pointer ptr,
                tags::memory_order order = c_order)
        : view_nd(shape, shape_to_strides(shape, order), axistags, ptr)
        {}

            /** Construct from shape, strides (offset of a sample to the
                next, measured in units if `sizeof(T)`) for every dimension,
                and pointer.
             */
        view_nd(shape_type const & shape,
                shape_type const & strides,
                const_pointer ptr)
        : view_nd(shape, strides,
                  axistags_type(shape.size(), tags::axis_unknown), ptr)
        {}

            /** Construct from shape, strides (offset of a sample to the
                next, measured in units if `sizeof(T)`), axistags for every
                dimension, and pointer.
             */
        view_nd(shape_type const & shape,
                shape_type const & strides,
                axistags_type const & axistags,
                const_pointer ptr)
        : shape_(shape)
        , strides_(strides)
        , axistags_(axistags)
        , data_(const_cast<pointer>(ptr))
        , flags_(is_contiguous_impl())
        {
            XVIGRA_ASSERT_MSG(all_greater_equal(shape, 0),
                "view_nd(): invalid shape.");
            zero_singleton_strides();
        }

        template <class E>
        view_nd(const xt::xexpression<E>& ex)
        : view_nd()
        {
            create_view_impl(ex.derived_cast());
        }

        view_nd & operator=(view_nd const & rhs)
        {
            if(this != &rhs)
            {
                if(!has_data())
                {
                    shape_     = rhs.shape_;
                    strides_   = rhs.strides_;
                    axistags_  = rhs.axistags_;
                    data_      = rhs.data_;
                    flags_     = rhs.flags_ & ~owns_memory_flag;
                }
                else
                {
                    vigra_precondition(can_broadcast_rhs(rhs.shape()),
                        "view_nd::operator=(): cannot broadcast RHS shape to LHS.");
                    assign_impl(rhs);
                }
            }
            return *this;
        }

        view_nd & operator=(view_nd && rhs)
        {
            return (*this) = static_cast<view_nd const &>(rhs);
        }

        self_type& operator=(const_value_type & v)
        {
            vigra_precondition(has_data(),
                "vigra_nd::operator=(): cannot assign a value to an empty array.");
            if(is_contiguous())
            {
                std::fill(data_, data_+size(), v);
            }
            else
            {
                semantic_base::assign(xt::xscalar<const_value_type>(v));
            }
            return *this;
        }

        template <class E,
                  VIGRA_REQUIRE<(!std::is_convertible<std::decay_t<E>, value_type const &>::value)>>
        self_type& operator=(const xt::xexpression<E>& ex)
        {
            E const & e = ex.derived_cast();
            if(!has_data())
            {
                create_view_impl(e); // FIXME: should be simplified using `if constexpr`
            }
            else
            {
                vigra_precondition(can_broadcast_rhs(e.shape()),
                    "view_nd::operator=(): cannot broadcast RHS shape to LHS.");
                assign_impl(e);
            }
            return *this;
        }

#define XVIGRA_COMPUTED_ASSIGNMENT(OP, FCT)                                                  \
        self_type& operator OP(const_value_type & v)                                         \
        {                                                                                    \
            vigra_precondition(has_data(),                                                   \
                "vigra_nd::operator" #OP "(): cannot assign a value to an empty view.");     \
            return semantic_base::FCT(xt::xscalar<const_value_type>(v));                     \
        }                                                                                    \
                                                                                             \
        template <class E>                                                                   \
        self_type& operator OP(const xt::xexpression<E>& ex)                                 \
        {                                                                                    \
            E const & e = ex.derived_cast();                                                 \
            vigra_precondition(can_broadcast_rhs(e.shape()),                                 \
                "view_nd::operator" #OP "(): cannot broadcast RHS shape to LHS.");           \
            detail::overlapping_memory_checker m(&(*this)(), &(*this)[shape()-1]+1);         \
            if(m(e))                                                                         \
            {                                                                                \
                /* memory overlaps => we need a temporary */                                 \
                /* FIXME: use array_nd instread of xarray */                                 \
                return semantic_base::FCT(xt::xarray<value_type>(e));                        \
            }                                                                                \
            else                                                                             \
            {                                                                                \
                return semantic_base::FCT(e);                                                \
            }                                                                                \
        }

        XVIGRA_COMPUTED_ASSIGNMENT(+=, plus_assign)
        XVIGRA_COMPUTED_ASSIGNMENT(-=, minus_assign)
        XVIGRA_COMPUTED_ASSIGNMENT(*=, multiplies_assign)
        XVIGRA_COMPUTED_ASSIGNMENT(/=, divides_assign)
        XVIGRA_COMPUTED_ASSIGNMENT(%=, modulus_assign)

#undef XVIGRA_COMPUTED_ASSIGNMENT

            // needed for operator==
        template <class It>
        reference element(It first, It last)
        {
            XVIGRA_ASSERT_MSG(std::distance(first, last) == dimension(),
                "view_nd::element(): invalid index.");
            return *(data_ + std::inner_product(strides_.begin(), strides_.end(), first, 0l));
        }

            // needed for operator==
        template <class It>
        const_reference element(It first, It last) const
        {
            XVIGRA_ASSERT_MSG(std::distance(first, last) == dimension(),
                "view_nd::element(): invalid index.");
            return *(data_ + std::inner_product(strides_.begin(), strides_.end(), first, 0l));
        }

            // needed for operator==
        xt::layout_type layout() const
        {
            return xt::layout_type::dynamic; // FIXME
        }

            // needed for semantic_base::assign(expr)
        static constexpr bool contiguous_layout = false; // FIXME
        static constexpr xt::layout_type static_layout = xt::layout_type::dynamic; // FIXME

            // needed for semantic_base::assign(expr)
        template <class S>
        bool broadcast_shape(S& s, bool=false) const
        {
            // FIXME: S here is svector
            return xt::broadcast_shape(shape_, s);
        }

            // needed for semantic_base::assign(expr)
        template <class S>
        constexpr bool is_trivial_broadcast(const S& str) const noexcept
        {
            return true; // FIXME: check
        }

            // reshape the view in-place
        template <class SHAPE>
        void
        reshape(SHAPE const & new_shape, tags::memory_order order = c_order)
        {
            vigra_precondition(is_contiguous(),
                "view_nd::reshape(): only contiguous arrays can be reshaped.");
            vigra_precondition(N == runtime_size || N == (index_t)new_shape.size(),
                "view_nd::reshape(): dimension mismatch between old and new shape.");
            vigra_precondition((index_t)xt::compute_size(new_shape) == size(),
                "view_nd::reshape(): size mismatch between old and new shape.");
            unsigned owner = flags_ & owns_memory_flag;
            view_nd(new_shape, data_, order).swap_impl(*this);
            flags_ |= owner;
        }

            // return a reshaped view
        template <index_t M>
        view_nd<T, M>
        reshaped(shape_t<M> const & new_shape,
                 axis_tags<M> new_axistags = axis_tags<M>{},
                 tags::memory_order order = c_order) const
        {
            vigra_precondition(is_contiguous(),
                "view_nd::reshaped(): only contiguous arrays can be reshaped.");
            vigra_precondition(prod(new_shape) == size(),
                "view_nd::reshaped(): size mismatch between old and new shape.");
            if(new_axistags.size() != new_shape.size())
            {
                new_axistags = axis_tags<M>(new_shape.size(), tags::axis_unknown);
            }
            return view_nd<T, M>(new_shape, new_axistags, data_, order);
        }

        decltype(auto)
        flattened() const
        {
            return reshaped(shape_t<1>{size()});
        }

            // needed for semantic_base::assign(expr)
        template <class ST>
        stepper stepper_begin(const ST& s)
        {
            size_type offset = s.size() - dimension();
            return stepper(this, offset);
        }

            // needed for semantic_base::assign(expr)
        template <class ST>
        stepper stepper_end(const ST& s, xt::layout_type = xt::layout_type::row_major)
        {
            size_type offset = s.size() - dimension();
            return stepper(this, offset, true);
        }

            // needed for operator==
        template <class ST>
        const_stepper stepper_begin(const ST& s) const
        {
            size_type offset = s.size() - dimension();
            return const_stepper(this, offset);
        }

            // needed for operator==
        template <class ST>
        const_stepper stepper_end(const ST& s, xt::layout_type = xt::layout_type::row_major) const
        {
            size_type offset = s.size() - dimension();
            return const_stepper(this, offset, true);
        }

            /** Access element.
             */
        reference operator[](shape_type const & d)
        {
            XVIGRA_ASSERT_INSIDE(d);
            return *(data_ + dot(d, strides_));
        }

            /** Access element via scalar index w.r.t. raw memory.
             */
        reference operator[](size_type i)
        {
            return *(data_ + i);
        }

            /** Get element.
             */
        const_reference operator[](shape_type const & d) const
        {
            XVIGRA_ASSERT_INSIDE(d);
            return *(data_ + dot(d, strides_));
        }

            /** Get element via scalar index w.r.t. raw memory.
             */
        const_reference operator[](size_type i) const
        {
            return *(data_ + i);
        }

            /** Access the array's first element.
             */
        reference operator()()
        {
            return *data_;
        }

            /** 1D array access. Use only if <tt>dimension() <= 1</tt>.
             */
        reference operator()(index_t i)
        {
            XVIGRA_ASSERT_MSG(dimension() <= 1,
                          "view_nd::operator()(int): only allowed if dimension() <= 1");
            return *(data_ + i*strides_[dimension()-1]);
        }

            /** N-D array access. Number of indices must match <tt>dimension()</tt>.
             */
        template <class ... INDICES>
        reference operator()(index_t i0, index_t i1,
                             INDICES ... i)
        {
            static const index_t M = 2 + sizeof...(INDICES);
            XVIGRA_ASSERT_MSG(dimension() == M,
                "view_nd::operator()(INDICES): number of indices must match dimension().");
            return *(data_ + dot(shape_t<M>{i0, i1, i...}, strides_));
        }

            /** Access the array's first element.
             */
        const_reference operator()() const
        {
            return *data_;
        }

            /** 1D array access. Use only if <tt>dimension() <= 1</tt>.
             */
        const_reference operator()(index_t i) const
        {
            XVIGRA_ASSERT_MSG(dimension() <= 1,
                          "view_nd::operator()(int): only allowed if dimension() <= 1");
            return *(data_ + i*strides_[dimension()-1]);
        }

            /** N-D array access. Number of indices must match <tt>dimension()</tt>.
             */
        template <class ... INDICES>
        const_reference operator()(index_t i0, index_t i1,
                                   INDICES ... i) const
        {
            static const index_t M = 2 + sizeof...(INDICES);
            XVIGRA_ASSERT_MSG(dimension() == M,
                "view_nd::operator()(INDICES): number of indices must match dimension().");
            return *(data_ + dot(shape_t<M>{i0, i1, i...}, strides_));
        }

            /** Bind 'axis' to 'index'.

                This reduces the dimensionality of the array by one.

                <b>Usage:</b>
                \code
                // create a 3D array of size 40x30x20
                array_nd<3, double> array3({40, 30, 20});

                // get a 2D array by fixing index 2 to 15
                view_nd<2, double> array2 = array3.bind(2, 15);
                \endcode
             */
        view_nd<T, ((N < 0) ? runtime_size : N-1)>
        bind(int axis, index_t index=0) const
        {
            using view_t = view_nd<T, ((N < 0) ? runtime_size : N-1)>;

            XVIGRA_ASSERT_MSG(0 <= axis && axis < dimension() && 0 <= index && index < shape_[axis],
                "view_nd::bind(): index out of range.");

            auto point = unit_vector(shape(), axis, index);
            if (dimension() == 1)
            {
                shape_t<view_t::internal_dimension> shape{ 1 }, strides{ 1 };
                tiny_vector<tags::axis_tag, view_t::internal_dimension> axistags{ tags::axis_unknown };
                return view_t(shape, strides, axistags, &operator[](point));
            }
            else
            {
                return view_t(shape_.erase(axis),
                              strides_.erase(axis),
                              axistags_.erase(axis),
                              &operator[](point));
            }
        }

            /** Bind the dimensions 'axes' to 'indices.

                Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
                The elements of 'axes' must be unique, contained in the interval
                <tt>0 <= element < dimension()<//t> and be sorted in ascending order.
                The elements of 'indices' must be in the valid range of the
                corresponding axes.
             */
        template <index_t M>
        decltype(auto)
        bind(shape_t<M> const & axes, shape_t<M> const & indices) const
        {
            static_assert(N == runtime_size || M <= N,
                "view_nd::bind(shape_t<M>): M <= N required.");
            return bind(axes.back(), indices.back())
                      .bind(axes.pop_back(), indices.pop_back());
        }

        template <index_t M>
        decltype(auto)
        bind(shape_t<M> const & axes) const
        {
            return bind(axes, shape_t<M>(axes.size(), 0));
        }

        decltype(auto)
        bind(shape_t<1> const & a, shape_t<1> const & i) const
        {
            return bind(a[0], i[0]);
        }

        view_nd const &
        bind(shape_t<0> const &, shape_t<0> const &) const
        {
            return *this;
        }

        view_nd<T, runtime_size>
        bind(shape_t<runtime_size> const & axes, shape_t<runtime_size> const & indices) const
        {
            vigra_precondition(axes.size() == indices.size(),
                "view_nd::bind(): size mismatch between 'axes' and 'indices'.");
            vigra_precondition(axes.size() <= dimension(),
                "view_nd::bind(): axes.size() <= dimension() required.");

            view_nd<T, runtime_size> a(*this);
            if(axes.size() == 0)
                return a;
            else
                return a.bind(axes.back(), indices.back())
                           .bind(axes.pop_back(), indices.pop_back());
        }

            /** Bind the first 'indices.size()' dimensions to 'indices'.

                Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
             */
        template <index_t M>
        decltype(auto)
        bind_left(shape_t<M> const & indices) const -> decltype(this->bind(indices, indices))
        {
            return bind(shape_t<M>::range((index_t)indices.size()), indices);
        }

            /** Bind the last 'indices.size()' dimensions to 'indices'.

                Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
             */
        template <index_t M>
        decltype(auto)
        bind_right(shape_t<M> const & indices) const -> decltype(this->bind(indices, indices))
        {
            return bind(shape_t<M>::range((index_t)indices.size()) + (index_t)(dimension() - indices.size()),
                        indices);
        }

            /** Bind the channel axis to index d.
                This calls <tt>array.bind(array.channel_axis(), d)</tt>
                when a channel axis is defined and throws an error otherwise.
                \endcode
             */
        template <class U=T,
                  VIGRA_REQUIRE<std::is_arithmetic<U>::value> >
        view_nd<T, ((N < 0) ? runtime_size : N-1)>
        bind_channel(index_t d) const
        {
            int m = channel_axis();

            XVIGRA_ASSERT_MSG(m != tags::axis_missing,
                "view_nd::bind_channel(): array has no channel axis.");

            return bind(m, d);
        }

            /** Create a view to channel 'i' of a vector-like value type. Possible value types
                (of the original array) are: \ref TinyVector, \ref RGBValue, \ref FFTWComplex,
                and <tt>std::complex</tt>. The function can be applied whenever the array's
                element type <tt>T</tt> defines an embedded type <tt>T::value_type</tt> which
                becomes the return type of <tt>bind_channel()</tt>.


                <b>Usage:</b>
                \code
                    array_nd<2, RGBValue<float> > rgb_image({h,w});

                    view_nd<2, float> red   = rgb_image.bind_channel(0);
                    view_nd<2, float> green = rgb_image.bind_channel(1);
                    view_nd<2, float> blue  = rgb_image.bind_channel(2);
                \endcode
            */
        template <class U=T,
                  VIGRA_REQUIRE<(U::static_size > 1)> >
        view_nd<typename U::value_type, N>
        bind_channel(index_t i) const
        {
            vigra_precondition(0 <= i && i < U::static_size,
                "view_nd::bind_channel(i): 'i' out of range.");
            return expand_elements(0).bind(0, i);
        }

            /** Create a view where a vector-like element type is expanded into a new
                array dimension. The new dimension is inserted at index position 'd',
                which must be between 0 and N inclusive.

                Possible value types of the original array are: \ref tiny_vector, \ref RGBValue,
                \ref FFTWComplex, <tt>std::complex</tt>, and the built-in number types (in this
                case, <tt>expand_elements</tt> is equivalent to <tt>newaxis</tt>).

                <b>Usage:</b>
                \code
                    array_nd<2, tiny_vector<float, 3> > rgb_image({h, w});

                    view_nd<3, float> multiband_image = rgb_image.expand_elements(2);
                \endcode
            */
        template <class U=T,
                  VIGRA_REQUIRE<(U::static_size > 0)> >
        view_nd<typename U::value_type, (N == runtime_size ? runtime_size : N+1)>
        expand_elements(index_t d) const
        {
            using value_t = typename T::value_type;
            using view_t  = view_nd<value_t, (N == runtime_size ? runtime_size : N + 1)>;

            vigra_precondition(0 <= d && d <= (index_t)dimension(),
                "view_nd::expand_elements(d): 0 <= 'd' <= dimension() required.");

            constexpr index_t s = T::static_size;
            return view_t(shape_.insert(d, s),
                          (strides_*U::static_size).insert(d, 1),
                          axistags_.insert(d, tags::axis_c),
                          reinterpret_cast<value_t*>(data_));
        }

            /** Create a view with an explicit channel axis at index \a d.

                There are three cases:
                <ul>
                <li> If the array's <tt>value_type</tt> is scalar, and the array already
                     has an axis marked as channel axis, the array is transposed such
                     that the channel axis is at index \a d.

                <li> If the array's <tt>value_type</tt> is scalar, and the array does
                     not have a channel axis, the function
                     <tt>newaxis(d, tags::axis_c)</tt> is called.

                <li> If the array's <tt>value_type</tt> is  vectorial, the function
                     <tt>expand_elements(d)</tt> is called.
                </ul>
                Thus, the function can be called repeatedly without error.

                <b>Usage:</b>
                \code
                    array_nd<2, tiny_vector<float, 3> > rgb_image({h, w});

                    view_nd<3, float> multiband_image = rgb_image.ensure_channel_axis(2);
                    assert(multiband_image.channel_axis() == 3);
                \endcode
            */
        template <class U=T,
                  VIGRA_REQUIRE<(U::static_size > 1)> >
        view_nd<typename U::value_type, runtime_size>
        ensure_channel_axis(index_t d) const
        {
            return expand_elements(d);
        }

        template <class U=T,
                  VIGRA_REQUIRE<std::is_arithmetic<U>::value>>
        view_nd<T, runtime_size>
        ensure_channel_axis(index_t d) const
        {
            vigra_precondition(d >= 0,
                "view_nd::ensure_channel_axis(d): d >= 0 required.");
            int c = channel_axis();
            if(c == d)
                return *this;
            if(c < 0)
                return newaxis(d, tags::axis_c);
            vigra_precondition(d < (index_t)dimension(),
                "view_nd::ensure_channel_axis(d): d < dimension() required.");
            auto permutation = shape_t<>::range(dimension()).erase(c).insert(d, c);
            return transpose(permutation);
        }

            /** Add a singleton dimension (dimension of length 1).

                Singleton dimensions don't change the size of the data, but introduce
                a new index that can only take the value 0. This is mainly useful for
                the 'reduce mode' of transformMultiArray() and combineTwoMultiArrays(),
                because these functions require the source and destination arrays to
                have the same number of dimensions.

                The range of \a i must be <tt>0 <= i <= N</tt>. The new dimension will become
                the i'th index, and the old indices from i upwards will shift one
                place to the right.

                <b>Usage:</b>

                Suppose we want have a 2D array and want to create a 1D array that contains
                the row average of the first array.
                \code
                typedef MultiArrayshape_t<2>::type Shape2;
                array_nd<2, double> original(Shape2(40, 30));

                typedef MultiArrayshape_t<1>::type Shape1;
                array_nd<1, double> rowAverages(Shape1(30));

                // temporarily add a singleton dimension to the destination array
                transformMultiArray(srcMultiArrayRange(original),
                                    destMultiArrayRange(rowAverages.newaxis(0)),
                                    FindAverage<double>());
                \endcode
             */
        view_nd <T, (N < 0) ? runtime_size : N+1>
        newaxis(index_t i,
                tags::axis_tag tag = tags::axis_unknown) const
        {
            using view_t = view_nd<T, (N < 0) ? runtime_size : N+1>;
            return view_t(shape_.insert(i, 1), strides_.insert(i, 0),
                          axistags_.insert(i, tag), raw_data());
        }

            /** Create a view to the diagonal elements of the array.

                This produces a 1D array view whose size equals the size
                of the shortest dimension of the original array.

                <b>Usage:</b>
                \code
                // create a 3D array of size 40x30x20
                array_nd<3, double> array3(shape_t<3>(40, 30, 20));

                // get a view to the diagonal elements
                view_nd<1, double> diagonal = array3.diagonal();
                assert(diagonal.shape(0) == 20);
                \endcode
            */
        view_nd<T, 1>
        diagonal() const
        {
            return view_nd<T, 1>(shape_t<1>{min(shape_)},
                                 shape_t<1>{sum(strides_)},
                                 raw_data());
        }

            /** create a rectangular subarray that spans between the
                points p and q, where p is in the subarray, q not.
                If an element of p or q is negative, it is subtracted
                from the correspongng shape.

                <b>Usage:</b>
                \code
                // create a 3D array of size 40x30x20
                array_nd<3, double> array3(shape_t<3>(40, 30, 20));

                // get a subarray set is smaller by one element at all sides
                view_nd<3, double> subarray  = array3.subarray(shape_t<3>(1,1,1), shape_t<3>(39, 29, 19));

                // specifying the end point with a vector of '-1' is equivalent
                view_nd<3, double> subarray2 = array3.subarray(shape_t<3>(1,1,1), shape_t<3>(-1, -1, -1));
                \endcode
            */
        view_nd
        subarray(shape_type p, shape_type q) const
        {
            vigra_precondition(p.size() == dimension() && q.size() == dimension(),
                "view_nd::subarray(): size mismatch.");
            for(decltype(dimension()) k=0; k<dimension(); ++k)
            {
                if(p[k] < 0)
                    p[k] += shape_[k];
                if(q[k] < 0)
                    q[k] += shape_[k];
            }
            vigra_precondition(is_inside(p) && all_less_equal(p, q) && all_less_equal(q, shape_),
                "view_nd::subarray(): invalid subarray limits.");
            const index_t offset = dot(strides_, p);
            return view_nd(q - p, strides_, axistags_, data_ + offset);
        }

            /** Transpose an array. If N==2, this implements the usual matrix transposition.
                For N > 2, it reverses the order of the indices.

                <b>Usage:</b><br>
                \code
                typedef array_nd<2, double>::shape_type Shape;
                array_nd<2, double> array(10, 20);

                view_nd<2, double> transposed = array.transpose();

                for(int i=0; i<array.shape(0), ++i)
                    for(int j=0; j<array.shape(1); ++j)
                        assert(array(i, j) == transposed(j, i));
                \endcode
            */
        view_nd<T, N>
        transpose() const
        {
            return view_nd<T, N>(reversed(shape_),
                                 reversed(strides_),
                                 reversed(axistags_),
                                 raw_data());
        }

            /** Permute the dimensions of the array.
                The function exchanges the order of the array's axes without copying the data.
                Argument\a permutation specifies the desired order such that
                <tt>permutation[k] = j</tt> means that axis <tt>j</tt> in the original array
                becomes axis <tt>k</tt> in the transposed array.

                <b>Usage:</b><br>
                \code
                typedef array_nd<2, double>::shape_type Shape;
                array_nd<2, double> array(10, 20);

                view_nd<2, double, StridedArrayTag> transposed = array.transpose(Shape(1,0));

                for(int i=0; i<array.shape(0), ++i)
                    for(int j=0; j<array.shape(1); ++j)
                        assert(array(i, j) == transposed(j, i));
                \endcode
            */
        template <index_t M>
        view_nd
        transpose(shape_t<M> const & permutation) const
        {
            static_assert(M == internal_dimension || M == runtime_size || N == runtime_size,
                "view_nd::transpose(): permutation.size() doesn't match dimension().");
            vigra_precondition(permutation.size() == dimension(),
                "view_nd::transpose(): permutation.size() doesn't match dimension().");
            shape_type p(permutation);
            view_nd res(transposed(shape_, p),
                        transposed(strides_, p),
                        transposed(axistags_, p),
                        raw_data());
            return res;
        }

        view_nd
        transpose(tags::memory_order order) const
        {
            return transpose(detail::permutation_to_order(strides_, order));
        }

            // cast the type of an array to its base view
        view_nd & view()
        {
            return *this;
        }

            // cast the type of a view to dimension M
            // (useful to create static views from matching runtime_size arrays for better performance)
        template <index_t M>
        view_nd<T, M>
        view()
        {
            static_assert(M == runtime_size || N == runtime_size || M == N,
                "view_nd::view(): desired dimension is incompatible with dimension().");
            vigra_precondition(M == runtime_size || M == (index_t)dimension(),
                "view_nd::view(): desired dimension is incompatible with dimension().");
            return view_nd<T, M>(shape_t<M>(shape_.begin(), shape_.begin()+dimension()),
                                 shape_t<M>(strides_.begin(), strides_.begin()+dimension()),
                                 axis_tags<M>(axistags_.begin(), axistags_.begin()+dimension()),
                                 raw_data());
        }

            // get a view with static slicing
        template <class S, class ... A,
                  VIGRA_REQUIRE<!std::is_base_of<slice_vector, S>::value>>
        auto
        view(S s, A ... a)
        {
            using traits = detail::slice_dimension_traits<N, S, A...>;
            constexpr index_t M = traits::target_dimension;
            shape_type point(dimension(), 0);
            shape_t<> new_shape, new_strides;
            detail::parse_slices(point, new_shape, new_strides, shape(), strides(), s, a...);
            const index_t offset = dot(strides(), point);
            using new_axistags_type = typename view_nd<T, M>::axistags_type;
            return view_nd<T, M>(new_shape, new_strides,
                                 new_axistags_type(), data_ + offset);
        }

            // get a view with dynamic slicing
        auto
        view(slice_vector const & s)
        {
            shape_type point(dimension(), 0);
            shape_t<> new_shape, new_strides;
            detail::parse_slices(point, new_shape, new_strides, shape(), strides(), s);
            const index_t offset = dot(strides(), point);
            using new_axistags_type = typename view_nd<T>::axistags_type;
            return view_nd<T>(new_shape, new_strides,
                              new_axistags_type(), data_ + offset);
        }

            // cast the type of a const array to its base view
        view_nd const &
        view() const
        {
            return *this;
        }

            // cast the type of a const view to dimension M
            // (useful to create static views from matching runtime_size arrays for better performance)
        template <index_t M>
        view_nd<const_value_type, M>
        view() const
        {
            return const_cast<self_type *>(this)->template view<M>();
        }

            // get a const view with static slicing
        template <class S, class ... A,
                  VIGRA_REQUIRE<!std::is_base_of<slice_vector, S>::value>>
        view_nd<const_value_type, detail::slice_dimension_traits<N, S, A...>::target_dimension>
        view(S s, A ... a) const
        {
            return const_cast<self_type *>(this)->view(s, a...);
        }

            // get a const view with dynamic slicing
        auto
        view(slice_vector const & s) const -> view_nd<const_value_type, runtime_size>
        {
            return const_cast<self_type *>(this)->view(s);
        }

            // cast the type of an array to its const base view
        view_nd const &
        cview() const
        {
            return *this;
        }

            // cast the type of a view or const view to dimension M
            // (useful to create static views from matching runtime_size arrays for better performance)
        template <index_t M>
        view_nd<const_value_type, M> cview() const
        {
            return const_cast<self_type *>(this)->template view<M>();
        }

            // get a const view with static slicing
        template <class S, class ... A,
                  VIGRA_REQUIRE<!std::is_base_of<slice_vector, S>::value>>
        view_nd<const_value_type, detail::slice_dimension_traits<N, S, A...>::value>
        cview(S s, A ... a) const
        {
            return const_cast<self_type *>(this)->view(s, a...);
        }

            // get a const view with dynamic slicing
        auto
        cview(slice_vector const & s) const -> view_nd<const_value_type, runtime_size>
        {
            return const_cast<self_type *>(this)->view(s);
        }

        // pointer data()
        // {
        //     return (pointer)data_;
        // }

        // const_pointer data() const
        // {
        //     return (pointer)data_;
        // }

        pointer raw_data() noexcept
        {
            return (pointer)data_;
        }

        const_pointer raw_data() const noexcept
        {
            return (pointer)data_;
        }

        size_type raw_data_offset() const noexcept
        {
            return 0;
        }

        self_type & data()
        {
            return *this;
        }

        self_type const & data() const
        {
            return *this;
        }

        const shape_type & shape() const
        {
            return shape_;
        }

            /** return the array's shape at a certain dimension.
             */
        index_t shape(index_t n) const
        {
            return shape_[n];
        }

        shape_type const & strides() const
        {
            return strides_;
        }

            /** return the array's stride at a certain dimension.
             */
        index_t strides(index_t n) const
        {
            return strides_[n];
        }

            /** number of the elements in the array.
             */
        index_t size() const
        {
            return max(0, prod(shape_));
        }

        template <index_t M = N>
        std::size_t dimension(std::enable_if_t<M == runtime_size, bool> = true) const
        {
            return shape_.size();
        }

        template <index_t M = N>
        constexpr std::size_t dimension(std::enable_if_t<(M > runtime_size), bool> = true) const
        {
            return N;
        }

            /** check whether the given point is in the array range.
             */
        bool is_inside(shape_type const & p) const
        {
            return all_greater_equal(p, 0) && all_less(p, shape());
        }

            /** check whether the given point is not in the array range.
             */
        bool is_outside(shape_type const & p) const
        {
            return !is_inside(p);
        }
            /**
             * Returns true iff this view refers to valid data,
             * i.e. data() is not a NULL pointer. In particular, the function
             * returns `false` when the array was created with the default
             * constructor.
             */
        bool has_data() const
        {
            return data_ != 0;
        }

            /**
            * Returns true iff this view refers to contiguous memory.
            */
        bool is_contiguous() const
        {
            return (flags_ & contiguous_memory_flag) != 0;
        }

            /**
            * Returns true iff this view owns its memory.
            */
        bool owns_memory() const
        {
            return (flags_ & owns_memory_flag) != 0;
        }

            /**
            * Returns the addresses of the first array element and one byte beyond the
            * last array element.
            */
        tiny_vector<char *, 2> memory_range() const
        {
            return tiny_vector<char *, 2>{ (char*)data_, (char*)(1 + &(*this)[shape() - 1]) };
        }

        unsigned flags() const
        {
            return flags_;
        }

        axistags_type const & axistags() const
        {
            return axistags_;
        }

        view_nd & set_axistags(axistags_type const & t)
        {
            XVIGRA_ASSERT_MSG(t.size() == dimension(),
                "view_nd::set_axistags(): size mismatch.");
            axistags_ = t;
            return *this;
        }

        view_nd & set_channel_axis(int c)
        {
            XVIGRA_ASSERT_MSG(0 <= c && c < dimension(),
                "view_nd::set_channel_axis(): index out of range.");
            axistags_[c] = tags::axis_c;
            return *this;
        }

        index_t channel_axis() const
        {
            for(decltype(dimension()) k=0; k<dimension(); ++k)
                if(axistags_[k] == tags::axis_c)
                    return k;
            return tags::axis_missing;
        }

        index_t axis_index(tags::axis_tag tag) const
        {
            for(decltype(dimension()) k=0; k<dimension(); ++k)
                if(axistags_[k] == tag)
                    return k;
            return tags::axis_missing;
        }

        bool has_axis(tags::axis_tag tag) const
        {
            return axis_index(tag) != tags::axis_missing;
        }

        bool has_channel_axis() const
        {
            return channel_axis() != tags::axis_missing;
        }

        void swap(view_nd & rhs)
        {
            vigra_precondition(!owns_memory() && !rhs.owns_memory(),
                "view_nd::swap(): only allowed when views don't own their memory.");
            swap_impl(rhs);
        }

            /** Swap the data between two view_nd objects.

                The shapes of the two array must match. Both array views
                still point to the same memory as before, just the contents
                are exchanged.
            */
        template <class U, index_t M>
        void
        swap_data(view_nd<U, M> rhs)
        {
            static_assert(M == N || M == runtime_size || N == runtime_size,
                "view_nd::swap_data(): incompatible dimensions.");
            vigra_precondition(shape() == rhs.shape(),
                "view_nd::swap_data(): shape mismatch.");
            using std::swap;
            auto i = this->begin(),
                 e = this->end();
            auto k = rhs.begin();
            for(; i != e; ++i, ++k)
            {
                swap(*i, *k);
            }
        }
    };

    template <class T, index_t N>
    inline decltype(auto)
    transpose(view_nd<T, N> const & array)
    {
        return array.transpose();
    }

    template <class T, index_t N>
    inline decltype(auto)
    transpose(view_nd<T, N> & array)
    {
        return array.transpose();
    }

    template <class T, index_t N, class A>
    inline decltype(auto)
    transpose(array_nd<T, N, A> const & array)
    {
        return array.transpose();
    }

    template <class T, index_t N, class A>
    inline decltype(auto)
    transpose(array_nd<T, N, A> & array)
    {
        return array.transpose();
    }

    template <class T, index_t N>
    inline void
    swap(view_nd<T,N> & v1, view_nd<T,N> & v2)
    {
        v1.swap(v2);
    }

    template <class E,
              VIGRA_REQUIRE<view_nd_concept<E>::value>>
    inline decltype(auto)
    make_view(E && e)
    {
        return std::forward<E>(e);
    }

    template <class EC, xt::layout_type L, class SC, class Tag>
    inline auto
    make_view(xt::xarray_container<EC, L, SC, Tag> const & e)
    {
        using T = typename xt::xarray_container<EC, L, SC, Tag>::value_type;
        return view_nd<T>(e);
    }

    template <class EC, std::size_t N, xt::layout_type L , class Tag>
    inline auto
    make_view(xt::xtensor_container<EC, N, L, Tag> const & e)
    {
        using T = typename xt::xtensor_container<EC, N, L, Tag>::value_type;
        return view_nd<T, (index_t)N>(e);
    }

    template <class E,
              VIGRA_REQUIRE<has_raw_data_api<E>::value>>
    inline decltype(auto)
    eval_expr(E && e)
    {
        return std::forward<E>(e);
    }

    template <class E,
              VIGRA_REQUIRE<!has_raw_data_api<E>::value>>
    inline auto
    eval_expr(E && e)
    {
        return xt::eval(std::forward<E>(e));
    }

    /************/
    /* array_nd */
    /************/

    template <class T, index_t N, class ALLOC>
    class array_nd
    : public view_nd<T, N>
    {
      public:
        using view_type = view_nd<T, N>;
        using view_type::internal_dimension;
        using value_type = typename view_type::value_type;
        using pointer = typename view_type::pointer;
        using const_pointer = typename view_type::const_pointer;
        using reference = typename view_type::reference;
        using const_reference = typename view_type::const_reference;
        using size_type = typename view_type::size_type;
        using difference_type = typename view_type::difference_type;
        using shape_type = typename view_type::shape_type;
        using axistags_type = typename view_type::axistags_type;
        using iterator = typename view_type::iterator;
        using const_iterator = typename view_type::const_iterator;

        using raw_value_type = std::decay_t<value_type>;
        using allocator_type = ALLOC;
        using buffer_type = std::vector<raw_value_type, allocator_type>;

        using self_type = array_nd<T, N, ALLOC>;
        using semantic_base = typename view_type::semantic_base;

        using inner_shape_type = shape_type;
        using inner_strides_type = inner_shape_type;

      private:

        buffer_type allocated_data_;

      public:
            /** default constructor
             */
        array_nd()
        {}

            /** construct with given allocator
             */
        explicit
        array_nd(allocator_type const & alloc)
        : view_type()
        , allocated_data_(alloc)
        {}


            /** construct with given shape
             */
        explicit
        array_nd(shape_type const & shape,
                 tags::memory_order order = c_order,
                 allocator_type const & alloc = allocator_type())
        : array_nd(shape, value_type(), order, alloc)
        {}

            /** construct with given shape and axistags
             */
        array_nd(shape_type const & shape,
                 axistags_type const & axistags,
                 tags::memory_order order = c_order,
                 allocator_type const & alloc = allocator_type())
        : array_nd(shape, axistags, value_type(), order, alloc)
        {}

            /** construct from shape with an initial value
             */
        array_nd(shape_type const & shape,
                 const_reference init,
                 tags::memory_order order = c_order,
                 allocator_type const & alloc = allocator_type())
        : array_nd(shape, axistags_type(shape.size(), tags::axis_unknown), init, order, alloc)
        {}

            /** construct from shape and axistags with an initial value
             */
        array_nd(shape_type const & shape,
                 axistags_type const & axistags,
                 const_reference init,
                 tags::memory_order order = c_order,
                 allocator_type const & alloc = allocator_type())
        : view_type(shape, axistags, 0, order)
        , allocated_data_(this->size(), init, alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }
           /** copy constructor
             */
        array_nd(array_nd const & rhs)
        : view_type(rhs)
        , allocated_data_(rhs.allocated_data_)
        {
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

            /** move constructor
             */
        array_nd(array_nd && rhs)
        : view_type()
        , allocated_data_()
        {
            operator=(std::forward<array_nd>(rhs));
        }

            /** construct by copying from a view_nd
             */
        template <index_t M, class U>
        array_nd(view_nd<U, M> const & rhs,
                 tags::memory_order order = c_order,
                 allocator_type const & alloc = allocator_type())
        : view_type(rhs.shape(), rhs.axistags(), 0, order)
        , allocated_data_(alloc)
        {
            allocated_data_.reserve(this->size());

            if(order == f_order)
            {
                auto end = rhs.template cend<f_order>();
                for(auto k = rhs.template cbegin<f_order>(); k != end; ++k)
                {
                    allocated_data_.emplace_back(conditional_cast<std::is_arithmetic<T>::value, T>(*k));
                }
            }
            else
            {
                auto end = rhs.cend();
                for(auto k = rhs.cbegin(); k != end; ++k)
                {
                    allocated_data_.emplace_back(conditional_cast<std::is_arithmetic<T>::value, T>(*k));
                }
            }
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

            /** Constructor from an xtensor data structure or array expression.
            */
        template <class E,
                  VIGRA_REQUIRE<is_xexpression<E>::value>>
        array_nd(E && e)
        : view_type(e.shape(), 0)
        , allocated_data_(this->size())
        {
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
            semantic_base::assign(std::forward<E>(e));
        }

            /** construct from shape and iterator pair
             */
        template <class ITER,
                  VIGRA_REQUIRE<forward_iterator_concept<ITER>::value>>
        array_nd(shape_type const & shape,
                 ITER b, ITER e,
                 tags::memory_order order = c_order,
                 allocator_type const & alloc = allocator_type())
        : array_nd(shape, axistags_type(shape.size(), tags::axis_unknown), b, e, order, alloc)
        {}

            /** construct from shape, axistags and iterator pair
             */
        template <class ITER,
                  VIGRA_REQUIRE<forward_iterator_concept<ITER>::value>>
        array_nd(shape_type const & shape,
                 axistags_type const & axistags,
                 ITER b, ITER e,
                 tags::memory_order order = c_order,
                 allocator_type const & alloc = allocator_type())
        : view_type(shape, axistags, 0, order)
        , allocated_data_(b, e, alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            vigra_precondition(this->size() == (index_t)allocated_data_.size(),
                "array_nd(): iterator range length contradics shape.");
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

        array_nd(xt::nested_initializer_list_t<T, 1> t)
        : view_type(xt::shape<shape_t<1>>(t), 0, c_order)
        , allocated_data_(this->size())
        {
            static_assert(N == runtime_size || N == 1,
                "array_nd(): initializer_list has incompatible dimension.");
            xt::nested_copy(allocated_data_.begin(), t);
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

        array_nd(xt::nested_initializer_list_t<T, 2> t)
        : view_type(xt::shape<shape_t<2>>(t), 0, c_order)
        , allocated_data_(this->size())
        {
            static_assert(N == runtime_size || N == 2,
                "array_nd(): initializer_list has incompatible dimension.");
            xt::nested_copy(allocated_data_.begin(), t);
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

        array_nd(xt::nested_initializer_list_t<T, 3> t)
        : view_type(xt::shape<shape_t<3>>(t), 0, c_order)
        , allocated_data_(this->size())
        {
            static_assert(N == runtime_size || N == 3,
                "array_nd(): initializer_list has incompatible dimension.");
            xt::nested_copy(allocated_data_.begin(), t);
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

        array_nd(xt::nested_initializer_list_t<T, 4> t)
        : view_type(xt::shape<shape_t<4>>(t), 0, c_order)
        , allocated_data_(this->size())
        {
            static_assert(N == runtime_size || N == 4,
                "array_nd(): initializer_list has incompatible dimension.");
            xt::nested_copy(allocated_data_.begin(), t);
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

        array_nd(xt::nested_initializer_list_t<T, 5> t)
        : view_type(xt::shape<shape_t<5>>(t), 0, c_order)
        , allocated_data_(this->size())
        {
            static_assert(N == runtime_size || N == 5,
                "array_nd(): initializer_list has incompatible dimension.");
            xt::nested_copy(allocated_data_.begin(), t);
            this->data_  = &allocated_data_[0];
            this->flags_ |= this->contiguous_memory_flag | this->owns_memory_flag;
        }

            /** Copy assignment.<br>
                If the size of \a rhs is the same as the left-hand side arrays's
                old size, only the data are copied. Otherwise, new storage is
                allocated, which invalidates all dependent objects (array views, iterators)
                of the lhs array.
             */
        self_type & operator=(array_nd const & rhs)
        {
            if (this != &rhs)
            {
                operator=(rhs.view());
            }
            return *this;
        }

            /** Move assignment.<br>
                If the size of \a rhs is the same as the left-hand side arrays's
                old size, only the data are copied. Otherwise, the storage of the
                rhs is moved to the lhs, which invalidates all dependent objects
                (array views, iterators) of the lhs array.
             */
        self_type & operator=(array_nd && rhs)
        {
            if (this != &rhs)
            {
                if(this->can_broadcast_rhs(rhs.shape()))
                {
                    this->assign_impl(rhs);
                }
                else
                {
                    this->shape_ = rhs.shape_;
                    rhs.shape_ = shape_type{};
                    this->strides_ = rhs.strides_;
                    rhs.strides_ = shape_type{};
                    this->axistags_ = rhs.axistags_;
                    rhs.shape_ = axistags_type{};
                    this->data_ = rhs.data_;
                    rhs.data_ = nullptr;
                    this->flags_ = rhs.flags_;
                    rhs.flags_ = 0;
                    allocated_data_ = std::move(rhs.allocated_data_);
                }
            }
            return *this;
        }

        template <class E,
                  VIGRA_REQUIRE<(!std::is_convertible<std::decay_t<E>, value_type const &>::value)>>
        self_type& operator=(const xt::xexpression<E>& ex)
        {
            E const & e = ex.derived_cast();
            if(this->can_broadcast_rhs(e.shape()))
            {
                this->assign_impl(e);
            }
            else
            {
                array_nd(e).swap(*this);
            }
            return *this;
        }

            /** Assignment of a scalar.
             */
        array_nd & operator=(const_reference v)
        {
            view_type::operator=(v);
            return *this;
        }

        template <class SHAPE>
        void
        resize(SHAPE const & new_shape,
               axistags_type const & new_axistags = axistags_type{},
               tags::memory_order order = c_order)
        {
            vigra_precondition(all_greater_equal(new_shape, 0),
                "array_nd::resize(): invalid shape.");
            if(new_axistags.size() != new_shape.size())
            {
                new_axistags = axistags_type(new_shape.size(), tags::axis_unknown);
            }
            if(this->size() == xt::compute_size(new_shape))
            {
                this->reshape(new_shape, new_axistags, order);
           }
            else
            {
                array_nd(new_shape, new_axistags, order).swap(*this);
            }
        }

        void
        resize(shape_type const & new_shape,
               tags::memory_order order)
        {
            resize(new_shape, axistags_type{}, order);
        }

        void swap(array_nd & rhs)
        {
            this->swap_impl(rhs);
            allocated_data_.swap(rhs.allocated_data_);
        }

            /** get the allocator.
             */
        allocator_type get_allocator() const
        {
            return allocated_data_.get_allocator();
        }
    };

    template <class T, index_t N, class A>
    inline void
    swap(array_nd<T,N,A> & a1, array_nd<T,N,A> & a2)
    {
        a1.swap(a2);
    }

} // namespace xvigra

#endif // XVIGRA_ARRAY_ND_HPP
