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

#include "global.hpp"
#include "concepts.hpp"
#include "error.hpp"
#include "math.hpp"
#include "tiny_vector.hpp"

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
    }

    template <xvigra::index_t N, class T>
    struct xcontainer_inner_types<xvigra::view_nd<N, T>>
    {
        using temporary_type = xvigra::view_nd<N, T>;
    };

    template <xvigra::index_t N, class T>
    struct xiterable_inner_types<xvigra::view_nd<N, T>>
    {
        using inner_shape_type = xvigra::shape_t<N>;
        using stepper = xindexed_stepper<xvigra::view_nd<N, T>, false>;
        using const_stepper = xindexed_stepper<xvigra::view_nd<N, T>, true>;
    };
}

namespace xvigra
{
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
    }

    /***********/
    /* view_nd */
    /***********/

    template <index_t N, class T>
    class view_nd
    : public xt::xiterable<view_nd<N, T>>
    , public xt::xview_semantic<view_nd<N, T>>
    {

      public:

        enum flags_t { consecutive_memory_flag = 1,
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
        // using iterator               = value_type *;
        // using const_iterator         = const_value_type *;
        // using reverse_iterator       = std::reverse_iterator<iterator>;
        // using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;
        using shape_type             = shape_t<internal_dimension>;
        using strides_type           = shape_type;
        using axistags_type          = tiny_vector<tags::axis_tag, internal_dimension>;
        // using container_type         = T[N];
        // static constexpr xt::layout_type static_layout = xt::layout_type::row_major;
        // static constexpr bool contiguous_layout = true;

        using self_type = view_nd<N, T>;
        using semantic_base = xt::xview_semantic<self_type>;

        using inner_shape_type = shape_type;
        using inner_strides_type = inner_shape_type;

        using iterable_base = xt::xiterable<self_type>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

      protected:

        unsigned is_consecutive_impl() const
        {
            return (size() == 0 || (char*)&operator[](shape_ - 1) == data_ + (size()-1)*sizeof(T))
                         ? consecutive_memory_flag
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
            for (index_t k = 0; k < dimension(); ++k)
            {
                if (shape_[k] == 1)
                {
                    strides_[k] = 0;
                }
            }
        }

        shape_type shape_;
        shape_type strides_;
        axistags_type axistags_;
        char * data_;
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
        : shape_(other.shape())
        , strides_(other.byte_strides())
        , axistags_(other.axistags())
        , data_((char*)other.raw_data())
        , flags_(other.flags() & ~owns_memory_flag)
        {}

        template <index_t M>
        view_nd(view_nd<M, T> const & other)
        : shape_(other.shape())
        , strides_(other.byte_strides())
        , axistags_(other.axistags())
        , data_((char*)other.raw_data())
        , flags_(other.flags() & ~owns_memory_flag)
        {
            // static_assert(CompatibleDimensions<M, N>::value,  // FIXME
            //     "view_nd<N>(view_nd<M>): ndim mismatch.");
        }

        template <index_t M>
        view_nd(shape_t<M> const & shape)
        : shape_(shape)
        {}

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
                next, measured in units if `sizeof(T)`) for every dimension,
                and pointer.
             */
        view_nd(shape_type const & shape,
                tags::byte_strides_proxy<N> const & strides,
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
                axistags_type   const & axistags,
                const_pointer ptr)
        : shape_(shape)
        , strides_(strides*sizeof(T))
        , axistags_(axistags)
        , data_((char*)ptr)
        , flags_(is_consecutive_impl())
        {
            XVIGRA_ASSERT_MSG(all_greater_equal(shape, 0),
                "view_nd(): invalid shape.");
            zero_singleton_strides();
        }

            /** Construct from shape, byte strides (offset of a sample to the
                next, measured in bytes), axistags for every dimension, and pointer.
             */
        view_nd(shape_type const & shape,
                tags::byte_strides_proxy<N> const & strides,
                axistags_type const & axistags,
                const_pointer ptr)
        : shape_(shape)
        , strides_(strides.value)
        , axistags_(axistags)
        , data_((char*)ptr)
        , flags_(is_consecutive_impl())
        {
            XVIGRA_ASSERT_MSG(all_greater_equal(shape, 0),
                "view_nd(): invalid shape.");
            zero_singleton_strides();
        }

        template <class E>
        self_type& operator=(const xt::xexpression<E>& e)
        {
            vigra_precondition(shape() == e.shape(),
                "view_nd::operator=(): shape mismatch.");
            return semantic_base::operator=(std::forward<E>(e));
        }


            // needed for operator==
        template <class It>
        reference element(It first, It last)
        {
            XVIGRA_ASSERT_MSG(std::distance(first, last) == dimension(),
                "view_nd::element(): invalid index.");
            return *(pointer)(data_ + std::inner_product(first, last, strides_.begin(), 0l));
        }

            // needed for operator==
        template <class It>
        const_reference element(It first, It last) const
        {
            XVIGRA_ASSERT_MSG(std::distance(first, last) == dimension(),
                "view_nd::element(): invalid index.");
            return *(const_pointer)(data_ + std::inner_product(first, last, strides_.begin(), 0l));
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
        bool broadcast_shape(S& s) const
        {
            // FIXME: S here is svector
            return xt::broadcast_shape(shape(), s);
        }

            // needed for semantic_base::assign(expr)
        template <class S>
        constexpr bool is_trivial_broadcast(const S& str) const noexcept
        {
            return true; // FIXME: check
        }

            // needed for operator==
        void reshape(const shape_type& s)
        {
            vigra_precondition(s == shape(),
                "view_nd::reshape(): invalid target shape.");
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
            return *(pointer)(data_ + dot(d, strides_));
        }

            /** Access element via scalar index. Only allowed if
                <tt>is_consecutive() == true</tt> or <tt>dimension() <= 1</tt>.
             */
        reference operator[](size_type i)
        {
            if(is_consecutive())
                return *(pointer)(data_ + i*sizeof(T));
            if(dimension() <= 1)
                return *(pointer)(data_ + i*strides_[0]);
            vigra_precondition(false,
                "view_nd::operator[](int) forbidden for strided multi-dimensional arrays.");
        }

            /** Get element.
             */
        const_reference operator[](shape_type const & d) const
        {
            XVIGRA_ASSERT_INSIDE(d);
            return *(const_pointer)(data_ + dot(d, strides_));
        }

            /** Get element via scalar index. Only allowed if
                <tt>is_consecutive() == true</tt> or <tt>dimension() <= 1</tt>.
             */
        const_reference operator[](size_type i) const
        {
            if(is_consecutive())
                return *(const_pointer)(data_ + i*sizeof(T));
            if(dimension() <= 1)
                return *(const_pointer)(data_ + i*strides_[0]);
            vigra_precondition(false,
                "view_nd::operator[](int) forbidden for strided multi-dimensional arrays.");
        }

            /** Access the array's first element.
             */
        reference operator()()
        {
            return *(pointer)data_;
        }

            /** 1D array access. Use only if <tt>dimension() <= 1</tt>.
             */
        reference operator()(index_t i)
        {
            XVIGRA_ASSERT_MSG(dimension() <= 1,
                          "view_nd::operator()(int): only allowed if dimension() <= 1");
            return *(pointer)(data_ + i*strides_[dimension()-1]);
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
            return *(pointer)(data_ + dot(shape_t<M>{i0, i1, i...}, strides_));
        }

            /** Access the array's first element.
             */
        const_reference operator()() const
        {
            return *(const_pointer)data_;
        }

            /** 1D array access. Use only if <tt>dimension() <= 1</tt>.
             */
        const_reference operator()(index_t i) const
        {
            XVIGRA_ASSERT_MSG(dimension() <= 1,
                          "view_nd::operator()(int): only allowed if dimension() <= 1");
            return *(const_pointer)(data_ + i*strides_[dimension()-1]);
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
            return *(const_pointer)(data_ + dot(shape_t<M>{i0, i1, i...}, strides_));
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
        view_nd<((N < 0) ? runtime_size : N-1), T>
        bind(int axis, index_t index) const
        {
            using view_t = view_nd<((N < 0) ? runtime_size : N-1), T>;

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
                              tags::byte_strides = strides_.erase(axis),
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
        view_nd<((N < 0) ? runtime_size : N-M), T>
        bind(shape_t<M> const & axes, shape_t<M> const & indices) const
        {
            static_assert(N == runtime_size || M <= N,
                "view_nd::bind(shape_t<M>): M <= N required.");
            return bind(axes.back(), indices.back())
                      .bind(axes.pop_back(), indices.pop_back());
        }


        view_nd<((N < 0) ? runtime_size : N-1), T>
        bind(shape_t<1> const & a, shape_t<1> const & i) const
        {
            return bind(a[0], i[0]);
        }

        view_nd const &
        bind(shape_t<0> const &, shape_t<0> const &) const
        {
            return *this;
        }

        view_nd<runtime_size, T>
        bind(shape_t<runtime_size> const & axes, shape_t<runtime_size> const & indices) const
        {
            vigra_precondition(axes.size() == indices.size(),
                "view_nd::bind(): size mismatch between 'axes' and 'indices'.");
            vigra_precondition(axes.size() <= dimension(),
                "view_nd::bind(): axes.size() <= dimension() required.");

            view_nd<runtime_size, T> a(*this);
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
        auto
        bind_left(shape_t<M> const & indices) const -> decltype(this->bind(indices, indices))
        {
            return bind(shape_t<M>::range((index_t)indices.size()), indices);
        }

            /** Bind the last 'indices.size()' dimensions to 'indices'.

                Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
             */
        template <index_t M>
        auto
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
        view_nd<((N < 0) ? runtime_size : N-1), T>
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
        view_nd<N, typename U::value_type>
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
        view_nd<(N == runtime_size ? runtime_size : N+1), typename U::value_type>
        expand_elements(index_t d) const
        {
            using value_t = typename T::value_type;
            using view_t  = view_nd<(N == runtime_size ? runtime_size : N + 1), value_t>;

            vigra_precondition(0 <= d && d <= dimension(),
                "view_nd::expand_elements(d): 0 <= 'd' <= dimension() required.");

            constexpr index_t s = T::static_size;
            return view_t(shape_.insert(d, s),
                          tags::byte_strides = strides_.insert(d, sizeof(value_t)),
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
        view_nd<runtime_size, typename U::value_type>
        ensure_channel_axis(index_t d) const
        {
            return expand_elements(d);
        }

        template <class U=T,
                  VIGRA_REQUIRE<std::is_arithmetic<U>::value>>
        view_nd<runtime_size, T>
        ensure_channel_axis(index_t d) const
        {
            vigra_precondition(d >= 0,
                "view_nd::ensure_channel_axis(d): d >= 0 required.");
            int c = channel_axis();
            if(c == d)
                return *this;
            if(c < 0)
                return newaxis(d, tags::axis_c);
            vigra_precondition(d < dimension(),
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
        view_nd <(N < 0) ? runtime_size : N+1, T>
        newaxis(index_t i,
                tags::axis_tag tag = tags::axis_unknown) const
        {
            using view_t = view_nd<(N < 0) ? runtime_size : N+1, T>;
            return view_t(shape_.insert(i, 1), tags::byte_strides = strides_.insert(i, sizeof(T)),
                          axistags_.insert(i, tag), raw_data());
        }

            // /** create a multiband view for this array.

                // The type <tt>view_nd<N, Multiband<T> ></tt> tells VIGRA
                // algorithms which recognize the <tt>Multiband</tt> modifier to
                // interpret the outermost (last) dimension as a channel dimension.
                // In effect, these algorithms will treat the data as a set of
                // (N-1)-dimensional arrays instead of a single N-dimensional array.
            // */
        // view_nd<N, Multiband<value_type>, StrideTag> multiband() const
        // {
            // return view_nd<N, Multiband<value_type>, StrideTag>(*this);
        // }

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
        view_nd<1, T> diagonal() const
        {
            return view_nd<1, T>(shape_t<1>{min(shape_)},
                                 tags::byte_strides = shape_t<1>{sum(strides_)},
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
            for(int k=0; k<dimension(); ++k)
            {
                if(p[k] < 0)
                    p[k] += shape_[k];
                if(q[k] < 0)
                    q[k] += shape_[k];
            }
            vigra_precondition(is_inside(p) && all_less_equal(p, q) && all_less_equal(q, shape_),
                "view_nd::subarray(): invalid subarray limits.");
            const index_t offset = dot(strides_, p);
            return view_nd(q - p, tags::byte_strides = strides_, axistags_, (const_pointer)(data_ + offset));
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
        view_nd<N, T>
        transpose() const
        {
            return view_nd<N, T>(reversed(shape_),
                                 tags::byte_strides = reversed(strides_),
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
                        tags::byte_strides = transposed(strides_, p),
                        transposed(axistags_, p),
                        raw_data());
            return res;
        }

        view_nd
        transpose(tags::memory_order order) const
        {
            return transpose(detail::permutation_to_order(strides_, order));
        }

        template <index_t M = N>
        view_nd<M, T> view()
        {
            static_assert(M == runtime_size || N == runtime_size || M == N,
                "view_nd::view(): desired dimension is incompatible with dimension().");
            vigra_precondition(M == runtime_size || M == dimension(),
                "view_nd::view(): desired dimension is incompatible with dimension().");
            return view_nd<M, T>(shape_t<M>(shape_.begin(), shape_.begin()+dimension()),
                                 tags::byte_strides = shape_t<M>(strides_.begin(), strides_.begin()+dimension()),
                                 axis_tags<M>(axistags_.begin(), axistags_.begin()+dimension()),
                                 raw_data());
        }

        template <index_t M = N>
        view_nd<M, const_value_type> view() const
        {
            return this->template view<M>();
        }

        template <index_t M = N>
        view_nd<M, const_value_type> cview() const
        {
            static_assert(M == runtime_size || N == runtime_size || M == N,
                "view_nd::cview(): desired dimension is incompatible with dimension().");
            vigra_precondition(M == runtime_size || M == dimension(),
                "view_nd::cview(): desired dimension is incompatible with dimension().");
            return view_nd<M, const_value_type>(
                        shape_t<M>(shape_.begin(), shape_.begin()+dimension()),
                        tags::byte_strides = shape_t<M>(strides_.begin(), strides_.begin()+dimension()),
                        axis_tags<M>(axistags_.begin(), axistags_.begin()+dimension()),
                        raw_data());
        }

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

        // pointer data()
        // {
        //     return (pointer)data_;
        // }

        // const_pointer data() const
        // {
        //     return (pointer)data_;
        // }

        const shape_type & shape() const
        {
            return shape_;
        }

        shape_type const & byte_strides() const
        {
            return strides_;
        }

        shape_type strides() const
        {
            return strides_ / sizeof(value_type);
        }

            /** number of the elements in the array.
             */
        index_t size() const
        {
            return max(0, prod(shape_));
        }

    #ifdef DOXYGEN
            /** the array's number of dimensions.
             */
        std::size_t dimension() const;
    #else
            // Actually, we use some template magic to turn dimension() into a
            // constexpr when it is known at compile time.
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
    #endif


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
            * Returns true iff this view refers to consecutive memory.
            */
        bool is_consecutive() const
        {
            return (flags_ & consecutive_memory_flag) != 0;
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
            return tiny_vector<char *, 2>{ data_, (char*)(1 + &(*this)[shape() - 1]) };
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

        int channel_axis() const
        {
            for(int k=0; k<dimension(); ++k)
                if(axistags_[k] == tags::axis_c)
                    return k;
            return tags::axis_missing;
        }

        int axis_index(tags::axis_tag tag) const
        {
            for(int k=0; k<dimension(); ++k)
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
    };

    template <index_t N, class T>
    inline auto
    transpose(view_nd<N, T> const & array)
    {
        return array.transpose();
    }

    template <index_t N, class T>
    inline auto
    transpose(view_nd<N, T> & array)
    {
        return array.transpose();
    }

    template <index_t N, class T, class A>
    inline auto
    transpose(array_nd<N, T, A> const & array)
    {
        return array.transpose();
    }

    template <index_t N, class T, class A>
    inline auto
    transpose(array_nd<N, T, A> & array)
    {
        return array.transpose();
    }

    // template <int N, class T>
    // inline void
    // swap(view_nd<N,T> & array1, view_nd<N,T> & array2)
    // {
    //     array1.swap(array2);
    // }

    /************/
    /* array_nd */
    /************/

    template <index_t N, class T, class Alloc>
    class array_nd
    : public view_nd<N, T>
    {
      public:
        using view_type = view_nd<N, T>;
        using buffer_type = std::vector<typename view_type::value_type, Alloc>;
        using allocator_type = Alloc;
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

        // using self_type = array_nd<N, T, Alloc>;
        // using semantic_base = xt::xview_semantic<self_type>;
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
        : view_type(shape, 0, order)
        , allocated_data_(this->size(), init, alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

        template <class E,
                  VIGRA_REQUIRE<is_xexpression<E>::value && !tensor_concept<E>::value>>
        array_nd(E && e)
        : view_type(e.shape(), 0)
        , allocated_data_(this->size())
        {
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
            semantic_base::assign(std::forward<E>(e));
        }


        //     /** construct from shape with an initial value
        //      */
        // array_nd(shape_type const & shape,
        //         axistags_type const & axistags,
        //         const_reference init,
        //         tags::memory_order order = c_order,
        //         allocator_type const & alloc = allocator_type())
        // : view_type(shape, axistags, 0, order)
        // , allocated_data_(this->size(), init, alloc)
        // {
        //     vigra_precondition(all_greater_equal(shape, 0),
        //         "array_nd(): invalid shape.");
        //     this->data_  = (char*)&allocated_data_[0];
        //     this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        // }

            // /** construct from shape and initialize with a linear sequence in scan order
                // (i.e. first pixel gets value 0, second on gets value 1 and so on).
             // */
        // array_nd (const shape_type &shape, MultiArrayInitializationTag init,
                    // allocator_type const & alloc = allocator_type());

        //     /** construct from shape and copy values from the given C array
        //      */
        // array_nd(shape_type const & shape,
        //          const_pointer init,
        //          tags::memory_order order = c_order,
        //          allocator_type const & alloc = allocator_type())
        // : view_type(shape, 0, order)
        // , allocated_data_(init, init + this->size(), alloc)
        // {
        //     vigra_precondition(all_greater_equal(shape, 0),
        //         "array_nd(): invalid shape.");
        //     this->data_  = (char*)&allocated_data_[0];
        //     this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        // }

        //     /** construct from shape and axistags and copy values from the given C array
        //      */
        // array_nd(shape_type const & shape,
        //         axistags_type const & axistags,
        //         const_pointer init,
        //         tags::memory_order order = c_order,
        //         allocator_type const & alloc = allocator_type())
        // : view_type(shape, axistags, 0, order)
        // , allocated_data_(init, init + this->size(), alloc)
        // {
        //     vigra_precondition(all_greater_equal(shape, 0),
        //         "array_nd(): invalid shape.");
        //     this->data_  = (char*)&allocated_data_[0];
        //     this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        // }

        //     /** construct from shape and copy values from the
        //         given initializer_list
        //      */
        // array_nd(shape_type const & shape,
        //         std::initializer_list<T> init,
        //         tags::memory_order order = c_order,
        //         allocator_type const & alloc = allocator_type())
        // : view_type(shape, 0, order)
        // , allocated_data_(init, alloc)
        // {
        //     vigra_precondition(all_greater_equal(shape, 0),
        //         "array_nd(): invalid shape.");
        //     vigra_precondition(this->size() == init.size(),
        //         "array_nd(): initializer_list has wrong size.");
        //     this->data_  = (char*)&allocated_data_[0];
        //     this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        // }

        //     /** construct from shape and axistags and copy values from the
        //         given initializer_list
        //      */
        // array_nd(shape_type const & shape,
        //         axistags_type const & axistags,
        //         std::initializer_list<T> init,
        //         tags::memory_order order = c_order,
        //         allocator_type const & alloc = allocator_type())
        // : view_type(shape, axistags, 0, order)
        // , allocated_data_(init, alloc)
        // {
        //     vigra_precondition(all_greater_equal(shape, 0),
        //         "array_nd(): invalid shape.");
        //     vigra_precondition(this->size() == init.size(),
        //         "array_nd(): initializer_list has wrong size.");
        //     this->data_  = (char*)&allocated_data_[0];
        //     this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        // }

        //     /** construct 1D-array from initializer_list
        //      */
        // template<index_t M = N,
        //          VIGRA_REQUIRE<(M == 1 || M == runtime_size)> >
        // array_nd(std::initializer_list<T> init,
        //         allocator_type const & alloc = allocator_type())
        // : view_type(shape_t<1>(init.size()), 0, c_order)
        // , allocated_data_(init, alloc)
        // {
        //     this->data_  = (char*)&allocated_data_[0];
        //     this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        // }

            /** copy constructor
             */
        array_nd(array_nd const & rhs)
        : view_type(rhs)
        , allocated_data_(rhs.allocated_data_)
        {
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** move constructor
             */
        array_nd(array_nd && rhs)
        : view_type()
        , allocated_data_(std::move(rhs.allocated_data_))
        {
            this->swap_impl(rhs);
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** construct by copying from a view_nd
             */
        template <index_t M, class U>
        array_nd(view_nd<M, U> const & rhs,
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
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

        //     /** Constructor from a temporary array expression.
        //     */
        // template<class ARG>
        // array_nd(ArrayMathExpression<ARG> && rhs,
        //         tags::memory_order order = c_order,
        //         allocator_type const & alloc = allocator_type())
        // : view_type(rhs.shape(), 0, order)
        // , allocated_data_(alloc)
        // {
        //     allocated_data_.reserve(this->size());

        //     if (order != c_order)
        //     {
        //         auto p = detail::permutationToOrder(this->byte_strides(), c_order);
        //         rhs.transpose_inplace(p);
        //     }

        //     typedef typename std::remove_reference<ArrayMathExpression<ARG>>::type RHS;
        //     using U = typename RHS::value_type;
        //     buffer_type & data = allocated_data_;
        //     universalPointerNDFunction(rhs, rhs.shape(),
        //         [&data](U const & u)
        //         {
        //             data.emplace_back(detail::RequiresExplicitCast<T>::cast(u));
        //         });

        //     this->data_ = (char*)&allocated_data_[0];
        //     this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        // }
    };

#if 0


/********************************************************/
/*                                                      */
/*                      view_nd                     */
/*                                                      */
/********************************************************/

    // Note: Strides are internally stored in units of bytes, whereas the
    // external API always measures strides in units of `sizeof(T)`, unless
    // byte-strides are explicitly enforced by calling the `byte_strides()`
    // member function or by passing strides via the `tags::byte_strides`
    // keyword argument. Byte-strides allow for a more flexible mapping between
    // numpy and vigra arrays.
    template <int N, class T>
    class view_nd
    : public array_ndTag
    {
      protected:
        enum Flags { consecutive_memory_flag = 1,
                     owns_memory_flag = 2
                   };

      public:
            /** the array's nominal dimensionality N.
             */
        static const int dimension = N;

            /** the array's actual dimensionality.
                This ensures that view_nd can also be used for
                scalars (that is, when <tt>N == 0</tt>). Calculated as:<br>
                \code
                internal_dimension = (N==0) ? 1 : N
                \endcode
             */
        static const int internal_dimension = (N==0) ? 1 : N;

            /** the array's value type
             */
        typedef T value_type;

            /** the read-only variants of the array's value type
             */
        typedef typename std::add_const<value_type>::type  const_value_type;

            /** reference type (result of operator[])
             */
        typedef value_type &reference;

            /** const reference type (result of operator[] const)
             */
        typedef const_value_type &const_reference;

            /** pointer type
             */
        typedef value_type *pointer;

            /** const pointer type
             */
        typedef const_value_type *const_pointer;

            /** difference type (used for multi-dimensional offsets and indices)
             */
        typedef shape_t<internal_dimension> shape_type;

            /** key type (argument of index operator array[i] -- same as shape_type)
             */
        typedef shape_type key_type;

            /** size type
             */
        typedef shape_type size_type;

            /** difference and index type for a single dimension
             */
        typedef index_t index_t;

            /** type for tags::axis_tag of every dimension
             */
        typedef tiny_vector<tags::axis_tag, internal_dimension> axistags_type;

            /** the array view's own type
             */
        typedef view_nd view_type;

            /** the array's pointer_nd type
             */
        typedef PointerND<N, value_type> pointer_nd_type;

            /** the array's const pointer_nd type
             */
        typedef PointerND<N, const_value_type> const_pointer_nd_type;

             /** scan-order iterator (array_ndIterator) type
             */
        typedef array_ndIterator<internal_dimension, T> iterator;

            /** const scan-order iterator (array_ndIterator) type
             */
        typedef array_ndIterator<internal_dimension, const_value_type> const_iterator;

            // /** the matrix type associated with this array.
             // */
        // typedef array_nd <N, T> matrix_type;

      protected:

        typedef typename shape_type::value_type diff_zero_t;

            /** the shape of the array pointed to.
            */
        shape_type shape_;

            /** the strides (offset between consecutive elements) for every dimension.
            */
        shape_type strides_;

            /** the axistags for every dimension.
            */
        axistags_type axistags_;

            /** pointer to the array.
             */
        char * data_;

            /** keep track of various properties
             */
        unsigned flags_;

        void assignImpl(view_nd const & rhs)
        {
            if(data_ == 0)
            {
                shape_     = rhs.shape();
                strides_   = rhs.byte_strides();
                axistags_  = rhs.axistags();
                data_      = (char*)rhs.data();
                flags_     = rhs.flags() & ~owns_memory_flag;
            }
            else
            {
                copyImpl(rhs);
            }
        }

        template <index_t M, class U>
        void copyImpl(view_nd<M, U> const & rhs)
        {
            universalarray_ndFunction(*this, rhs,
                [](value_type & v, U const & u)
                {
                    v = detail::RequiresExplicitCast<value_type>::cast(u);
                },
                "view_nd::operator=(view_nd const &)"
            );
        }

            // ensure that singleton axes have zero stride
        void zero_singleton_strides()
        {
            for (int k = 0; k < dimension(); ++k)
                if (shape_[k] == 1)
                    strides_[k] = 0;
        }

        unsigned is_consecutive_impl() const
        {
            return (size() == 0 || (char*)&operator[](shape_ - 1) == data_ + (size()-1)*sizeof(T))
                         ? consecutive_memory_flag
                         : 0;
        }

        void swapImpl(view_nd & rhs)
        {
            if(this != &rhs)
            {
                shape_.swap(rhs.shape_);
                strides_.swap(rhs.strides_);
                axistags_.swap(rhs.axistags_);
                swap(data_, rhs.data_);
                swap(flags_, rhs.flags_);
            }
        }

    public:

            /** default constructor: create an invalid view,
             * i.e. hasData() returns false and size() is zero.
             */
        view_nd()
        : shape_()
        , strides_()
        , axistags_()
        , data_(0)
        , flags_(0)
        {}

        view_nd(view_nd const & other)
        : shape_(other.shape())
        , strides_(other.byte_strides())
        , axistags_(other.axistags())
        , data_((char*)other.data())
        , flags_(other.flags() & ~owns_memory_flag)
        {}

        template <index_t M>
        view_nd(view_nd<M, T> const & other)
        : shape_(other.shape())
        , strides_(other.byte_strides())
        , axistags_(other.axistags())
        , data_((char*)other.data())
        , flags_(other.flags() & ~owns_memory_flag)
        {
            static_assert(CompatibleDimensions<M, N>::value,
                "view_nd<N>(view_nd<M>): ndim mismatch.");
        }

        // FIXME: add constructor from explicit channel to vector pixels

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
                      axistags_type(tags::size = shape.size(), tags::axis_unknown), ptr)
        {}

            /** Construct from shape, strides (offset of a sample to the
                next, measured in units if `sizeof(T)`) for every dimension,
                and pointer.
             */
        view_nd(shape_type const & shape,
                    tags::byte_strides_proxy<N> const & strides,
                    const_pointer ptr)
        : view_nd(shape, strides,
                      axistags_type(tags::size = shape.size(), tags::axis_unknown), ptr)
        {}

            /** Construct from shape, strides (offset of a sample to the
                next, measured in units if `sizeof(T)`), axistags for every
                dimension, and pointer.
             */
        view_nd(shape_type const & shape,
                    shape_type const & strides,
                    axistags_type   const & axistags,
                    const_pointer ptr)
        : shape_(shape)
        , strides_(strides*sizeof(T))
        , axistags_(axistags)
        , data_((char*)ptr)
        , flags_(is_consecutive_impl())
        {
            XVIGRA_ASSERT_MSG(all_greater_equal(shape, 0),
                "view_nd(): invalid shape.");
            zero_singleton_strides();
        }

            /** Construct from shape, byte strides (offset of a sample to the
                next, measured in bytes), axistags for every dimension, and pointer.
             */
        view_nd(shape_type const & shape,
                    tags::byte_strides_proxy<N> const & strides,
                    axistags_type const & axistags,
                    const_pointer ptr)
        : shape_(shape)
        , strides_(strides.value)
        , axistags_(axistags)
        , data_((char*)ptr)
        , flags_(is_consecutive_impl())
        {
            XVIGRA_ASSERT_MSG(all_greater_equal(shape, 0),
                "view_nd(): invalid shape.");
            zero_singleton_strides();
        }

            /* Construct 0-dimensional array from 0-dimensional shape/stride
               (needed in functions recursing on dimension()).
             */
        view_nd(shape_t<0> const &,
                    tags::byte_strides_proxy<0> const &,
                    tiny_vector<tags::axis_tag, 0> const &,
                    const_pointer ptr)
        : shape_{1}
        , strides_{sizeof(T)}
        , axistags_{tags::axis_unknown}
        , data_((char*)ptr)
        , flags_(consecutive_memory_flag)
        {
            static_assert(N <= 0,
                "view_nd(): 0-dimensional constructor can only be called when N == 0 or N == runtime_size.");
        }

            /** Assignment. There are 3 cases:

                <ul>
                <li> When this <tt>view_nd</tt> does not point to valid data
                     (e.g. after default construction), it becomes a new view of \a rhs.
                <li> Otherwise, when the shapes of the two arrays match, the contents
                     (i.e. the elements) of \a rhs are copied.
                <li> Otherwise, a <tt>PreconditionViolation</tt> exception is thrown.
                </ul>
             */
        view_nd & operator=(view_nd const & rhs)
        {
            if(this != &rhs)
                assignImpl(rhs);
            return *this;
        }

            /** Init with given value.
             */
        view_nd & init(value_type const & u)
        {
            return operator=(u);
        }

    #ifdef DOXYGEN
            /** Assignment of a scalar.
             */
        view_nd &
        operator=(value_type const & u);

            /** Assignment of a differently typed array or an array expression. It copies the elements
                of\a rhs or fails with a <tt>PreconditionViolation</tt> exception when
                the shapes do not match.
             */
        template<class ARRAY_LIKE>
        view_nd & operator=(ARRAY_LIKE const & rhs);

            /** Add-assignment of a differently typed array or an array expression.
                It adds the elements of \a rhs or fails with a <tt>PreconditionViolation</tt>
                exception when the shapes do not match.
             */
        template <class ARRAY_LIKE>
        view_nd & operator+=(ARRAY_LIKE const & rhs);

            /** Subtract-assignment of a differently typed array or an array expression.
                It subtracts the elements of \a rhs or fails with a <tt>PreconditionViolation</tt>
                exception when the shapes do not match.
             */
        template <class ARRAY_LIKE>
        view_nd & operator-=(ARRAY_LIKE const & rhs);

            /** Multiply-assignment of a differently typed array or an array expression.
                It multiplies with the elements of \a rhs or fails with a
                <tt>PreconditionViolation</tt> exception when the shapes do not match.
             */
        template <class ARRAY_LIKE>
        view_nd & operator*=(ARRAY_LIKE const & rhs);

            /** Divide-assignment of a differently typed array or an array expression.
                It divides by the elements of \a rhs or fails with a <tt>PreconditionViolation</tt>
                exception when the shapes do not match.
             */
        template <class ARRAY_LIKE>
        view_nd & operator/=(ARRAY_LIKE const & rhs);

            /** Add-assignment of a scalar.
             */
        view_nd & operator+=(value_type const & u);

            /** Subtract-assignment of a scalar.
             */
        view_nd & operator-=(value_type const & u);

            /** Multiply-assignment of a scalar.
             */
        view_nd & operator*=(value_type const & u);

            /** Divide-assignment of a scalar.
             */
        view_nd & operator/=(value_type const & u);

    #else

    #define VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(OP) \
        template <index_t M, class U> \
        view_nd & \
        operator OP(view_nd<M, U> const & rhs) \
        { \
            static_assert(std::is_convertible<U, value_type>::value, \
                "view_nd::operator" #OP "(view_nd const &): value_types of lhs and rhs are incompatible."); \
                \
            universalarray_ndFunction(*this, rhs, \
                [](value_type & v, U const & u) \
                { \
                    v OP detail::RequiresExplicitCast<value_type>::cast(u); \
                }, \
               "view_nd::operator" #OP "(view_nd const &)" \
            ); \
            return *this; \
        } \
        \
        template <class ARG> \
        view_nd & \
        operator OP(ArrayMathExpression<ARG> && rhs) \
        { \
            typedef typename ArrayMathExpression<ARG>::value_type U; \
            static_assert(std::is_convertible<U, value_type>::value, \
                "view_nd::operator" #OP "(ARRAY_MATH_EXPRESSION const &): value types of lhs and rhs are incompatible."); \
            \
            universalarray_ndFunction(*this, std::move(rhs), \
                [](value_type & v, U const & u) \
                { \
                    v OP detail::RequiresExplicitCast<value_type>::cast(u); \
                }, \
                "view_nd::operator" #OP "(ARRAY_MATH_EXPRESSION const &)" \
            ); \
            return *this; \
        } \
        \
        template <class ARG> \
        view_nd & \
        operator OP(ArrayMathExpression<ARG> const & rhs) \
        { \
            return operator OP(ArrayMathExpression<ARG>(rhs)); \
        } \
        \
        view_nd & operator OP(value_type const & u) \
        { \
            universalarray_ndFunction(*this, [u](value_type & v) { v OP u; }, \
                "view_nd::operator" #OP "(value_type const &)"); \
            return *this; \
        }

        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(+=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(-=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(*=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(/=)

    #undef VIGRA_array_nd_ARITHMETIC_ASSIGNMENT

    #endif // DOXYGEN

            /** Access element.
             */
        reference operator[](shape_type const & d)
        {
            XVIGRA_ASSERT_INSIDE(d);
            return *(pointer)(data_  + dot(d, strides_));
        }

            /** Access element via scalar index. Only allowed if
                <tt>is_consecutive() == true</tt> or <tt>dimension() <= 1</tt>.
             */
        reference operator[](index_t i)
        {
            if(is_consecutive())
                return *(pointer)(data_  + i*sizeof(T));
            if(dimension() <= 1)
                return *(pointer)(data_  + i*strides_[0]);
            vigra_precondition(false,
                "view_nd::operator[](int) forbidden for strided multi-dimensional arrays.");
        }

            /** Get element.
             */
        const_reference operator[](shape_type const & d) const
        {
            XVIGRA_ASSERT_INSIDE(d);
            return *(const_pointer)(data_  + dot(d, strides_));
        }

            /** Get element via scalar index. Only allowed if
                <tt>is_consecutive() == true</tt> or <tt>dimension() <= 1</tt>.
             */
        const_reference operator[](index_t i) const
        {
            if(is_consecutive())
                return *(const_pointer)(data_  + i*sizeof(T));
            if(dimension() <= 1)
                return *(const_pointer)(data_  + i*strides_[0]);
            vigra_precondition(false,
                "view_nd::operator[](int) forbidden for strided multi-dimensional arrays.");
        }

            /** 1D array access. Use only if <tt>dimension() <= 1</tt>.
             */
        reference operator()(index_t i)
        {
            XVIGRA_ASSERT_MSG(dimension() <= 1,
                          "view_nd::operator()(int): only allowed if dimension() <= 1");
            return *(pointer)(data_  + i*strides_[0]);
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
            return *(pointer)(data_  + dot(shape_t<M>(i0, i1, i...), strides_));
        }

            /** 1D array access. Use only if <tt>dimension() <= 1</tt>.
             */
        const_reference operator()(index_t i) const
        {
            XVIGRA_ASSERT_MSG(dimension() <= 1,
                          "view_nd::operator()(int): only allowed if dimension() <= 1");
            return *(const_pointer)(data_  + i*strides_[0]);
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
            return *(const_pointer)(data_  + dot(shape_t<M>(i0, i1, i...), strides_));
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
        view_nd<((N < 0) ? runtime_size : N-1), T>
        bind(int axis, index_t index) const
        {
            typedef view_nd<((N < 0) ? runtime_size : N-1), T> Result;

            XVIGRA_ASSERT_MSG(0 <= axis && axis < dimension() && 0 <= index && index < shape_[axis],
                "view_nd::bind(): index out of range.");

            shape_type point(tags::size = dimension(), 0);
            point[axis] = index;
            if (dimension() == 1)
            {
                shape_t<Result::internal_dimension> shape{ 1 }, strides{ 1 };
                tiny_vector<tags::axis_tag, Result::internal_dimension> axistags{ tags::axis_unknown };
                return Result(shape, strides, axistags, &operator[](point));
            }
            else
            {
                return Result(shape_.erase(axis),
                              tags::byte_strides = strides_.erase(axis),
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
        view_nd<((N < 0) ? runtime_size : N-M), T>
        bind(shape_t<M> const & axes, shape_t<M> const & indices) const
        {
            static_assert(N == runtime_size || M <= N,
                "view_nd::bind(shape_t<M>): M <= N required.");
            return bind(axes.back(), indices.back())
                      .bind(axes.pop_back(), indices.pop_back());
        }


        view_nd<((N < 0) ? runtime_size : N-1), T>
        bind(shape_t<1> const & a, shape_t<1> const & i) const
        {
            return bind(a[0], i[0]);
        }

        view_nd const &
        bind(shape_t<0> const &, shape_t<0> const &) const
        {
            return *this;
        }

        view_nd<runtime_size, T>
        bind(shape_t<runtime_size> const & axes, shape_t<runtime_size> const & indices) const
        {
            vigra_precondition(axes.size() == indices.size(),
                "view_nd::bind(): size mismatch between 'axes' and 'indices'.");
            vigra_precondition(axes.size() <= dimension(),
                "view_nd::bind(): axes.size() <= dimension() required.");

            view_nd<runtime_size, T> a(*this);
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
        auto
        bind_left(shape_t<M> const & indices) const -> decltype(this->bind(indices, indices))
        {
            return bind(shape_t<M>::range(indices.size()), indices);
        }

            /** Bind the last 'indices.size()' dimensions to 'indices'.

                Only applicable when <tt>M <= N</tt> or <tt>N == runtime_size</tt>.
             */
        template <index_t M>
        auto
        bindRight(shape_t<M> const & indices) const -> decltype(this->bind(indices, indices))
        {
            return bind(shape_t<M>::range(indices.size()) + dimension() - indices.size(),
                        indices);
        }

            /** Bind the channel axis to index d.
                This calls <tt>array.bind(array.channel_axis(), d)</tt>
                when a channel axis is defined and throws an error otherwise.
                \endcode
             */
        template <class U=T,
                  VIGRA_REQUIRE<(U::static_size == 1)> >
        view_nd<((N < 0) ? runtime_size : N-1), T>
        bind_channel(index_t d) const
        {
            int m = channel_axis();

            XVIGRA_ASSERT_MSG(m != tags::axis_missing,
                "view_nd::bindChannel(): array has no channel axis.");

            return bind(m, d);
        }

            /** Create a view to channel 'i' of a vector-like value type. Possible value types
                (of the original array) are: \ref TinyVector, \ref RGBValue, \ref FFTWComplex,
                and <tt>std::complex</tt>. The function can be applied whenever the array's
                element type <tt>T</tt> defines an embedded type <tt>T::value_type</tt> which
                becomes the return type of <tt>bindChannel()</tt>.


                <b>Usage:</b>
                \code
                    array_nd<2, RGBValue<float> > rgb_image({h,w});

                    view_nd<2, float> red   = rgb_image.bindChannel(0);
                    view_nd<2, float> green = rgb_image.bindChannel(1);
                    view_nd<2, float> blue  = rgb_image.bindChannel(2);
                \endcode
            */
        template <class U=T,
                  VIGRA_REQUIRE<(NumericTraits<U>::static_size > 1)> >
        view_nd<N, typename NumericTraits<U>::value_type>
        bindChannel(index_t i) const
        {
            vigra_precondition(0 <= i && i < NumericTraits<U>::static_size,
                "view_nd::bindChannel(i): 'i' out of range.");
            return expand_elements(0).bind(0, i);
        }

            /** Create a view where a vector-like element type is expanded into a new
                array dimension. The new dimension is inserted at index position 'd',
                which must be between 0 and N inclusive.

                Possible value types of the original array are: \ref tiny_vector, \ref RGBValue,
                \ref FFTWComplex, <tt>std::complex</tt>, and the built-in number types (in this
                case, <tt>expandElements</tt> is equivalent to <tt>newaxis</tt>).

                <b>Usage:</b>
                \code
                    array_nd<2, tiny_vector<float, 3> > rgb_image({h, w});

                    view_nd<3, float> multiband_image = rgb_image.expandElements(2);
                \endcode
            */
        template <class U=T,
                  VIGRA_REQUIRE<(NumericTraits<U>::static_size > 0)> >
        view_nd<(N == runtime_size ? runtime_size : N+1), typename NumericTraits<U>::value_type>
        expandElements(index_t d) const
        {
            using Value  = typename NumericTraits<T>::value_type;
            using Result = view_nd <(N == runtime_size ? runtime_size : N + 1), Value>;

            vigra_precondition(0 <= d && d <= dimension(),
                "view_nd::expandElements(d): 0 <= 'd' <= dimension() required.");

            static const int s = NumericTraits<T>::static_size;
            return Result(shape_.insert(d, s),
                          tags::byte_strides = strides_.insert(d, sizeof(Value)),
                          axistags_.insert(d, tags::axis_c),
                          reinterpret_cast<Value*>(data_));
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
                     <tt>expandElements(d)</tt> is called.
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
                  VIGRA_REQUIRE<(NumericTraits<U>::static_size > 1)> >
        view_nd<runtime_size, typename NumericTraits<U>::value_type>
        ensure_channel_axis(index_t d) const
        {
            return expandElements(d);
        }

        template <class U=T,
                  VIGRA_REQUIRE<(NumericTraits<U>::static_size == 1)> >
        view_nd<runtime_size, T>
        ensure_channel_axis(index_t d) const
        {
            vigra_precondition(d >= 0,
                "view_nd::ensure_channel_axis(d): d >= 0 required.");
            int c = channel_axis();
            if(c == d)
                return *this;
            if(c < 0)
                return newaxis(d, tags::axis_c);
            vigra_precondition(d < dimension(),
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
        view_nd <(N < 0) ? runtime_size : N+1, T>
        newaxis(index_t i,
                                 tags::axis_tag tag = tags::axis_unknown) const
        {
            typedef view_nd <(N < 0) ? runtime_size : N+1, T> Result;
            return Result(shape_.insert(i, 1), tags::byte_strides = strides_.insert(i, sizeof(T)),
                          axistags_.insert(i, tag), data());
        }
            // /** create a multiband view for this array.

                // The type <tt>view_nd<N, Multiband<T> ></tt> tells VIGRA
                // algorithms which recognize the <tt>Multiband</tt> modifier to
                // interpret the outermost (last) dimension as a channel dimension.
                // In effect, these algorithms will treat the data as a set of
                // (N-1)-dimensional arrays instead of a single N-dimensional array.
            // */
        // view_nd<N, Multiband<value_type>, StrideTag> multiband() const
        // {
            // return view_nd<N, Multiband<value_type>, StrideTag>(*this);
        // }

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
        view_nd<1, T> diagonal() const
        {
            return view_nd<1, T>(shape_t<1>(min(shape_)),
                                     tags::byte_strides = shape_t<1>(sum(strides_)),
                                     data());
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
            for(int k=0; k<dimension(); ++k)
            {
                if(p[k] < 0)
                    p[k] += shape_[k];
                if(q[k] < 0)
                    q[k] += shape_[k];
            }
            vigra_precondition(isInside(p) && allLessEqual(p, q) && allLessEqual(q, shape_),
                "view_nd::subarray(): invalid subarray limits.");
            const index_t offset = dot(strides_, p);
            return view_nd(q - p, tags::byte_strides = strides_, axistags_, (const_pointer)(data_ + offset));
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
        view_nd<N, T>
        transpose() const
        {
            return view_nd<N, T>(reversed(shape_),
                                     tags::byte_strides = reversed(strides_),
                                     reversed(axistags_),
                                     data());
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
            view_nd res(shape_.transpose(p),
                            tags::byte_strides = strides_.transpose(p),
                            axistags_.transpose(p),
                            data());
            return res;
        }

        view_nd
        transpose(tags::memory_order order) const
        {
            return transpose(detail::permutationToOrder(strides_, order));
        }

        /** Check if the array contains only non-zero elements (or if all elements
        are 'true' if the value type is 'bool').
        */
        bool all() const
        {
            bool res = true;
            value_type zero = value_type();
            universalarray_ndFunction(*this,
                [zero, &res](value_type const & v)
                {
                    if (v == zero)
                        res = false;
                }, "view_nd::all()");
            return res;
        }

        /** Check if the array contains only finite values (no Inf or NaN).
        */
        bool all_finite() const
        {
            bool res = true;
            universalarray_ndFunction(*this,
                [&res](value_type const & v)
                {
                    if (!isfinite(v))
                        res = false;
                }, "view_nd::all_finite()");
                return res;
        }

            /** Check if the array contains a non-zero element (or an element
                that is 'true' if the value type is 'bool').
             */
        bool any() const
        {
            bool res = false;
            value_type zero = value_type();
            universalarray_ndFunction(*this,
                [zero, &res](value_type const & v)
                {
                    if(v != zero)
                        res = true;
                }, "view_nd::any()");
            return res;
        }

            /** Find the minimum and maximum element in this array.
                See \ref FeatureAccumulators for a general feature
                extraction framework.
             */
        tiny_vector<T, 2> minmax() const
        {
            tiny_vector<T, 2> res(NumericTraits<T>::max(), NumericTraits<T>::min());
            universalarray_ndFunction(*this,
                [&res](value_type const & v)
                {
                    if(v < res[0])
                        res[0] = v;
                    if(res[1] < v)
                        res[1] = v;
                }, "view_nd::minmax()");
            return res;
        }

            // /** Compute the mean and variance of the values in this array.
                // See \ref FeatureAccumulators for a general feature
                // extraction framework.
             // */
        // template <class U>
        // void meanVariance(U * mean, U * variance) const
        // {
            // typedef typename NumericTraits<U>::RealPromote R;
            // R zero = R();
            // triple<double, R, R> res(0.0, zero, zero);
            // detail::reduceOverMultiArray(traverser_begin(), shape(),
                                         // res,
                                         // detail::MeanVarianceReduceFunctor(),
                                         // MetaInt<internal_dimension-1>());
            // *mean     = res.second;
            // *variance = res.third / res.first;
        // }

            /** Compute the sum of the array elements.

                You must provide the type of the result by an explicit template parameter:
                \code
                array_nd<2, UInt8> A(width, height);

                double sum = A.sum<double>();
                \endcode
             */
        template <typename U = T>
        PromoteType<U> sum(PromoteType<U> res = PromoteType<U>{}) const
        {
            universalarray_ndFunction(*this,
                [&res](value_type const & v)
                {
                    res += v;
                }, "view_nd::sum()");
            return res;
        }

            /** Compute the sum of the array elements over selected axes.

                \arg sums must have the same shape as this array, except for the
                axes along which the sum is to be accumulated. These axes must be
                singletons. Note that you must include <tt>multi_pointoperators.hxx</tt>
                for this function to work.

                <b>Usage:</b>
                \code
                #include <vigra/multi_array.hxx>
                #include <vigra/multi_pointoperators.hxx>

                array_nd<2, double> A(Shape2(rows, cols));
                ... // fill A

                // make the first axis a singleton to sum over the first index
                array_nd<2, double> rowSums(Shape2(1, cols));
                A.sum(rowSums);

                // this is equivalent to
                transformMultiArray(srcMultiArrayRange(A),
                                    destMultiArrayRange(rowSums),
                                    FindSum<double>());
                \endcode
             */
        template <index_t M, class U>
        void sum(view_nd<M, U> sums) const
        {
            vigra_precondition(sums.dimension() == dimension(),
                "view_nd::sum(view_nd): ndim mismatch.");

            universalarray_ndFunction(sums, *this,
                [](U & u, T const & v)
                {
                    u += detail::RequiresExplicitCast<U>::cast(v);
                },
                "view_nd::sum(view_nd)"
            );
        }

        template <class U=T>
        array_nd<N, PromoteType<U>>
        sum(tags::axis_selection_proxy axis) const
        {
            int d = axis.value;
            vigra_precondition(0 <= d && d < dimension(),
                "view_nd::sum(axis): axis out of range.");

            auto s = shape();
            s[d] = 1;
            array_nd<N, PromoteType<U>> res(s);
            sum(res);
            return res;
        }

        template <class U=T>
        array_nd<N, RealPromoteType<U>>
        mean(tags::axis_selection_proxy axis) const
        {
            int d = axis.value;
            vigra_precondition(0 <= d && d < dimension(),
                "view_nd::mean(axis): axis out of range.");

            auto s = shape();
            s[d] = 1;
            array_nd<N, PromoteType<U>> res(s);
            sum(res);
            res /= shape(d);
            return res;
        }

            /** Compute the product of the array elements.

                You must provide the type of the result by an explicit template parameter:
                \code
                array_nd<2, UInt8> A(width, height);

                double prod = A.product<double>();
                \endcode
             */
        template <class U = T>
        PromoteType<U> prod(PromoteType<U> res = PromoteType<U>{1}) const
        {
            universalarray_ndFunction(*this,
                [&res](value_type const & v)
                {
                    res *= v;
                },
                "view_nd::prod()"
            );
            return res;
        }

        void swap(view_nd & rhs)
        {
            vigra_precondition(!owns_memory_flag() && !rhs.owns_memory_flag(),
                "view_nd::swap(): only allowed when views don't own their memory.");
            swapImpl(rhs);
        }

            /** Swap the data between two view_nd objects.

                The shapes of the two array must match. Both array views
                still point to the same memory as before, just the contents
                are exchanged.
            */
        template <index_t M, class U>
        void
        swapData(view_nd<M, U> rhs)
        {
            static_assert(M == N || M == runtime_size || N == runtime_size,
                "view_nd::swapData(): incompatible dimensions.");
            vigra_precondition(shape() == rhs.shape(),
                "view_nd::swapData(): shape mismatch.");
            universalarray_ndFunction(*this, rhs,
                [](value_type & v, U & u)
                {
                    swap(u, v);
                },
                "view_nd::swapData()"
            );
        }

        template <index_t M>
        view_nd<M, T>
        reshape(shape_t<M> new_shape,
                axis_tags<M> new_axistags = axis_tags<M>{},
                tags::memory_order order = c_order) const
        {
            vigra_precondition(is_consecutive(),
                "view_nd::reshape(): only consecutive arrays can be reshaped.");
            if(M <= 1 && new_shape == shape_t<M>{})
            {
                new_shape = shape_t<M>{size()};
            }
            vigra_precondition(prod(new_shape) == size(),
                "view_nd::reshape(): size mismatch between old and new shape.");
            if(new_axistags == axis_tags<M>{})
                new_axistags = axis_tags<M>(tags::size = new_shape.size(), tags::axis_unknown);
            vigra_precondition(M != runtime_size || new_axistags.size() == new_shape.size(),
               "view_nd::reshape(): size mismatch between new shape and axistags.");
            return view_nd<M, T>(new_shape, new_axistags, data(), order);
        }

        template <index_t M>
        view_nd<M, T>
        reshape(shape_t<M> const & new_shape,
                tags::memory_order order) const
        {
            return reshape(new_shape, axis_tags<M>{}, order);
        }

        view_nd<(N == runtime_size ? N : 1), T>
        flatten() const
        {
            return reshape(shape_t<(N == runtime_size ? N : 1)>{});
        }

            /** number of the elements in the array.
             */
        index_t size() const
        {
            return max(0, prod(shape_));
        }

    #ifdef DOXYGEN
            /** the array's number of dimensions.
             */
        int dimension() const;
    #else
            // Actually, we use some template magic to turn dimension() into a
            // constexpr when it is known at compile time.
        template <index_t M = N>
        int dimension(std::enable_if_t<M == runtime_size, bool> = true) const
        {
            return shape_.size();
        }

        template <index_t M = N>
        constexpr int dimension(std::enable_if_t<(M > runtime_size), bool> = true) const
        {
            return N;
        }
    #endif

        template <index_t M>
        bool unifyShape(shape_t<M> & target) const
        {
            return detail::unifyShape(target, shape_);
        }

            /** the array's shape.
             */
        shape_type const & shape() const
        {
            return shape_;
        }

            /** return the array's shape at a certain dimension.
             */
        index_t shape(int n) const
        {
            return shape_[n];
        }

            /** return the array's strides for every dimension.
             */
        shape_type strides() const
        {
            return strides_ / sizeof(T);
        }

            /** return the array's stride at a certain dimension.
             */
        index_t strides(int n) const
        {
            return strides_[n] / sizeof(T);
        }

            /** return the array's strides for every dimension.
             */
        shape_type const & byte_strides() const
        {
            return strides_;
        }

            /** return the array's stride at a certain dimension.
             */
        index_t byte_strides(int n) const
        {
            return strides_[n];
        }

        template <index_t M>
        void principalStrides(shape_t<M> & target) const
        {
            target = strides_;
        }

            /** return the array's axistags for every dimension.
             */
        axistags_type const & axistags() const
        {
            return axistags_;
        }

            /** return the array's tags::axis_tag at a certain dimension.
             */
        tags::axis_tag axistags(int n) const
        {
            return axistags_[n];
        }

            /** check whether the given point is in the array range.
             */
        bool isInside(shape_type const & p) const
        {
            return Box<internal_dimension>(shape_).contains(p);
        }

            /** check whether the given point is not in the array range.
             */
        bool isOutside(shape_type const & p) const
        {
            return !isInside(p);
        }

            /** return the pointer to the image data
             */
        pointer data()
        {
            return (pointer)data_;
        }

            /** return the pointer to the image data
             */
        const_pointer data() const
        {
            return (const_pointer)data_;
        }

            /**
             * Returns true iff this view refers to valid data,
             * i.e. data() is not a NULL pointer. In particular, the function
             * returns `false` when the array was created with the default
             * constructor.
             */
        bool hasData() const
        {
            return data_ != 0;
        }

            /**
            * Returns true iff this view refers to consecutive memory.
            */
        bool is_consecutive() const
        {
            return (flags_ & consecutive_memory_flag) != 0;
        }

            /**
            * Returns true iff this view owns its memory.
            */
        bool owns_memory_flag() const
        {
            return (flags_ & owns_memory_flag) != 0;
        }

            /**
            * Returns the addresses of the first array element and one byte beyond the
            * last array element.
            */
        tiny_vector<char *, 2> memoryRange() const
        {
            return{ data_, (char*)(1 + &(*this)[shape() - 1]) };
        }

        unsigned flags() const
        {
            return flags_;
        }

        view_nd & setAxistags(axistags_type const & t)
        {
            XVIGRA_ASSERT_MSG(t.size() == dimension(),
                "view_nd::setAxistags(): size mismatch.");
            axistags_ = t;
            return *this;
        }

        view_nd & setchannel_axis(int c)
        {
            XVIGRA_ASSERT_MSG(0 <= c && c < dimension(),
                "view_nd::setchannel_axis(): index out of range.");
            axistags_[c] = tags::axis_c;
            return *this;
        }

        int channel_axis() const
        {
            for(int k=0; k<dimension(); ++k)
                if(axistags_[k] == tags::axis_c)
                    return k;
            return tags::axis_missing;
        }

        int axisIndex(tags::axis_tag tag) const
        {
            for(int k=0; k<dimension(); ++k)
                if(axistags_[k] == tag)
                    return k;
            return tags::axis_missing;
        }

        bool hasAxis(tags::axis_tag tag) const
        {
            return axisIndex(tag) != tags::axis_missing;
        }

        bool haschannel_axis() const
        {
            return channel_axis() != tags::axis_missing;
        }

        CoordinateIterator<internal_dimension>
        coordinates(tags::memory_order order = c_order) const
        {
            return CoordinateIterator<internal_dimension>(shape(), order);
        }

        pointer_nd_type pointer_nd()
        {
            return pointer_nd_type(tags::byte_strides = strides_, data());
        }

        pointer_nd_type pointer_nd(shape_type const & permutation)
        {
            return pointer_nd_type(tags::byte_strides = strides_.transpose(permutation), data());
        }

        pointer_nd_type pointer_nd(tags::memory_order order)
        {
            return pointer_nd(detail::permutationToOrder(strides_, order));
        }

        const_pointer_nd_type pointer_nd() const
        {
            return const_pointer_nd_type(tags::byte_strides = strides_, data());
        }

        const_pointer_nd_type pointer_nd(shape_type const & permutation) const
        {
            return const_pointer_nd_type(tags::byte_strides = strides_.transpose(permutation), data());
        }

        const_pointer_nd_type pointer_nd(tags::memory_order order) const
        {
            return const_pointer_nd_type(detail::permutationToOrder(strides_, order));
        }

        const_pointer_nd_type cpointer_nd() const
        {
            return const_pointer_nd_type(tags::byte_strides = strides_, data());
        }

        const_pointer_nd_type cpointer_nd(shape_type const & permutation) const
        {
            return const_pointer_nd_type(tags::byte_strides = strides_.transpose(permutation), data());
        }

        const_pointer_nd_type cpointer_nd(tags::memory_order order) const
        {
            return const_pointer_nd_type(detail::permutationToOrder(strides_, order));
        }

            /** returns a scan-order iterator pointing
                to the first array element.
            */
        iterator begin(tags::memory_order order)
        {
            return iterator(*this, order);
        }

        iterator begin()
        {
            return iterator(*this, detail::permutationToOrder(strides_, f_order));
        }

            /** returns a const scan-order iterator pointing
                to the first array element.
            */
        const_iterator begin(tags::memory_order order) const
        {
            return const_iterator(*this, order);
        }

        const_iterator begin() const
        {
            return const_iterator(*this, detail::permutationToOrder(strides_, f_order));
        }

            /** returns a const scan-order iterator pointing
                to the first array element.
            */
        const_iterator cbegin(tags::memory_order order) const
        {
            return const_iterator(*this, order);
        }

        const_iterator cbegin() const
        {
            return const_iterator(*this, detail::permutationToOrder(strides_, f_order));
        }

            /** returns a scan-order iterator pointing
                beyond the last array element.
            */
        iterator end(tags::memory_order order)
        {
            return begin(order).end();
        }

        iterator end()
        {
            return begin().end();
        }

            /** returns a const scan-order iterator pointing
                beyond the last array element.
            */
        const_iterator end(tags::memory_order order) const
        {
            return begin(order).end();
        }

        const_iterator end() const
        {
            return begin().end();
        }

            /** returns a const scan-order iterator pointing
                beyond the last array element.
            */
        const_iterator cend(tags::memory_order order) const
        {
            return begin(order).end();
        }

        const_iterator cend() const
        {
            return begin().end();
        }

        template <index_t M = N>
        view_nd<M, T> view()
        {
            static_assert(M == runtime_size || N == runtime_size || M == N,
                "view_nd::view(): desired dimension is incompatible with dimension().");
            vigra_precondition(M == runtime_size || M == dimension(),
                "view_nd::view(): desired dimension is incompatible with dimension().");
            return view_nd<M, T>(shape_t<M>(shape_.begin(), shape_.begin()+dimension()),
                                     tags::byte_strides = shape_t<M>(strides_.begin(), strides_.begin()+dimension()),
                                     axis_tags<M>(axistags_.begin(), axistags_.begin()+dimension()),
                                     data());
        }

        template <index_t M = N>
        view_nd<M, const_value_type> view() const
        {
            return this->template view<M>();
        }

        template <index_t M = N>
        view_nd<M, const_value_type> cview() const
        {
            static_assert(M == runtime_size || N == runtime_size || M == N,
                "view_nd::cview(): desired dimension is incompatible with dimension().");
            vigra_precondition(M == runtime_size || M == dimension(),
                "view_nd::cview(): desired dimension is incompatible with dimension().");
            return view_nd<M, const_value_type>(
                        shape_t<M>(shape_.begin(), shape_.begin()+dimension()),
                        tags::byte_strides = shape_t<M>(strides_.begin(), strides_.begin()+dimension()),
                        axis_tags<M>(axistags_.begin(), axistags_.begin()+dimension()),
                        data());
        }
    };

    /********************************************************/
    /*                                                      */
    /*                 view_nd functions                */
    /*                                                      */
    /********************************************************/

    template <int N, class T>
    SquaredNormType<view_nd<N, T> >
    squaredNorm(view_nd<N, T> const & a)
    {
        auto res = SquaredNormType<view_nd<N, T> >();
        universalarray_ndFunction(a,
            [&res](T const & v)
            {
                res += v*v;
            }, "squaredNorm(view_nd)");
        return res;
    }

        /** Compute various norms of the given array.
            The norm is determined by parameter \a type:

            <ul>
            <li> type == -1: maximum norm (L-infinity): maximum of absolute values of the array elements
            <li> type == 0: count norm (L0): number of non-zero elements
            <li> type == 1: Manhattan norm (L1): sum of absolute values of the array elements
            <li> type == 2: Euclidean norm (L2): square root of <tt>squaredNorm()</tt> when \a useSquaredNorm is <tt>true</tt>,<br>
                 or direct algorithm that avoids underflow/overflow otherwise.
            </ul>

            Parameter \a useSquaredNorm has no effect when \a type != 2. Defaults: compute L2 norm as square root of
            <tt>squaredNorm()</tt>.
         */
    template <int N, class T>
    NormType<view_nd<N, T> >
    norm(view_nd<N, T> const & array, int type = 2)
    {
        switch(type)
        {
          case -1:
          {
            auto res = NormType<view_nd<N, T> >();
            universalarray_ndFunction(array,
                [&res](T const & v)
                {
                    if(res < abs(v))
                        res = abs(v);
                }, "norm(view_nd)");
            return res;
          }
          case 0:
          {
            auto res = NormType<view_nd<N, T> >();
            auto zero = T();
            universalarray_ndFunction(array,
                [&res, zero](T const & v)
                {
                    if(v != zero)
                        res += 1;
                }, "norm(view_nd)");
            return res;
          }
          case 1:
          {
            auto res = NormType<view_nd<N, T> >();
            universalarray_ndFunction(array,
                [&res](T const & v)
                {
                    res += abs(v);
                }, "norm(view_nd)");
            return res;
          }
          case 2:
          {
            auto res = SquaredNormType<view_nd<N, T> >();
            universalarray_ndFunction(array,
                [&res](T const & v)
                {
                    res += v*v;
                }, "norm(view_nd)");
            return sqrt(res);
          }
          default:
            vigra_precondition(false,
                "norm(view_nd, type): type must be 0, 1, or 2.");
            return NormType<view_nd<N, T> >();
        }
    }

    template <int N, class T, class U = PromoteType<T> >
    inline U
    sum(view_nd<N, T> const & array, U init = U{})
    {
        return array.template sum<U>(init);
    }

    template <int N, class T, class U = PromoteType<T> >
    inline U
    prod(view_nd<N, T> const & array, U init = U{1})
    {
        return array.template prod<U>(init);
    }

    template <int N, class T>
    inline bool
    all(view_nd<N, T> const & array)
    {
        return array.all();
    }

    template <int N, class T>
    inline bool
    all_finite(view_nd<N, T> const & array)
    {
        return array.all_finite();
    }

    template <int N, class T>
    inline bool
    any(view_nd<N, T> const & array)
    {
        return array.any();
    }

    template <int N, class T>
    inline view_nd<N, T>
    transpose(view_nd<N, T> const & array)
    {
        return array.transpose();
    }

    template <int N, class T>
    inline void
    swap(view_nd<N,T> & array1, view_nd<N,T> & array2)
    {
        array1.swap(array2);
    }

    /********************************************************/
    /*                                                      */
    /*                       array_nd                        */
    /*                                                      */
    /********************************************************/

    template <int N, class T, class Alloc /* default already declared */ >
    class array_nd
    : public view_nd<N, T>
    {
      public:
        typedef view_nd<N, T> view_type;

      private:
        typedef std::vector<typename view_type::value_type, Alloc> buffer_type;

        buffer_type allocated_data_;

      public:

        using view_type::internal_dimension;

            /** the allocator type used to allocate the memory
             */
        typedef Alloc allocator_type;

            /** the array's value type
             */
        typedef typename view_type::value_type value_type;

            /** pointer type
             */
        typedef typename view_type::pointer pointer;

            /** const pointer type
             */
        typedef typename view_type::const_pointer const_pointer;

            /** reference type (result of operator[])
             */
        typedef typename view_type::reference reference;

            /** const reference type (result of operator[] const)
             */
        typedef typename view_type::const_reference const_reference;

            /** size type
             */
        typedef typename view_type::size_type size_type;

            /** difference type (used for multi-dimensional offsets and indices)
             */
        typedef typename view_type::shape_type shape_type;

            /** difference and index type for a single dimension
             */
        typedef typename view_type::index_t index_t;

        typedef typename view_type::axistags_type axistags_type;

            /** sequential (random access) iterator type
             */
        typedef typename view_type::iterator iterator;

            /** sequential (random access) const iterator type
             */
        typedef typename view_type::const_iterator const_iterator;

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

            /** Construct with shape given by explicit parameters.

                The number of parameters must match <tt>dimension()</tt>.
             */
        template <class ... V>
        array_nd(index_t l0, V ... l)
        : array_nd(shape_t<sizeof...(V)+1>{l0, l...})
        {
            static_assert(N == runtime_size || N == sizeof...(V)+1,
                "array_nd(int, ...): mismatch between dimension() and number of arguments.");
        }

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
        : view_type(shape, 0, order)
        , allocated_data_(this->size(), init, alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** construct from shape with an initial value
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
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            // /** construct from shape and initialize with a linear sequence in scan order
                // (i.e. first pixel gets value 0, second on gets value 1 and so on).
             // */
        // array_nd (const shape_type &shape, MultiArrayInitializationTag init,
                    // allocator_type const & alloc = allocator_type());

            /** construct from shape and copy values from the given C array
             */
        array_nd(shape_type const & shape,
                const_pointer init,
                tags::memory_order order = c_order,
                allocator_type const & alloc = allocator_type())
        : view_type(shape, 0, order)
        , allocated_data_(init, init + this->size(), alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** construct from shape and axistags and copy values from the given C array
             */
        array_nd(shape_type const & shape,
                axistags_type const & axistags,
                const_pointer init,
                tags::memory_order order = c_order,
                allocator_type const & alloc = allocator_type())
        : view_type(shape, axistags, 0, order)
        , allocated_data_(init, init + this->size(), alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** construct from shape and copy values from the
                given initializer_list
             */
        array_nd(shape_type const & shape,
                std::initializer_list<T> init,
                tags::memory_order order = c_order,
                allocator_type const & alloc = allocator_type())
        : view_type(shape, 0, order)
        , allocated_data_(init, alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            vigra_precondition(this->size() == init.size(),
                "array_nd(): initializer_list has wrong size.");
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** construct from shape and axistags and copy values from the
                given initializer_list
             */
        array_nd(shape_type const & shape,
                axistags_type const & axistags,
                std::initializer_list<T> init,
                tags::memory_order order = c_order,
                allocator_type const & alloc = allocator_type())
        : view_type(shape, axistags, 0, order)
        , allocated_data_(init, alloc)
        {
            vigra_precondition(all_greater_equal(shape, 0),
                "array_nd(): invalid shape.");
            vigra_precondition(this->size() == init.size(),
                "array_nd(): initializer_list has wrong size.");
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** construct 1D-array from initializer_list
             */
        template<index_t M = N,
                 VIGRA_REQUIRE<(M == 1 || M == runtime_size)> >
        array_nd(std::initializer_list<T> init,
                allocator_type const & alloc = allocator_type())
        : view_type(shape_t<1>(init.size()), 0, c_order)
        , allocated_data_(init, alloc)
        {
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** copy constructor
             */
        array_nd(array_nd const & rhs)
        : view_type(rhs)
        , allocated_data_(rhs.allocated_data_)
        {
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** move constructor
             */
        array_nd(array_nd && rhs)
        : view_type()
        , allocated_data_(std::move(rhs.allocated_data_))
        {
            this->swapImpl(rhs);
            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** construct by copying from a view_nd
             */
        template <index_t M, class U>
        array_nd(view_nd<M, U> const & rhs,
                tags::memory_order order = c_order,
                allocator_type const & alloc = allocator_type())
        : view_type(rhs.shape(), rhs.axistags(), 0, order)
        , allocated_data_(alloc)
        {
            allocated_data_.reserve(this->size());

            auto p = detail::permutationToOrder(this->byte_strides(), c_order);
            buffer_type & data = allocated_data_;
            universalPointerNDFunction(rhs.pointer_nd(p), this->shape().transpose(p),
                [&data](U const & u)
                {
                    data.emplace_back(detail::RequiresExplicitCast<T>::cast(u));
                });

            this->data_  = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** Constructor from a temporary array expression.
            */
        template<class ARG>
        array_nd(ArrayMathExpression<ARG> && rhs,
                tags::memory_order order = c_order,
                allocator_type const & alloc = allocator_type())
        : view_type(rhs.shape(), 0, order)
        , allocated_data_(alloc)
        {
            allocated_data_.reserve(this->size());

            if (order != c_order)
            {
                auto p = detail::permutationToOrder(this->byte_strides(), c_order);
                rhs.transpose_inplace(p);
            }

            typedef typename std::remove_reference<ArrayMathExpression<ARG>>::type RHS;
            using U = typename RHS::value_type;
            buffer_type & data = allocated_data_;
            universalPointerNDFunction(rhs, rhs.shape(),
                [&data](U const & u)
                {
                    data.emplace_back(detail::RequiresExplicitCast<T>::cast(u));
                });

            this->data_ = (char*)&allocated_data_[0];
            this->flags_ |= this->consecutive_memory_flag | this->owns_memory_flag;
        }

            /** Constructor from a array expression const reference.
            */
        template<class ARG>
        array_nd(ArrayMathExpression<ARG> const & rhs,
                tags::memory_order order = c_order,
                allocator_type const & alloc = allocator_type())
        : array_nd(ArrayMathExpression<ARG>(rhs), order, alloc)
        {}

            /** Assignment.<br>
                If the size of \a rhs is the same as the left-hand side arrays's
                old size, only the data are copied. Otherwise, new storage is
                allocated, which invalidates all objects (array views, iterators)
                depending on the lhs array.
             */
        array_nd & operator=(array_nd const & rhs)
        {
            if (this != &rhs)
            {
                if(this->shape() == rhs.shape())
                    this->copyImpl(rhs);
                else
                    array_nd(rhs).swap(*this);
            }
            return *this;
        }

            /** Move assignment.<br>
                If the size of \a rhs is the same as the left-hand side arrays's
                old size, only the data are copied. Otherwise, the storage of the
                rhs is moved to the lhs, which invalidates all
                objects (array views, iterators) depending on the lhs array.
             */
        array_nd & operator=(array_nd && rhs)
        {
            if (this != &rhs)
            {
                if(this->shape() == rhs.shape())
                    this->copyImpl(rhs);
                else
                    rhs.swap(*this);
            }
            return *this;
        }

            /** Assignment of a scalar.
             */
        array_nd & operator=(value_type const & u)
        {
            view_type::operator=(u);
            return *this;
        }

    #ifdef DOXYGEN
            /** Assignment from arbitrary ARRAY.

                If the left array has no data or the shapes match, it becomes a copy
                of \a rhs. Otherwise, the function fails with an exception.
             */
        template<class ARRAY_LIKE>
        array_nd & operator=(ARRAY_LIKE const & rhs);

            /** Add-assignment of a differently typed array or an array expression.

                The function fails with an exception when the shapes do not match, unless
                the left array has no data (hasData() is false), in which case the function acts as
                a normal assignment.
             */
        template <class ARRAY_LIKE>
        array_nd & operator+=(ARRAY_LIKE const & rhs);

            /** Subtract-assignment of a differently typed array or an array expression.

                The function fails with an exception when the shapes do not match, unless
                the left array has no data (hasData() is false), in which case the function acts as
                a normal assignment.
             */
        template <class ARRAY_LIKE>
        array_nd & operator-=(ARRAY_LIKE const & rhs);

            /** Multiply-assignment of a differently typed array or an array expression.

                The function fails with an exception when the shapes do not match, unless
                the left array has no data (hasData() is false), in which case the function acts as
                a normal assignment.
             */
        template <class ARRAY_LIKE>
        array_nd & operator*=(ARRAY_LIKE const & rhs);

            /** Divide-assignment of a differently typed array or an array expression.

                The function fails with an exception when the shapes do not match, unless
                the left array has no data (hasData() is false), in which case the function acts as
                a normal assignment.
             */
        template <class ARRAY_LIKE>
        array_nd & operator/=(ARRAY_LIKE const & rhs);

    #else

    #define VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(OP) \
        template <class ARG> \
        std::enable_if_t<array_ndConcept<ARG>::value, \
                    array_nd &> \
        operator OP(ARG const & rhs) \
        { \
            if(this->hasData()) \
                view_type::operator OP(rhs); \
            else \
                array_nd(rhs, c_order, get_allocator()).swap(*this); \
            return *this; \
        } \
        \
        template <class ARG> \
        array_nd & \
        operator OP(ArrayMathExpression<ARG> && rhs) \
        { \
            if(this->hasData()) \
                view_type::operator OP(std::move(rhs)); \
            else \
                array_nd(std::move(rhs), c_order, get_allocator()).swap(*this); \
            return *this; \
        } \
        \
        template <class ARG> \
        array_nd & \
        operator OP(ArrayMathExpression<ARG> const & rhs) \
        { \
            return operator OP(ArrayMathExpression<ARG>(rhs)); \
        }

        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(+=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(-=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(*=)
        VIGRA_array_nd_ARITHMETIC_ASSIGNMENT(/=)

    #undef VIGRA_array_nd_ARITHMETIC_ASSIGNMENT

    #endif  // DOXYGEN

            /** Add-assignment of a scalar.
             */
        array_nd & operator+=(value_type const & u)
        {
            view_type::operator+=(u);
            return *this;
        }

            /** Subtract-assignment of a scalar.
             */
        array_nd & operator-=(value_type const & u)
        {
            view_type::operator-=(u);
            return *this;
        }

            /** Multiply-assignment of a scalar.
             */
        array_nd & operator*=(value_type const & u)
        {
            view_type::operator*=(u);
            return *this;
        }

            /** Divide-assignment of a scalar.
             */
        array_nd & operator/=(value_type const & u)
        {
            view_type::operator/=(u);
            return *this;
        }

        void
        resize(shape_type const & new_shape,
               axistags_type const & new_axistags = axistags_type{},
               tags::memory_order order = c_order)
        {
            vigra_precondition(all_greater_equal(new_shape, 0),
                "array_nd::resize(): invalid shape.");
            if(this->size() == prod(new_shape))
            {
                auto this_r = this->reshape(new_shape, new_axistags, order);
                this->swapImpl(this_r);
                this->flags_ |= this->owns_memory_flag;
            }
            else
            {
                array_nd(new_shape, new_axistags, order).swap(*this);
            }
        }

        template <index_t M>
        void
        resize(shape_type const & new_shape,
               tags::memory_order order)
        {
            resize(new_shape, axistags_type{}, order);
        }

        void swap(array_nd & rhs)
        {
            this->swapImpl(rhs);
            allocated_data_.swap(rhs.allocated_data_);
        }

            /** get the allocator.
             */
        allocator_type get_allocator() const
        {
            return allocated_data_.get_allocator();
        }
    };

    template <int N, class T, class A>
    inline void
    swap(array_nd<N,T,A> & array1, array_nd<N,T,A> & array2)
    {
        array1.swap(array2);
    }

#endif

} // namespace xvigra

#endif // XVIGRA_ARRAY_ND_HPP
