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

#ifndef XVIGRA_SLICE_HPP
#define XVIGRA_SLICE_HPP

#include <algorithm>
#include <limits>
#include <vector>
#include <xtensor/xstrided_view.hpp>
#include "global.hpp"
#include "concepts.hpp"
#include "tiny_vector.hpp"

namespace xvigra
{
    struct slice;

    namespace slicing
    {
        using underscore_t = xt::placeholders::xtuph;
        using xt::placeholders::_;
        using xt::all;
        using xt::newaxis;
        using xt::ellipsis;

        using newaxis_tag = xt::xnewaxis_tag;
        using all_tag = xt::xall_tag;
        using ellipsis_tag = xt::xellipsis_tag;

        template <class B = underscore_t, class E = underscore_t, class S = underscore_t>
        inline auto
        range(B b = underscore_t(), E e = underscore_t(), S s = underscore_t())
        {
            return slice(b, e, s);
        }
    }

    /*********/
    /* slice */
    /*********/

    struct slice
    {
        using underscore_t = slicing::underscore_t;

      private:

        struct special_step_tag {};

        static index_t parse_step(underscore_t)
        {
            return 1;
        }

        static index_t parse_step(index_t s)
        {
            return s;
        }

        template <class T>
        static index_t parse_start(index_t b, T)
        {
            return b;
        }

        static index_t parse_start(underscore_t, index_t s)
        {
            return s > 0 ? 0 : -1;
        }

        static index_t parse_start(underscore_t, underscore_t)
        {
            return 0;
        }

        template <class T>
        static index_t parse_stop(index_t e, T)
        {
            return e;
        }

        static index_t parse_stop(underscore_t, index_t s)
        {
            return s > 0 ? std::numeric_limits<index_t>::max() : std::numeric_limits<index_t>::lowest();
        }

        static index_t parse_stop(underscore_t, underscore_t)
        {
            return std::numeric_limits<index_t>::max();
        }

        slice(index_t b, index_t e, special_step_tag)
        : start(b)
        , stop(e)
        , step(0)
        {}

      public:

        index_t start, stop, step;

        template <class B = underscore_t, class E = underscore_t, class S = underscore_t,
                  VIGRA_REQUIRE<!std::is_same<S, special_step_tag>::value>>
        slice(B b = underscore_t(), E e = underscore_t(), S s = underscore_t())
        : start(parse_start(b, s))
        , stop(parse_stop(e, s))
        , step(parse_step(s))
        {
            vigra_precondition(step != 0,
                "slice(): step must be non-zero.");
        }

        slice(slicing::all_tag)
        : slice()
        {}

        slice(slicing::newaxis_tag)
        : slice(0, 1, special_step_tag())
        {}

        slice(slicing::ellipsis_tag)
        : slice(1, 0, special_step_tag())
        {}

        static slice bind(index_t b)
        {
            return slice(b, b, special_step_tag());
        }

        bool operator==(slice const & o) const
        {
            return start == o.start && stop == o.stop && step == o.step;
        }

        bool operator!=(slice const & o) const
        {
            return start != o.start || stop != o.stop || step != o.step;
        }
    };

    /****************/
    /* slice_vector */
    /****************/

    class slice_vector
    : public std::vector<slice>
    {
      public:
        using base_type = std::vector<slice>;

        using base_type::base_type;
        using base_type::operator=;

        slice_vector & emplace_back(slice && s)
        {
            base_type::emplace_back(std::move(s));
            return *this;
        }

        slice_vector & push_back(slice const & s)
        {
            base_type::push_back(s);
            return *this;
        }

        slice_vector & push_back(index_t i)
        {
            base_type::emplace_back(slice::bind(i));
            return *this;
        }

        slice_vector & push_back(slicing::all_tag)
        {
            base_type::emplace_back(slice());
            return *this;
        }

        slice_vector & push_back(slicing::newaxis_tag)
        {
            base_type::emplace_back(slice(slicing::newaxis()));
            return *this;
        }

        slice_vector & push_back(slicing::ellipsis_tag)
        {
            base_type::emplace_back(slice(slicing::ellipsis()));
            return *this;
        }
    };

    /**********/
    /* slicer */
    /**********/

    class slicer
    {
      public:
        using shape_type = shape_t<>;

        slicer(shape_type const & shape, tags::memory_order order=c_order)
        : shape_(shape)
        , order_(order)
        {}

        template <class ... T>
        void set_free_axes(index_t a, T ... r)
        {
            set_free_axes(shape_t<1+sizeof...(T)>{a, r...});
        }

        template <class C,
                  VIGRA_REQUIRE<container_concept<C>::value>>
        void set_free_axes(C axes)
        {
            slice_ = slice_vector(shape_.size(), slice::bind(0));
            iter_axes_ = shape_type::range(shape_.size());

            std::sort(axes.begin(), axes.end());
            for(index_t k=axes.size()-1; k>=0; --k)
            {
                iter_axes_ = iter_axes_.erase(axes[k]);
                slice_[axes[k]] = slice(slicing::all());
            }
            if(order_ == c_order)
            {
                iter_axes_ = reversed(iter_axes_);
            }
        }

        template <class ... T>
        void set_iterate_axes(index_t a, T ... r)
        {
            set_iterate_axes(shape_t<1+sizeof...(T)>{a, r...});
        }

        template <class C,
                  VIGRA_REQUIRE<container_concept<C>::value>>
        void set_iterate_axes(C axes)
        {
            slice_ = slice_vector(shape_.size(), slice(slicing::all()));
            std::sort(axes.begin(), axes.end());
            for(index_t k=0; k<axes.size(); ++k)
            {
                slice_[axes[k]] = slice::bind(0);
            }
            iter_axes_ = shape_type(axes.begin(), axes.end());
            if(order_ == c_order)
            {
                iter_axes_ = reversed(iter_axes_);
            }
        }

        slice_vector const & operator*() const
        {
            return slice_;
        }

        void operator++()
        {
            index_t k = 0;
            for(; k < iter_axes_.size(); ++k)
            {
                index_t i = iter_axes_[k];
                auto & s = slice_[i];
                if(s.start >= shape_[i]-1)
                {
                    s.start = 0;
                    s.stop  = 0;
                }
                else
                {
                    ++s.start;
                    ++s.stop;
                    break;
                }
            }
            if(k == iter_axes_.size())
            {
                iter_axes_.resize(0);
            }
        }

        void operator++(int)
        {
            ++(*this);
        }

        bool has_more() const
        {
            return iter_axes_.size() > 0;
        }

      private:
        shape_type shape_, iter_axes_;
        slice_vector slice_;
        tags::memory_order order_;
    };

    /***********/
    /* xslicer */
    /***********/

    class xslicer
    {
      public:
        using shape_type = xt::dynamic_shape<std::size_t>;

        template <class SHAPE>
        xslicer(SHAPE const & shape, tags::memory_order order=c_order)
        : shape_(shape.begin(), shape.end())
        , iter_inc_(order == c_order ? -1 : 1)
        {}

        template <class ... T>
        void set_free_axes(index_t a, T ... r)
        {
            set_free_axes(std::array<index_t, 1+sizeof...(T)>{a, r...});
        }

        template <class C,
                  VIGRA_REQUIRE<container_concept<C>::value>>
        void set_free_axes(C axes)
        {
            if(iter_inc_ < 0)
            {
                start_axis_ = shape_.size() - 1;
                end_axis_ = -1;
            }
            else
            {
                start_axis_ = 0;
                end_axis_ = shape_.size();
            }
            slice_.clear();

            std::sort(axes.begin(), axes.end());
            for(std::size_t k=0, n=0; k < shape_.size(); ++k)
            {
                if(n < axes.size() && k == (std::size_t)axes[n])
                {
                    slice_.push_back(xt::all());
                    ++n;
                }
                else
                {
                    slice_.push_back(0);
                }
            }
        }

        template <class ... T>
        void set_iterate_axes(index_t a, T ... r)
        {
            set_iterate_axes(std::array<index_t, 1+sizeof...(T)>{a, r...});
        }

        template <class C,
                  VIGRA_REQUIRE<container_concept<C>::value>>
        void set_iterate_axes(C axes)
        {
            if(iter_inc_ < 0)
            {
                start_axis_ = shape_.size() - 1;
                end_axis_ = -1;
            }
            else
            {
                start_axis_ = 0;
                end_axis_ = shape_.size();
            }
            slice_.clear();

            std::sort(axes.begin(), axes.end());
            for(std::size_t k=0, n=0; k < shape_.size(); ++k)
            {
                if(n >= axes.size() || k != (std::size_t)axes[n])
                {
                    slice_.push_back(xt::all());
                }
                else
                {
                    slice_.push_back(0);
                    ++n;
                }
            }
        }

        xt::slice_vector const & operator*() const
        {
            return slice_;
        }

        void operator++()
        {
            int k = start_axis_;
            for(; k != end_axis_; k += iter_inc_)
            {
                auto p = xtl::get_if<int>(&slice_[k]);
                if(p == nullptr)
                {
                    continue;
                }
                ++(*p);
                if(*p < (int)shape_[k])
                {
                    break;
                }
                *p = 0;
            }
            if(k == end_axis_)
            {
                start_axis_ = end_axis_;
            }
        }

        void operator++(int)
        {
            ++(*this);
        }

        bool has_more() const
        {
            return start_axis_ != end_axis_;
        }

      private:
        shape_type shape_;
        xt::slice_vector slice_;
        int iter_inc_, start_axis_, end_axis_;
    };

    namespace detail
    {
        /**************************/
        /* slice_dimension_traits */
        /**************************/

        template <class S, bool = std::is_integral<S>::value>
        struct slice_dimension_traits_integral
        {
            static constexpr index_t value = 0;
        };

        template <class S>
        struct slice_dimension_traits_integral<S, false>
        {
            static constexpr index_t value = 1;
        };

        template <index_t M, class ...A>
        struct slice_dimension_traits_base;

        template <index_t M, class S, class ...A>
        struct slice_dimension_traits_base<M, S, A...>
        {
            static constexpr index_t target_dimension = slice_dimension_traits_integral<S>::value +
                                                        slice_dimension_traits_base<M-1, A...>::target_dimension;
            static constexpr index_t slice_dimension  = 1 + slice_dimension_traits_base<M, A...>::slice_dimension;
        };

        template <index_t M, class ...A>
        struct slice_dimension_traits_base<M, xt::xellipsis_tag, A...>
        {
            static constexpr index_t target_dimension = slice_dimension_traits_base<M, A...>::target_dimension;
            static constexpr index_t slice_dimension  = slice_dimension_traits_base<M, A...>::slice_dimension;
        };

        template <index_t M, class ...A>
        struct slice_dimension_traits_base<M, xt::xnewaxis_tag, A...>
        {
            static constexpr index_t target_dimension = 1 + slice_dimension_traits_base<M, A...>::target_dimension;
            static constexpr index_t slice_dimension  = slice_dimension_traits_base<M, A...>::slice_dimension;
        };

        template <index_t M>
        struct slice_dimension_traits_base<M>
        {
            static constexpr index_t target_dimension = M;
            static constexpr index_t slice_dimension  = 0;
        };

            // attempt to determine the dimension of the resulting view at compile time
        template <index_t N, class S, class ...A>
        struct slice_dimension_traits
        : public slice_dimension_traits_base<N, S, A...>
        {
            static_assert(slice_dimension_traits_base<N, S, A...>::target_dimension >= 0,
                "slice has too many indices.");
        };

        template <class S, class ...A>
        struct slice_dimension_traits<runtime_size, S, A...>
        : public slice_dimension_traits_base<runtime_size, S, A...>
        {
            static constexpr index_t target_dimension = runtime_size;
        };

        template <>
        struct slice_dimension_traits<runtime_size, slice_vector>
        {
            static constexpr index_t target_dimension = runtime_size;
            static constexpr index_t slice_dimension  = runtime_size;
        };

        /*******************/
        /* slice_dimension */
        /*******************/

        template <class S, class ... A>
        index_t slice_dimension(bool & has_ellipsis, S s, A ... a);

        template <class ... A>
        index_t slice_dimension(bool & has_ellipsis, xt::xellipsis_tag, A ... a);

        template <class ... A>
        index_t slice_dimension(bool & has_ellipsis, xt::xnewaxis_tag, A ... a);

        inline index_t slice_dimension(bool & /* has_ellipsis */)
        {
            return 0;
        }

        template <class S, class ... A>
        index_t slice_dimension(bool & has_ellipsis, S s, A ... a)
        {
            return 1 + slice_dimension(has_ellipsis, a...);
        }

        template <class ... A>
        index_t slice_dimension(bool & has_ellipsis, xt::xellipsis_tag, A ... a)
        {
            vigra_precondition(!has_ellipsis,
                "parse_slices(): an index can only have a single ellipsis");
            has_ellipsis = true;
            return slice_dimension(has_ellipsis, a...);
        }

        template <class ... A>
        index_t slice_dimension(bool & has_ellipsis, xt::xnewaxis_tag, A ... a)
        {
            return slice_dimension(has_ellipsis, a...);
        }

        inline index_t
        slice_dimension(bool & has_ellipsis, slice_vector const & s)
        {
            index_t dim = 0;
            for(std::size_t k=0; k<s.size(); ++k)
            {
                index_t step = s[k].step;
                if(step == 0)
                {
                    index_t start = s[k].start,
                            stop  = s[k].stop;
                    if(start != stop)
                    {
                        if(stop == 0) // ellipsis
                        {
                            vigra_precondition(!has_ellipsis,
                                "parse_slices(): an index can only have a single ellipsis");
                            has_ellipsis = true;
                        }
                        // else { pass; }  // newaxis
                    }
                    else // bind axis at index
                    {
                        ++dim;
                    }
                }
                else // actual slice or all
                {
                    ++dim;
                }
            }
            return dim;
        }

        /*********************/
        /* parse_slices_impl */
        /*********************/

        template <index_t N>
        index_t parse_slices_impl(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                                  shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                                  index_t ellipsis_size,
                                  xt::xnewaxis_tag)
        {
            shape = shape.push_back(1);
            strides = strides.push_back(0);
            return axis;
        }

        template <index_t N>
        index_t parse_slices_impl(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                                  shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                                  index_t ellipsis_size,
                                  xt::xall_tag)
        {
            point[axis] = 0;
            shape = shape.push_back(old_shape[axis]);
            strides = strides.push_back(old_strides[axis]);
            return axis+1;
        }

        template <index_t N>
        index_t parse_slices_impl(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                                  shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                                  index_t ellipsis_size,
                                  xt::xellipsis_tag)
        {
            for(; ellipsis_size > 0; --ellipsis_size, ++axis)
            {
                point[axis] = 0;
                shape = shape.push_back(old_shape[axis]);
                strides = strides.push_back(old_strides[axis]);
            }
            return axis;
        }

        template <index_t N>
        index_t parse_slices_impl(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                                  shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                                  index_t ellipsis_size,
                                  index_t i)
        {
            point[axis] = (i >= 0) ? i : (i + old_shape[axis]);
            vigra_precondition(point[axis] >= 0 && point[axis] < old_shape[axis],
                "index " + std::to_string(i) + " out of bounds for axis " + std::to_string(axis) + ".");
            return axis+1;
        }

        template <index_t N>
        index_t parse_slices_impl(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                                  shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                                  index_t ellipsis_size,
                                  slice const & s)
        {
            index_t step  = s.step;
            if(step == 0)
            {
                // slice has special meaning
                index_t start = s.start;
                index_t stop  = s.stop;

                if(start == stop)
                {
                    return parse_slices_impl(axis, point, shape, strides,
                                             old_shape, old_strides, ellipsis_size, start);
                }
                else if(start == 0)
                {
                    return parse_slices_impl(axis, point, shape, strides,
                                             old_shape, old_strides, ellipsis_size, slicing::newaxis());
                }
                else
                {
                    return parse_slices_impl(axis, point, shape, strides,
                                             old_shape, old_strides, ellipsis_size, slicing::ellipsis());
                }
            }
            else
            {
                index_t start = (s.start >= 0) ? s.start : s.start + old_shape[axis];
                index_t stop  = (s.stop  >= 0) ? s.stop  : s.stop  + old_shape[axis];
                index_t size;
                if(step > 0)
                {
                    start = max(0, min(old_shape[axis], start));
                    stop  = max(0, min(old_shape[axis], stop));
                    size  = (stop - start + step - 1) / step;
                }
                else
                {
                    start = max(-1, min(old_shape[axis]-1, start));
                    stop  = max(-1, min(old_shape[axis]-1, stop));
                    size  = (stop - start + step + 1) / step;
                }

                point[axis] = start;
                shape = shape.push_back((size <= 0) ? 0 : size);
                strides = strides.push_back((size <= 0) ? old_strides[axis] : old_strides[axis]*step);
                return axis+1;
            }
        }

        /****************/
        /* parse_slices */
        /****************/

        template <index_t N>
        void parse_slices(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                          shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                          index_t ellipsis_size)
        {
            for(; axis<(index_t)point.size(); ++axis)
            {
                point[axis] = 0;
                shape = shape.push_back(old_shape[axis]);
                strides = strides.push_back(old_strides[axis]);
            }
        }

        template <index_t N, class S, class ... A>
        void parse_slices(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                          shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                          index_t ellipsis_size,
                          S s, A ... a)
        {
            axis = parse_slices_impl(axis, point, shape, strides, old_shape, old_strides, ellipsis_size, s);
            parse_slices(axis, point, shape, strides, old_shape, old_strides, ellipsis_size, a...);
        }

        template <index_t N, class ... A>
        void parse_slices(shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                          shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                          A ... a)
        {
            bool has_ellipsis = false;
            index_t c = slice_dimension(has_ellipsis, a...),
                    ellipsis_size = point.size() - c;
            vigra_precondition(c <= (index_t)point.size(),
                "slice has too many indices.");

            parse_slices(0, point, shape, strides, old_shape, old_strides, ellipsis_size, a...);
        }

        template <index_t N>
        void parse_slices(shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                          shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                          slice_vector const & s)
        {
            bool has_ellipsis = false;
            index_t c = slice_dimension(has_ellipsis, s),
                    ellipsis_size = point.size() - c,
                    axis = 0;
            vigra_precondition(c <= (index_t)point.size(),
                "slice has too many indices.");

            for(std::size_t k=0; k<s.size(); ++k)
            {
                axis = parse_slices_impl(axis, point, shape, strides, old_shape, old_strides, ellipsis_size, s[k]);
            }
            while(axis < (index_t)point.size())
            {
                axis = parse_slices_impl(axis, point, shape, strides, old_shape, old_strides, ellipsis_size,
                                         slicing::all());
            }
        }

        // template <index_t N, class ... SLICES>
        // struct slice_parser
        // {
        //     // N is source_dimension or runtime_size
        //     using traits = slice_dimension_traits<N, SLICES...>;
        //     static constexpr index_t M = traits::target_dimension;
        //     static constexpr index_t S = traits::slice_dimension;

        //     index_t source_axis, target_axis;
        //     shape_t<N> point;
        //     shape_t<target_dimension> new_shape, new_strides;

        //     slice_parser(shape_t<N> const & old_shape, shape_t<N> const & old_strides,
        //                  SLICES ... s)
        //     : source_axis(0)
        //     , target_axis(0)
        //     , point(old_shape.size(), 0)
        //     , new_shape(target_dimension(s), 0)
        //     , new_strides(new_shape.size(), 0)
        //     {
        //         parse(old_shape, old_strides, s...);
        //     }

        //     template <class K=M,
        //               VIGRA_REQUIRE<(K != runtime_size)>>
        //     static constexpr index_t target_dimension(SLICES ...)
        //     {
        //         return M;
        //     }

        //     template <class K=M,
        //               VIGRA_REQUIRE<(K == runtime_size)>>
        //     static constexpr index_t target_dimension(SLICES ...s)
        //     {
        //         return M;
        //     }
        // };

    } // namespace detail

} // namespace xvigra

#endif // XVIGRA_SLICE_HPP
