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
#include <xtensor/xstrided_view.hpp>
#include "global.hpp"
#include "concepts.hpp"
#include "tiny_vector.hpp"

namespace xvigra
{
    /*********/
    /* slice */
    /*********/

    struct slice
    {
        index_t start, stop, step;

        using underscore = xt::placeholders::xtuph;

        static index_t parse_step(underscore)
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

        static index_t parse_start(underscore, index_t s)
        {
            return s > 0 ? 0 : -1;
        }

        static index_t parse_start(underscore, underscore)
        {
            return 0;
        }

        template <class T>
        static index_t parse_stop(index_t e, T)
        {
            return e;
        }

        static index_t parse_stop(underscore, index_t s)
        {
            return s > 0 ? -1 : 0;
        }

        static index_t parse_stop(underscore, underscore)
        {
            return -1;
        }

        template <class B, class E, class S = underscore>
        slice(B b, E e, S s = underscore())
        : start(parse_start(b, s))
        , stop(parse_stop(e, s))
        , step(parse_step(s))
        {
            vigra_precondition(step != 0,
                "slice(): step must be non-zero.");
        }
    };

    /**********/
    /* slicer */
    /**********/

    class slicer
    {
      public:
        using shape_type = xt::dynamic_shape<std::size_t>;

        template <class SHAPE>
        slicer(SHAPE const & shape, tags::memory_order order=c_order)
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
            for(index_t k=0, n=0; k < shape_.size(); ++k)
            {
                if(n < axes.size() && k == axes[n])
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
            for(index_t k=0, n=0; k < shape_.size(); ++k)
            {
                if(n >= axes.size() || k != axes[n])
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
                if(*p < shape_[k])
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
        struct slice_dimension_traits_impl
        {
            static constexpr index_t value = 0;
        };

        template <class S>
        struct slice_dimension_traits_impl<S, false>
        {
            static constexpr index_t value = 1;
        };

        template <index_t N, index_t M, class ...A>
        struct slice_dimension_traits;

        template <index_t N, index_t M, class S, class ...A>
        struct slice_dimension_traits<N, M, S, A...>
        {
            static constexpr index_t value = slice_dimension_traits_impl<S>::value + slice_dimension_traits<N, M-1, A...>::value;
        };

        template <index_t N, index_t M, class ...A>
        struct slice_dimension_traits<N, M, xt::xellipsis_tag, A...>
        {
            static constexpr index_t value = slice_dimension_traits<N, M, A...>::value;
        };

        template <index_t N, index_t M, class ...A>
        struct slice_dimension_traits<N, M, xt::xnewaxis_tag, A...>
        {
            static constexpr index_t value = 1 + slice_dimension_traits<N, M, A...>::value;
        };

        template <index_t N, index_t M>
        struct slice_dimension_traits<N, M>
        {
            static_assert(N == runtime_size || M >= 0, "slice has too many indices.");
            static constexpr index_t value = M;
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

        index_t slice_dimension(bool & /* has_ellipsis */)
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
            point[axis] = (i < 0) ? (i + old_shape[axis]) : i;
            vigra_precondition(i >= 0 && i < old_shape[axis],
                "index " + std::to_string(i) + " out of bounds for axis " + std::to_string(axis) + ".");
            return axis+1;
        }

        template <index_t N>
        index_t parse_slices_impl(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                                  shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                                  index_t ellipsis_size,
                                  slice const & s)
        {
            index_t start = max(0, min(old_shape[axis]-1,
                                      (s.start < 0) ? s.start + old_shape[axis] : s.start));
            index_t stop  = max(0, min(old_shape[axis]-1,
                                      (s.stop < 0) ? s.stop + old_shape[axis] : s.stop));
            index_t step  = s.step;
            index_t size = (step > 0)
                              ? (stop - start + 1) / step
                              : (stop - start - 1) / step;
            point[axis] = start;
            if(size <= 0)
            {
                shape = shape.push_back(size < 0 ? 0 : 1);
                strides = strides.push_back(0);
            }
            else
            {
                shape = shape.push_back(size);
                strides = strides.push_back(old_strides[axis]*step);
            }
            return axis+1;
        }

        /****************/
        /* parse_slices */
        /****************/

        template <index_t N>
        void parse_slices(index_t axis, shape_t<N> & point, shape_t<> & shape, shape_t<> & strides,
                          shape_t<N> const & old_shape, shape_t<N> const & old_strides,
                          index_t ellipsis_size)
        {
            for(; axis<point.size(); ++axis)
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
            index_t c = slice_dimension(has_ellipsis, a...);
            vigra_precondition(c <= point.size(),
                "slice has too many indices.");
            index_t ellipsis_size = has_ellipsis ? point.size() - c : 0;
            parse_slices(0, point, shape, strides, old_shape, old_strides, ellipsis_size, a...);
        }
    } // namespace detail

} // namespace xvigra

#endif // XVIGRA_SLICE_HPP
