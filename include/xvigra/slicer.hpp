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

#ifndef XVIGRA_SLICER_HPP
#define XVIGRA_SLICER_HPP

#include <algorithm>
#include <xtensor/xstrided_view.hpp>
#include "global.hpp"
#include "concepts.hpp"

namespace xvigra
{
    /**********/
    /* slicer */
    /**********/

    class slicer
    {
      public:
        using shape_type = xt::dynamic_shape<std::size_t>;

        template <class SHAPE>
        slicer(SHAPE const & shape, tags::memory_order order=tags::c_order)
        : shape_(shape.begin(), shape.end())
        , iter_inc_(order == tags::c_order ? -1 : 1)
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

} // namespace xvigra

#endif // XVIGRA_SLICER_HPP
