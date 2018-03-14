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
        // FIXME: support C- and F-order, higher dimensional slices
      public:
        using shape_type = xt::dynamic_shape<std::size_t>;

        template <class SHAPE>
        slicer(SHAPE const & shape, index_t keep_axis)
        : shape_(shape.begin(), shape.end())
        , final_index_(shape_.size())
        {
            this->keep_axis(keep_axis);
        }

        template <class SHAPE>
        slicer(SHAPE const & shape)
        : shape_(shape.begin(), shape.end())
        , final_index_(shape_.size())
        {}

        void keep_axis(index_t axis)
        {
            keep_axes(std::array<index_t, 1>{axis});
        }

        template <class C,
                  VIGRA_REQUIRE<container_concept<C>::value>>
        void keep_axes(C axes)
        {
            std::sort(axes.begin(), axes.end());
            for(index_t k=0, n=0; k < shape_.size(); ++k)
            {
                if(n < axes.size() && k == axes[n])
                {
                    shape_[k] = 1;
                    slice_.push_back(xt::all());
                    ++n;
                }
                else
                {
                    if(final_index_ == shape_.size())
                    {
                        final_index_ = k;
                    }
                    slice_.push_back(0);
                }
            }
        }

        void bind_axis(index_t axis)
        {
            bind_axes(std::array<index_t, 1>{axis});
        }

        template <class C,
                  VIGRA_REQUIRE<container_concept<C>::value>>
        void bind_axes(C axes)
        {
            std::sort(axes.begin(), axes.end());
            for(index_t k=0, n=0; k < shape_.size(); ++k)
            {
                if(n < axes.size() && k != axes[n])
                {
                    shape_[k] = 1;
                    slice_.push_back(xt::all());
                }
                else
                {
                    if(final_index_ == shape_.size())
                    {
                        final_index_ = k;
                    }
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
            for(int k = shape_.size() - 1; k >= final_index_; --k)
            {
                auto p = xtl::get_if<int>(&slice_[k]);
                if(p == nullptr)
                {
                    continue;
                }
                ++(*p);
                if(*p < shape_[k] || k == final_index_)
                {
                    break;
                }
                *p = 0;
            }
        }

        bool has_more() const
        {
            return (final_index_ != shape_.size()) &&
                   (*xtl::get_if<int>(&slice_[final_index_]) != shape_[final_index_]);
        }

      private:
        shape_type shape_;
        xt::slice_vector slice_;
        int final_index_;
    };

} // namespace xvigra

#endif // XVIGRA_SLICER_HPP