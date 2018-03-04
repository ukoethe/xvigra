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

#include <xtensor/xstrided_view.hpp>
#include "global.hpp"

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
        
        slicer(shape_type const & shape, index_t skip_axis)
        : shape_(shape)
        , slice_(shape)
        {
            shape_[skip_axis] = 1;
            for(index_t k=0; k < shape.size(); ++k)
            {
                if(k == skip_axis)
                {
                    slice_.push_back(xt::all());
                }
                else
                {
                    slice_.push_back(0);
                }
            }
        }

        xt::slice_vector const & operator*() const
        {
            return slice_;
        }

        void operator++()
        {
            index_t k = shape_.size() - 1;
            ++slice_[k][0];
            while(slice_[k][0] == shape_[k] && k > 0)
            {
                slice_[k][0] = 0;
                --k;
                ++slice_[k][0];
            }
        }

        bool has_more() const
        {
            return slice_[0][0] != shape_[0];
        }

      private:
        shape_type shape_;
        xt::slice_vector slice_;
    };

} // namespace xvigra

#endif // XVIGRA_SLICER_HPP