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

#ifndef XVIGRA_PADDING_HPP
#define XVIGRA_PADDING_HPP

#include <algorithm>
#include "global.hpp"
#include "concepts.hpp"
#include "error.hpp"

namespace xvigra
{
    enum padding_mode
    {
        no_padding,
        zero_padding,
        periodic_padding,
        repeat_padding,
        reflect_padding,
        reflect0_padding
    };

    // 'in' and 'out' must be 1-dimensional arrays whose sizes fulfill:
    // 'left_padding_size + in.size() + right_padding_size == out.size()'
    // For padding modes periodic_padding, reflect_padding, reflect0_padding:
    // 'left_padding_size < in.size() && right_padding_size < in.size()'
    // For padding mode no_padding:
    // 'left_padding_size == 0 && right_padding_size == 0'
    template <class InArray, class OutArray,
              VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
    void copy_with_padding(InArray const & in, OutArray && out,
                           padding_mode left_padding_mode, index_t left_padding_size,
                           padding_mode right_padding_mode, index_t right_padding_size)
    {
        using dest_type =  typename std::decay_t<OutArray>::value_type;

        index_t size = in.size();
        vigra_precondition(left_padding_size + size + right_padding_size == out.size(),
            "copy_with_padding(): output size must equal input size plus padding sizes.");

        std::copy(in.begin(), in.end(), out.begin()+left_padding_size);

        switch(left_padding_mode)
        {
            case zero_padding:
            {
                std::fill(out.begin(), out.begin()+left_padding_size, dest_type());
                break;
            }
            case repeat_padding:
            {
                vigra_precondition(size > 0,
                    "copy_with_padding(): input size must be non-zero.");

                std::fill(out.begin(), out.begin()+left_padding_size, dest_type(in(0)));
                break;
            }
            case periodic_padding:
            {
                vigra_precondition(left_padding_size < size,
                    "copy_with_padding(): left_padding_size must be less than input size.");

                index_t offset = size-left_padding_size;
                for(index_t k=0; k < left_padding_size; ++k)
                {
                    out(k) = conditional_cast<std::is_arithmetic<dest_type>::value, dest_type>(in(offset+k));
                }
                break;
            }
            case reflect_padding:
            case reflect0_padding:
            {
                vigra_precondition(left_padding_size < size,
                    "copy_with_padding(): left_padding_size must be less than input size.");

                index_t offset = left_padding_size;
                if(left_padding_mode == reflect0_padding)
                {
                    offset -= 1;
                }
                for(index_t k=0; k < left_padding_size; ++k)
                {
                    out(k) = conditional_cast<std::is_arithmetic<dest_type>::value, dest_type>(in(offset-k));
                }
                break;
            }
            default:
            {
                vigra_precondition(left_padding_mode == no_padding,
                    "copy_with_padding(): illegal left_padding_mode.");
            }
        }

        switch(right_padding_mode)
        {
            case zero_padding:
            {
                std::fill(out.begin()+size+right_padding_size, out.end(), dest_type());
                break;
            }
            case repeat_padding:
            {
                vigra_precondition(size > 0,
                    "copy_with_padding(): input size must be non-zero.");

                std::fill(out.begin()+size+left_padding_size, out.end(), dest_type(in(size-1)));
                break;
            }
            case periodic_padding:
            {
                vigra_precondition(right_padding_size < size,
                    "copy_with_padding(): right_padding_size must be less than input size.");

                index_t offset = size + left_padding_size;
                for(index_t k=0; k < right_padding_size; ++k)
                {
                    out(offset+k) = conditional_cast<std::is_arithmetic<dest_type>::value, dest_type>(in(k));
                }
                break;
            }
            case reflect_padding:
            case reflect0_padding:
            {
                vigra_precondition(right_padding_size < size,
                    "copy_with_padding(): right_padding_size must be less than input size.");

                index_t in_offset  = size - 1,
                        out_offset = size + left_padding_size;
                if(right_padding_mode == reflect_padding)
                {
                    in_offset -= 1;
                }
                for(index_t k=0; k < right_padding_size; ++k)
                {
                    out(out_offset+k) = conditional_cast<std::is_arithmetic<dest_type>::value, dest_type>(in(in_offset-k));
                }
                break;
            }
            default:
            {
                vigra_precondition(right_padding_mode == no_padding,
                    "copy_with_padding(): illegal right_padding_mode.");
            }
        }
    }

    template <class InArray, class OutArray,
              VIGRA_REQUIRE<tensor_concept<InArray>::value && tensor_concept<OutArray>::value>>
    void copy_with_padding(InArray const & in, OutArray && out,
                           padding_mode pad_mode, index_t pad_size)
    {
        copy_with_padding(in, std::forward<OutArray>(out), pad_mode, pad_size, pad_mode, pad_size);
    }
}

#endif // XVIGRA_PADDING_HPP
