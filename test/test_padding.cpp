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

#include "unittest.hpp"
#include <xvigra/padding.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace xvigra
{
    TEST(padding, padding_modes)
    {
        xt::xtensor<int, 1> in{1, 2, 3, 4, 5};
        {
            xt::xtensor<int, 1> out(in.shape(), 0);

            copy_with_padding(in, out, no_padding, 0);
            EXPECT_EQ(in, out);
            EXPECT_THROW(copy_with_padding(in, out, no_padding, 1), std::runtime_error);
        }
        {
            xt::xtensor<int, 1> ref{0, 0, 1, 2, 3, 4, 5, 0, 0},
                                out(ref.shape(), 0);

            copy_with_padding(in, out, zero_padding, 2);
            EXPECT_EQ(out, ref);
        }
        {
            xt::xtensor<int, 1> ref{0, 0, 1, 2, 3, 4, 5, 0, 0, 0},
                                out(ref.shape(), 0);

            copy_with_padding(in, out, zero_padding, 2, zero_padding, 3);
            EXPECT_EQ(out, ref);
        }
        {
            xt::xtensor<int, 1> ref{1, 1, 1, 2, 3, 4, 5, 5, 5, 5},
                                out(ref.shape(), 0);

            copy_with_padding(in, out, repeat_padding, 2, repeat_padding, 3);
            EXPECT_EQ(out, ref);
        }
        {
            xt::xtensor<int, 1> ref{4, 5, 1, 2, 3, 4, 5, 1, 2, 3},
                                out(ref.shape(), 0);

            copy_with_padding(in, out, periodic_padding, 2, periodic_padding, 3);
            EXPECT_EQ(out, ref);
        }
        {
            xt::xtensor<int, 1> ref{3, 2, 1, 2, 3, 4, 5, 4, 3, 2},
                                out(ref.shape(), 0);

            copy_with_padding(in, out, reflect_padding, 2, reflect_padding, 3);
            EXPECT_EQ(out, ref);
        }
        {
            xt::xtensor<int, 1> ref{2, 1, 1, 2, 3, 4, 5, 5, 4, 3},
                                out(ref.shape(), 0);

            copy_with_padding(in, out, reflect0_padding, 2, reflect0_padding, 3);
            EXPECT_EQ(out, ref);
        }
        {
            xt::xtensor<int, 1> ref{4, 3, 2, 1, 2, 3, 4, 5, 1, 2},
                                out(ref.shape(), 0);

            copy_with_padding(in, out, reflect_padding, 3, periodic_padding, 2);
            EXPECT_EQ(out, ref);
        }
    }

} // namespace xvigra
