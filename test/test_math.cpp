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
#include <xvigra/math.hpp>

namespace xvigra 
{
    TEST(math, functions)
    {
        EXPECT_EQ(sq(2), 4);
        EXPECT_EQ(sq(1.5), 2.25);
        EXPECT_EQ(sq(-1.5), 2.25);
    }

    TEST(math, is_close)
    {
        double eps = 1e-5;

        // test default tolerance
        EXPECT_TRUE(is_close(numeric_constants<>::PI, 3.141592653589793238463));
        EXPECT_FALSE(is_close(numeric_constants<>::PI, 3.141));
        // test custom tolerance
        EXPECT_TRUE(is_close(numeric_constants<>::PI, 3.141, 1e-3));
        EXPECT_FALSE(is_close(numeric_constants<>::PI, 3.141, 1e-4));
        EXPECT_TRUE(is_close(numeric_constants<>::PI, 3.141, 1e-4, 1e-3));
        // test NaN
        EXPECT_FALSE(is_close(std::log(-1.0), 3.141));
        EXPECT_FALSE(is_close(std::log(-1.0), std::log(-2.0)));
        EXPECT_TRUE(is_close(std::log(-1.0), std::log(-2.0), eps, eps, true));
    }


    TEST(math, norm)
    {
        std::vector<int> v {3, 4, -5};
        std::array<int, 3> a {3, 4, -5};
        EXPECT_EQ(norm_sq(v), 50);
        EXPECT_EQ(norm_linf(v), 5);
        EXPECT_EQ(norm_sq(a), 50);
        EXPECT_EQ(norm_linf(a), 5);
    }
} // namespace xvigra
