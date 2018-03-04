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

#include <gtest/gtest.h>
#include <xvigra/global.hpp>
#include <limits>
#include <type_traits>

namespace xvigra 
{
    TEST(global, types)
    {
        EXPECT_TRUE(std::is_integral<index_t>::value);
        EXPECT_TRUE(std::is_signed<index_t>::value);
        EXPECT_EQ(sizeof(index_t), sizeof(std::size_t));
    }

    TEST(global, rebind_container)
    {
        EXPECT_TRUE((std::is_same<rebind_container_t<std::vector<int>, double>, std::vector<double>>::value));
        EXPECT_TRUE((std::is_same<rebind_container_t<std::array<int,2>, double>, std::array<double,2>>::value));
        EXPECT_TRUE((std::is_same<rebind_container_t<xt::xarray<int>, double>, xt::xarray<double>>::value));
        EXPECT_TRUE((std::is_same<rebind_container_t<xt::xtensor<int,2>, double>, xt::xtensor<double,2>>::value));
    }

} // namespace xvigra
