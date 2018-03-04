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
#include <xvigra/slicer.hpp>

namespace xvigra 
{
    TEST(slicer, c_order)
    {
        xt::dynamic_shape<std::size_t> shape{2,3,4};
        {
            slicer nav(shape, 0);
            for(index_t k=0; k<shape[1]; ++k)
            {
                for(index_t i=0; i<shape[2]; ++i, ++nav)
                {
                    EXPECT_EQ((*nav)[0][0], 0);
                    EXPECT_EQ((*nav)[0][1], shape[0]);
                    EXPECT_EQ((*nav)[1][0], k);
                    EXPECT_EQ((*nav)[1][1], 0);
                    EXPECT_EQ((*nav)[2][0], i);
                    EXPECT_EQ((*nav)[2][1], 0);
                    EXPECT_TRUE(nav.has_more());
                }
            }
            EXPECT_FALSE(nav.has_more());
        }
        {
            slicer nav(shape, 1);
            for(index_t k=0; k<shape[0]; ++k)
            {
                for(index_t i=0; i<shape[2]; ++i, ++nav)
                {
                    EXPECT_EQ((*nav)[0][0], k);
                    EXPECT_EQ((*nav)[0][1], 0);
                    EXPECT_EQ((*nav)[1][0], 0);
                    EXPECT_EQ((*nav)[1][1], shape[1]);
                    EXPECT_EQ((*nav)[2][0], i);
                    EXPECT_EQ((*nav)[2][1], 0);
                    EXPECT_TRUE(nav.has_more());
                }
            }
            EXPECT_FALSE(nav.has_more());
        }
        {
            slicer nav(shape, 2);
            for(index_t k=0; k<shape[0]; ++k)
            {
                for(index_t i=0; i<shape[1]; ++i, ++nav)
                {
                    EXPECT_EQ((*nav)[0][0], k);
                    EXPECT_EQ((*nav)[0][1], 0);
                    EXPECT_EQ((*nav)[1][0], i);
                    EXPECT_EQ((*nav)[1][1], 0);
                    EXPECT_EQ((*nav)[2][0], 0);
                    EXPECT_EQ((*nav)[2][1], shape[2]);
                    EXPECT_TRUE(nav.has_more());
                }
            }
            EXPECT_FALSE(nav.has_more());
        }
    }
} // namespace xvigra
