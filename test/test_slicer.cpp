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
#include <xvigra/slicer.hpp>

namespace xvigra
{
    TEST(slicer, c_order)
    {
        xt::dynamic_shape<std::size_t> shape{2,3,4};
        slicer nav1(shape), nav2(shape);

        nav1.set_free_axes(0);
        nav2.set_iterate_axes(1, 2);
        for(index_t k=0; k<shape[1]; ++k)
        {
            for(index_t i=0; i<shape[2]; ++i, ++nav1, nav2++)
            {
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav1)[0]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[1]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[2]), i);
                EXPECT_TRUE(nav1.has_more());
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav2)[0]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[1]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[2]), i);
                EXPECT_TRUE(nav2.has_more());
            }
        }
        EXPECT_FALSE(nav1.has_more());
        EXPECT_FALSE(nav2.has_more());

        nav1.set_free_axes(1);
        nav2.set_iterate_axes(0, 2);
        for(index_t k=0; k<shape[0]; ++k)
        {
            for(index_t i=0; i<shape[2]; ++i, ++nav1, ++nav2)
            {
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[0]), k);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav1)[1]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[2]), i);
                EXPECT_TRUE(nav1.has_more());
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[0]), k);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav2)[1]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[2]), i);
                EXPECT_TRUE(nav2.has_more());
            }
        }
        EXPECT_FALSE(nav1.has_more());
        EXPECT_FALSE(nav2.has_more());

        nav1.set_free_axes(2);
        nav2.set_iterate_axes(0, 1);
        for(index_t k=0; k<shape[0]; ++k)
        {
            for(index_t i=0; i<shape[1]; ++i, ++nav1, ++nav2)
            {
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[0]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[1]), i);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav1)[2]), nullptr);
                EXPECT_TRUE(nav1.has_more());
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[0]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[1]), i);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav2)[2]), nullptr);
                EXPECT_TRUE(nav2.has_more());
            }
        }
        EXPECT_FALSE(nav1.has_more());
        EXPECT_FALSE(nav2.has_more());
    }

    TEST(slicer, f_order)
    {
        xt::dynamic_shape<std::size_t> shape{2,3,4};
        slicer nav1(shape, tags::f_order), nav2(shape, tags::f_order);

        nav1.set_free_axes(0);
        nav2.set_iterate_axes(1, 2);
        for(index_t i=0; i<shape[2]; ++i)
        {
            for(index_t k=0; k<shape[1]; ++k, ++nav1, nav2++)
            {
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav1)[0]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[1]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[2]), i);
                EXPECT_TRUE(nav1.has_more());
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav2)[0]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[1]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[2]), i);
                EXPECT_TRUE(nav2.has_more());
            }
        }
        EXPECT_FALSE(nav1.has_more());
        EXPECT_FALSE(nav2.has_more());

        nav1.set_free_axes(1);
        nav2.set_iterate_axes(0, 2);
        for(index_t i=0; i<shape[2]; ++i)
        {
            for(index_t k=0; k<shape[0]; ++k, ++nav1, ++nav2)
            {
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[0]), k);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav1)[1]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[2]), i);
                EXPECT_TRUE(nav1.has_more());
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[0]), k);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav2)[1]), nullptr);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[2]), i);
                EXPECT_TRUE(nav2.has_more());
            }
        }
        EXPECT_FALSE(nav1.has_more());
        EXPECT_FALSE(nav2.has_more());

        nav1.set_free_axes(2);
        nav2.set_iterate_axes(0, 1);
        for(index_t i=0; i<shape[1]; ++i)
        {
            for(index_t k=0; k<shape[0]; ++k, ++nav1, ++nav2)
            {
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[0]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav1)[1]), i);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav1)[2]), nullptr);
                EXPECT_TRUE(nav1.has_more());
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[0]), k);
                EXPECT_EQ(*xtl::get_if<int>(&(*nav2)[1]), i);
                EXPECT_NE(xtl::get_if<xt::xall_tag>(&(*nav2)[2]), nullptr);
                EXPECT_TRUE(nav2.has_more());
            }
        }
        EXPECT_FALSE(nav1.has_more());
        EXPECT_FALSE(nav2.has_more());
    }
} // namespace xvigra
