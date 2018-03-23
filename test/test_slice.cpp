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
#include <xvigra/slice.hpp>

namespace xvigra
{
    TEST(slice, parsing)
    {
        using S = shape_t<>;
        using R = std::runtime_error;
        using namespace slicing;
        shape_t<3> old_shape{4,3,2}, old_strides = shape_to_strides(old_shape), point(old_shape.size());
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides);
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, old_shape);
            EXPECT_EQ(strides, old_strides);
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, all(), all(), all());
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, old_shape);
            EXPECT_EQ(strides, old_strides);
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, ellipsis(), all());
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, old_shape);
            EXPECT_EQ(strides, old_strides);
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(), ellipsis(), all());
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, old_shape);
            EXPECT_EQ(strides, old_strides);
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, 1);
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, old_shape.erase(0));
            EXPECT_EQ(strides, old_strides.erase(0));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, all(), 1);
            EXPECT_EQ(point, (S{0,1,0}));
            EXPECT_EQ(shape, old_shape.erase(1));
            EXPECT_EQ(strides, old_strides.erase(1));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, ellipsis(), 1);
            EXPECT_EQ(point, (S{0,0,1}));
            EXPECT_EQ(shape, old_shape.erase(2));
            EXPECT_EQ(strides, old_strides.erase(2));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, ellipsis(), 1, newaxis());
            EXPECT_EQ(point, (S{0,0,1}));
            EXPECT_EQ(shape, old_shape.erase(2).push_back(1));
            EXPECT_EQ(strides, old_strides.erase(2).push_back(0));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, all(), 1, newaxis());
            EXPECT_EQ(point, (S{0,1,0}));
            EXPECT_EQ(shape, old_shape.erase(1).insert(1, 1));
            EXPECT_EQ(strides, old_strides.erase(1).insert(1, 0));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, all(), -1, 1);
            EXPECT_EQ(point, (S{0,2,1}));
            EXPECT_EQ(shape, old_shape.erase(2).erase(1));
            EXPECT_EQ(strides, old_strides.erase(2).erase(1));
        }
        // slices with positive step
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{3,3,2}));
            EXPECT_EQ(strides, old_strides);
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1, 3));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, old_strides);
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(_, 3));
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, (S{3,3,2}));
            EXPECT_EQ(strides, old_strides);
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(_, _, 2));
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{2,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(_, _, 3));
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1, _, 3));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{1,3,2}));
            EXPECT_EQ(strides, (old_strides*S{3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(0, 4, 3));
            EXPECT_EQ(point, (S{0,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1, 4, 3));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{1,3,2}));
            EXPECT_EQ(strides, (old_strides*S{3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1, 3, 3));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{1,3,2}));
            EXPECT_EQ(strides, (old_strides*S{3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1, 1, 3));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{0,3,2}));
            EXPECT_EQ(strides, old_strides); // numpy compatibility
        }
        // slices with negative step
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1,_,-1));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-1,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(3, 1, -1));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-1,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(3, _, -1));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{4,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-1,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(_, _, -2));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-2,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(_, _, -3));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(_, 1, -3));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{1,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(3, -5, -3));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{2,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(3, 0, -3));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{1,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(3, 1, -3));
            EXPECT_EQ(point, (S{3,0,0}));
            EXPECT_EQ(shape, (S{1,3,2}));
            EXPECT_EQ(strides, (old_strides*S{-3,1,1}));
        }
        {
            shape_t<> shape, strides;
            detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(1, 1, -3));
            EXPECT_EQ(point, (S{1,0,0}));
            EXPECT_EQ(shape, (S{0,3,2}));
            EXPECT_EQ(strides, old_strides); // numpy compatibility
        }
        // errors
        {
            shape_t<> shape, strides;
            // index too big
            EXPECT_THROW(detail::parse_slices(point, shape, strides, old_shape, old_strides, 4, 2, 1), R);
            // index too small
            EXPECT_THROW(detail::parse_slices(point, shape, strides, old_shape, old_strides, 0, 2, -3), R);
            // too many indices
            EXPECT_THROW(detail::parse_slices(point, shape, strides, old_shape, old_strides, all(), -1, 1, 0), R);
            // multiple ellipses
            EXPECT_THROW(detail::parse_slices(point, shape, strides, old_shape, old_strides, ellipsis(), 1, ellipsis()), R);
            EXPECT_THROW(detail::parse_slices(point, shape, strides, old_shape, old_strides, ellipsis(), all(), ellipsis()), R);
            EXPECT_THROW(detail::parse_slices(point, shape, strides, old_shape, old_strides, ellipsis(), newaxis(), ellipsis()), R);
            // zero step
            EXPECT_THROW(detail::parse_slices(point, shape, strides, old_shape, old_strides, slice(_,_,0)), R);
        }
   }

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
        slicer nav1(shape, f_order), nav2(shape, f_order);

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
