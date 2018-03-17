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

#ifndef XVIGRA_UNITTEST_HPP
#define XVIGRA_UNITTEST_HPP

#include <xtensor/xio.hpp>

// #define XVIGRA_USE_DOCTEST

#ifndef XVIGRA_USE_DOCTEST

    #include <gtest/gtest.h>

    #define TYPED_TEST_SETUP(ID, TYPES)        \
    template <class T>                             \
    class ID : public testing::Test                \
    {};                                            \
    TYPED_TEST_CASE(ID, TYPES)

#else

    #include <doctest.h>

    namespace testing = doctest;

    #define TEST(A, B)            TEST_CASE(#A "." #B)

    #define EXPECT_TRUE(A)        CHECK(A)
    #define EXPECT_FALSE(A)       CHECK_FALSE(A)

    #define EXPECT_EQ(A, B)       CHECK_EQ(A, B)
    #define EXPECT_NE(A, B)       CHECK_NE(A, B)
    #define EXPECT_LT(A, B)       CHECK_LT(A, B)
    #define EXPECT_LE(A, B)       CHECK_LE(A, B)
    #define EXPECT_GT(A, B)       CHECK_GT(A, B)
    #define EXPECT_GE(A, B)       CHECK_GE(A, B)
    #define EXPECT_NEAR(A, B, C)  CHECK(A == doctest::Approx(B).epsilon(C))

    #define EXPECT_THROW(A, B)    CHECK_THROWS_AS(A, B)

    #define TYPED_TEST_SETUP(ID, TYPES)             \
    using ID = TYPES

    #define TYPED_TEST(ID, NAME)                    \
    TEST_CASE_TEMPLATE(#ID "." #NAME, TypeParam, ID)

#endif

#endif // XVIGRA_UNITTEST_HPP
