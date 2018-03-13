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
#include <xvigra/gaussian.hpp>

namespace xvigra
{
    TEST(gaussian, values)
    {
        gaussian<double> g,
                         g1(2.0, 1),
                         g2(1.0, 2),
                         g3(2.0, 3),
                         g4(2.0, 4),
                         g5(2.0, 5);

        double epsilon = 1e-15;
        EXPECT_EQ(g.derivative_order(), 0u);
        EXPECT_EQ(g.sigma(), 1.0);
        EXPECT_NEAR(g(0.0), 0.3989422804014327, epsilon);
        EXPECT_NEAR(g(0.5), 0.35206532676429952, epsilon);
        EXPECT_NEAR(g(1.0), 0.24197072451914337, epsilon);
        EXPECT_NEAR(g(-1.0), 0.24197072451914337, epsilon);

        EXPECT_EQ(g1.derivative_order(), 1u);
        EXPECT_EQ(g1.sigma(), 2.0);
        EXPECT_NEAR(g1(0.0), 0, epsilon);
        EXPECT_NEAR(g1(0.5), -0.024166757300178077, epsilon);
        EXPECT_NEAR(g1(1.0), -0.044008165845537441, epsilon);
        EXPECT_NEAR(g1(-1.0), 0.044008165845537441, epsilon);

        EXPECT_EQ(g2.derivative_order(), 2u);
        EXPECT_EQ(g2.sigma(), 1.0);
        EXPECT_NEAR(g2(0.0), -0.3989422804014327, epsilon);
        EXPECT_NEAR(g2(0.5), -0.26404899507322466, epsilon);
        EXPECT_NEAR(g2(1.0), 0, epsilon);
        EXPECT_NEAR(g2(-1.0), 0, epsilon);
        EXPECT_NEAR(g2(1.5), 0.16189699458236467, epsilon);
        EXPECT_NEAR(g2(-1.5), 0.16189699458236467, epsilon);

        EXPECT_EQ(g3.derivative_order(), 3u);
        EXPECT_EQ(g3.sigma(), 2.0);
        EXPECT_NEAR(g3(0.0), 0, epsilon);
        EXPECT_NEAR(g3(0.5), 0.017747462392318277, epsilon);
        EXPECT_NEAR(g3(1.0), 0.030255614018806987, epsilon);
        EXPECT_NEAR(g3(-1.0), -0.030255614018806987, epsilon);
        EXPECT_NEAR(g3(2.0*std::sqrt(3.0)), 0, epsilon);
        EXPECT_NEAR(g3(-2.0*std::sqrt(3.0)), 0, epsilon);

        EXPECT_NEAR(g4(0.0), 0.037400838787634318, epsilon);
        EXPECT_NEAR(g4(1.0), 0.017190689783413062, epsilon);
        EXPECT_NEAR(g4(-1.0), 0.017190689783413062, epsilon);
        EXPECT_NEAR(g4(1.483927568605452), 0, epsilon);
        EXPECT_NEAR(g4(4.668828436677955), 0, epsilon);
        EXPECT_NEAR(g5(0.0), 0, epsilon);
        EXPECT_NEAR(g5(1.0), -0.034553286464660257, epsilon);
        EXPECT_NEAR(g5(-1.0), 0.034553286464660257, epsilon);
        EXPECT_NEAR(g5(2.711252359948531), 0, epsilon);
        EXPECT_NEAR(g5(5.713940027745611), 0, epsilon);
    }
} // namespace xvigra
